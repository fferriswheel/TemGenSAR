#!/usr/bin/env python3
"""Build a Rec-adapter dataset from changed-item-affected rows plus replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--base-mapping", type=Path, required=True)
    parser.add_argument("--current-mapping", type=Path, required=True)
    parser.add_argument("--id-maps", type=Path, required=True)
    parser.add_argument("--replay-size", type=int, default=4000)
    parser.add_argument("--max-rec-history", type=int, default=5)
    parser.add_argument(
        "--keep-all-train-rows",
        action="store_true",
        help="Keep the complete update set and only annotate rows for adaptive distillation.",
    )
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.source_dataset.resolve() == args.output_dataset.resolve():
        raise ValueError("source and output datasets must differ")

    base = json.loads(args.base_mapping.read_text(encoding="utf-8"))
    current = json.loads(args.current_mapping.read_text(encoding="utf-8"))
    if set(base) != set(current):
        raise ValueError("base/current item universes differ")
    recoded_asins = {
        item_id
        for item_id in base
        if base[item_id]["rec_suffix_codes"] != current[item_id]["rec_suffix_codes"]
    }
    id_maps = json.loads(args.id_maps.read_text(encoding="utf-8"))
    external_to_int = id_maps.get("asin_to_int", id_maps.get("item_to_int"))
    if external_to_int is None:
        raise ValueError("id maps must contain asin_to_int or item_to_int")
    external_to_int = {str(key): int(value) for key, value in external_to_int.items()}
    missing = recoded_asins - set(external_to_int)
    if missing:
        raise ValueError(f"recoded ASINs absent from id maps: {sorted(missing)[:5]}")
    recoded_items = {external_to_int[item_id] for item_id in recoded_asins}

    source_train = args.source_dataset / "dataset/rec_train.pkl"
    rec_train: pd.DataFrame = pd.read_pickle(source_train)
    with (args.source_dataset / "vocab/user_vocab.pkl").open("rb") as handle:
        user_vocab = pickle.load(handle)

    target_changed = np.zeros(len(rec_train), dtype=bool)
    history_changed = np.zeros(len(rec_train), dtype=bool)
    for position, row in enumerate(rec_train.itertuples(index=False)):
        target_changed[position] = int(row.item_id) in recoded_items
        user = user_vocab[int(row.user_id)]
        history = user["rec_his"][: int(row.rec_his)][-args.max_rec_history :]
        history_changed[position] = bool(recoded_items.intersection(map(int, history)))
    affected = target_changed | history_changed
    unaffected_indices = np.flatnonzero(~affected)
    if args.replay_size > len(unaffected_indices):
        raise ValueError("replay size exceeds unaffected rows")
    rng = np.random.default_rng(args.seed)
    replay_indices = rng.choice(unaffected_indices, size=args.replay_size, replace=False)
    if args.keep_all_train_rows:
        selected_indices = np.arange(len(rec_train))
    else:
        selected_indices = np.sort(np.concatenate([np.flatnonzero(affected), replay_indices]))
    output_train = rec_train.iloc[selected_indices].copy().reset_index(drop=True)
    output_train["_distill_weight"] = (~affected[selected_indices]).astype(np.float32)

    src_train: pd.DataFrame = pd.read_pickle(args.source_dataset / "dataset/src_train.pkl")
    src_history_changed = np.zeros(len(src_train), dtype=bool)
    for position, row in enumerate(src_train.itertuples(index=False)):
        user = user_vocab[int(row.user_id)]
        history = user["rec_his"][: int(row.rec_his)][-args.max_rec_history :]
        src_history_changed[position] = bool(recoded_items.intersection(map(int, history)))
    output_src_train = src_train.copy()
    output_src_train["_distill_weight"] = (~src_history_changed).astype(np.float32)

    if args.output_dataset.exists():
        shutil.rmtree(args.output_dataset)
    shutil.copytree(args.source_dataset, args.output_dataset)
    output_path = args.output_dataset / "dataset/rec_train.pkl"
    output_train.to_pickle(output_path)
    output_src_path = args.output_dataset / "dataset/src_train.pkl"
    output_src_train.to_pickle(output_src_path)

    untouched = [
        "dataset/rec_val.pkl",
        "dataset/src_val.pkl",
        "dataset/rec_test.pkl",
        "dataset/src_test.pkl",
    ]
    for relative in untouched:
        if sha256(args.source_dataset / relative) != sha256(args.output_dataset / relative):
            raise RuntimeError(f"copied protocol file changed: {relative}")

    manifest = {
        "method": "affected Rec rows plus unchanged replay for task-local adapter",
        "source_dataset": str(args.source_dataset),
        "base_mapping": str(args.base_mapping),
        "current_mapping": str(args.current_mapping),
        "seed": args.seed,
        "max_rec_history": args.max_rec_history,
        "recoded_items": len(recoded_items),
        "source_rec_rows": len(rec_train),
        "target_changed_rows": int(target_changed.sum()),
        "history_changed_rows": int(history_changed.sum()),
        "affected_union_rows": int(affected.sum()),
        "replay_rows": args.replay_size,
        "output_rec_rows": len(output_train),
        "keep_all_train_rows": args.keep_all_train_rows,
        "rec_distill_rows": int((~affected[selected_indices]).sum()),
        "search_rows": len(src_train),
        "search_history_changed_rows": int(src_history_changed.sum()),
        "search_distill_rows": int((~src_history_changed).sum()),
        "source_rec_train_sha256": sha256(source_train),
        "output_rec_train_sha256": sha256(output_path),
        "untouched_protocol_files": untouched,
    }
    (args.output_dataset / "affected_rec_adapter_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
