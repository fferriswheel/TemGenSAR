#!/usr/bin/env python3
"""Split a KuaiSAR incremental dataset into W3 train and leakage-free Rec gate."""

import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=4)


def link(source, target):
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)


def attach_negatives(frame, user_vocab, src_session_vocab, num_items, count, seed):
    rng = random.Random(seed)
    rows = []
    for line in frame.to_dict("records"):
        user = int(line["user_id"])
        history = user_vocab[user]
        excluded = set(history["rec_his"][: int(line["rec_his"])])
        for session_id in history["src_session_his"][: int(line["src_session_his"])]:
            excluded.update(src_session_vocab[session_id]["pos_items"])
        excluded.add(int(line["item_id"]))
        negatives = set()
        while len(negatives) < count:
            candidate = rng.randint(1, num_items)
            if candidate not in excluded:
                negatives.add(candidate)
        line["neg_items"] = list(negatives)
        rng.shuffle(line["neg_items"])
        rows.append(line)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-gate-users", type=int, default=5000)
    parser.add_argument("--num-negatives", type=int, default=99)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    source = args.source_dataset.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rec = load_pickle(source / "dataset/rec_train.pkl").reset_index(drop=True)

    by_user = defaultdict(list)
    for index, row in rec.iterrows():
        by_user[int(row["user_id"])].append((int(row["ts"]), index))
    eligible = sorted(user for user, rows in by_user.items() if len(rows) >= 2)
    if len(eligible) > args.max_gate_users:
        eligible = sorted(random.Random(args.seed).sample(eligible, args.max_gate_users))
    gate_indices = {max(by_user[user])[1] for user in eligible}
    train = rec.loc[~rec.index.isin(gate_indices)].reset_index(drop=True)
    gate = rec.loc[sorted(gate_indices)].sort_values(["ts", "user_id"]).reset_index(drop=True)

    user_vocab = load_pickle(source / "vocab/user_vocab.pkl")
    src_session_vocab = load_pickle(source / "vocab/src_session_vocab.pkl")
    item_vocab = load_pickle(source / "vocab/item_vocab.pkl")
    gate = attach_negatives(
        gate,
        user_vocab,
        src_session_vocab,
        len(item_vocab) - 1,
        args.num_negatives,
        args.seed + 1,
    )
    dump_pickle(output / "dataset/rec_train.pkl", train)
    dump_pickle(output / "dataset/rec_val.pkl", gate)
    dump_pickle(output / "dataset/rec_test.pkl", gate)

    for name in ["src_train.pkl", "src_val.pkl", "src_test.pkl"]:
        path = source / "dataset" / name
        if path.exists():
            link(path, output / "dataset" / name)
    for path in (source / "vocab").glob("*.pkl"):
        link(path, output / "vocab" / path.name)
    for path in (source / "emb").glob("*.json"):
        link(path, output / "emb" / path.name)

    manifest = {
        "source_dataset": str(source),
        "split": "latest W3 Rec row per sampled user",
        "seed": args.seed,
        "original_rec_rows": len(rec),
        "rec_train_rows": len(train),
        "rec_gate_rows": len(gate),
        "gate_users": len(eligible),
        "num_negatives": args.num_negatives,
        "gate_min_ts": int(gate["ts"].min()),
        "gate_max_ts": int(gate["ts"].max()),
    }
    (output / "utility_gate_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
