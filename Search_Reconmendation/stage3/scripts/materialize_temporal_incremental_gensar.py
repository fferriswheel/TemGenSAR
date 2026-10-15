#!/usr/bin/env python3
"""Build a GenSAR view for W1 incremental training and W2 evaluation."""

import argparse
import json
import pickle
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quarterly-dir", type=Path, required=True)
    p.add_argument("--update-window", default="W1")
    p.add_argument(
        "--val-window",
        help="Window used for validation; defaults to the update window.",
    )
    p.add_argument("--eval-window", default="W2")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--semantic-sid", type=Path, required=True)
    p.add_argument("--collaborate-sid", type=Path, required=True)
    p.add_argument("--base-mapping", type=Path, required=True)
    p.add_argument("--current-mapping", type=Path, required=True)
    p.add_argument("--item-ids", type=Path, required=True)
    p.add_argument("--update-task", choices=["rec", "search", "both"], default="rec")
    p.add_argument("--rec-replay-size", type=int, default=1000)
    p.add_argument("--search-replay-size", type=int, default=1000)
    p.add_argument("--use-search-update", action="store_true")
    p.add_argument("--all-items-for-alignment", action="store_true")
    p.add_argument(
        "--alignment-items",
        type=Path,
        help="Optional JSON list of ASINs shared by all experimental arms.",
    )
    p.add_argument(
        "--search-alignment-items",
        type=Path,
        help="Optional Search-specific JSON ASIN list for semantic SID alignment.",
    )
    p.add_argument(
        "--rec-alignment-items",
        type=Path,
        help="Optional Rec-specific JSON ASIN list for collaborative SID alignment.",
    )
    p.add_argument(
        "--balance-update-tasks",
        action="store_true",
        help="Oversample the smaller update task to the larger task's row count.",
    )
    p.add_argument(
        "--balanced-alignment",
        action="store_true",
        help="Cycle evenly over task-local alignment items instead of random sampling.",
    )
    p.add_argument(
        "--preserve-alignment-duplicates",
        action="store_true",
        help="Keep duplicate entries in task-specific alignment JSON schedules.",
    )
    p.add_argument("--seed", type=int, default=2024)
    args = p.parse_args()

    root = args.quarterly_dir.resolve()
    update = root / "windows" / args.update_window
    validation = root / "windows" / (args.val_window or args.update_window)
    evaluation = root / "windows" / args.eval_window
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rec_window = load_pickle(update / "rec_update.pkl").reset_index(drop=True)
    src_window = load_pickle(update / "src_update.pkl").reset_index(drop=True)
    rec_base = load_pickle(root / "dataset" / "rec_train.pkl")
    src_base = load_pickle(root / "dataset" / "src_train.pkl")
    if args.update_task in {"rec", "both"}:
        rec_update = rec_window
    else:
        rec_update = rec_base.sample(
            n=min(args.rec_replay_size, len(rec_base)), random_state=args.seed
        ).reset_index(drop=True)
    if args.update_task in {"search", "both"} or args.use_search_update:
        src_replay = src_window
    else:
        src_replay = src_base.sample(
            n=min(args.search_replay_size, len(src_base)), random_state=args.seed
        ).reset_index(drop=True)
    raw_rec_rows = len(rec_update)
    raw_search_rows = len(src_replay)
    if args.balance_update_tasks and args.update_task == "both":
        target_rows = max(raw_rec_rows, raw_search_rows)
        if raw_rec_rows < target_rows:
            rec_update = rec_update.sample(
                n=target_rows, replace=True, random_state=args.seed
            ).reset_index(drop=True)
        if raw_search_rows < target_rows:
            src_replay = src_replay.sample(
                n=target_rows, replace=True, random_state=args.seed
            ).reset_index(drop=True)
    dump_pickle(output / "dataset/rec_train.pkl", rec_update)
    dump_pickle(output / "dataset/src_train.pkl", src_replay)
    link(validation / "rec_eval.pkl", output / "dataset/rec_val.pkl")
    link(validation / "src_eval.pkl", output / "dataset/src_val.pkl")
    link(evaluation / "rec_eval.pkl", output / "dataset/rec_test.pkl")
    link(evaluation / "src_eval.pkl", output / "dataset/src_test.pkl")

    item_ids = [json.loads(line)["item_id"] for line in args.item_ids.read_text().splitlines()]
    actual_changed = []
    rec_sid_migrations = []
    base = json.loads(args.base_mapping.read_text(encoding="utf-8"))
    current = json.loads(args.current_mapping.read_text(encoding="utf-8"))
    for idx, item_id in enumerate(item_ids, start=1):
        rec_changed = base[item_id]["rec_sid"] != current[item_id]["rec_sid"]
        search_changed = base[item_id]["search_sid"] != current[item_id]["search_sid"]
        relevant = {
            "rec": rec_changed,
            "search": search_changed,
            "both": rec_changed or search_changed,
        }[args.update_task]
        if relevant:
            actual_changed.append(idx)
        if rec_changed:
            rec_sid_migrations.append(
                {
                    "item_id": idx,
                    "asin": item_id,
                    "old_sid": base[item_id]["rec_sid"],
                    "new_sid": current[item_id]["rec_sid"],
                }
            )
    dump_pickle(output / "vocab/rec_sid_migrations.pkl", rec_sid_migrations)
    task_specific_alignment = bool(
        args.search_alignment_items or args.rec_alignment_items
    )
    if task_specific_alignment:
        if not (args.search_alignment_items and args.rec_alignment_items):
            raise ValueError(
                "--search-alignment-items and --rec-alignment-items must be used together"
            )
        if args.alignment_items or args.all_items_for_alignment:
            raise ValueError("Task-specific alignment cannot be combined with shared alignment")

        def load_alignment(path):
            raw_asins = [str(value) for value in json.loads(path.read_text(encoding="utf-8"))]
            asins = raw_asins if args.preserve_alignment_duplicates else sorted(set(raw_asins))
            unknown = set(asins) - set(item_ids)
            if unknown:
                raise ValueError(f"Unknown alignment items: {sorted(unknown)[:5]}")
            if args.preserve_alignment_duplicates:
                item_to_index = {item_id: idx for idx, item_id in enumerate(item_ids, start=1)}
                return [item_to_index[item_id] for item_id in asins]
            return [idx for idx, item_id in enumerate(item_ids, start=1) if item_id in set(asins)]

        search_changed = load_alignment(args.search_alignment_items)
        rec_changed = load_alignment(args.rec_alignment_items)
        changed = sorted(set(search_changed) | set(rec_changed))
    elif args.alignment_items:
        alignment_asins = set(json.loads(args.alignment_items.read_text(encoding="utf-8")))
        unknown = alignment_asins - set(item_ids)
        if unknown:
            raise ValueError(f"Unknown alignment items: {sorted(unknown)[:5]}")
        changed = [idx for idx, item_id in enumerate(item_ids, start=1) if item_id in alignment_asins]
    elif args.all_items_for_alignment:
        changed = list(range(1, len(item_ids) + 1))
    else:
        changed = actual_changed
    if task_specific_alignment:
        dump_pickle(output / "vocab/rec_item_set.pkl", rec_changed)
        dump_pickle(output / "vocab/src_item_set.pkl", search_changed)
        dump_pickle(output / "vocab/task_specific_alignment.pkl", True)
    else:
        dump_pickle(output / "vocab/rec_item_set.pkl", changed)
        dump_pickle(output / "vocab/src_item_set.pkl", changed)
    if args.balanced_alignment:
        dump_pickle(output / "vocab/balanced_alignment.pkl", True)
    for path in (root / "vocab").glob("*.pkl"):
        if path.name not in {"rec_item_set.pkl", "src_item_set.pkl"}:
            link(path, output / "vocab" / path.name)
    link(args.semantic_sid, output / "emb" / args.semantic_sid.name)
    link(args.collaborate_sid, output / "emb" / args.collaborate_sid.name)

    manifest = {
        "update_window": args.update_window,
        "val_window": args.val_window or args.update_window,
        "eval_window": args.eval_window,
        "update_task": args.update_task,
        "rec_update_rows": len(rec_update),
        "search_update_rows": len(src_replay),
        "raw_rec_update_rows": raw_rec_rows,
        "raw_search_update_rows": raw_search_rows,
        "balanced_update_tasks": args.balance_update_tasks,
        "changed_items": len(changed),
        "actual_changed_items": len(actual_changed),
        "rec_sid_migrations": len(rec_sid_migrations),
        "alignment_items": len(changed),
        "search_alignment_items": len(search_changed) if task_specific_alignment else len(changed),
        "rec_alignment_items": len(rec_changed) if task_specific_alignment else len(changed),
        "task_specific_alignment": task_specific_alignment,
        "balanced_alignment": args.balanced_alignment,
        "alignment_duplicates_preserved": args.preserve_alignment_duplicates,
        "search_alignment_unique_items": len(set(search_changed)) if task_specific_alignment else len(changed),
        "rec_alignment_unique_items": len(set(rec_changed)) if task_specific_alignment else len(changed),
        "alignment_items_source": str(args.alignment_items) if args.alignment_items else "mapping_diff",
        # Backward-compatible names retained for earlier Rec-only reports.
        "search_replay_rows": len(src_replay),
        "changed_rec_items": len(changed) if args.update_task == "rec" else 0,
        "rec_val_rows": len(load_pickle(validation / "rec_eval.pkl")),
        "search_val_rows": len(load_pickle(validation / "src_eval.pkl")),
        "rec_test_rows": len(load_pickle(evaluation / "rec_eval.pkl")),
        "search_test_rows": len(load_pickle(evaluation / "src_eval.pkl")),
    }
    (output / "incremental_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
