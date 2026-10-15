#!/usr/bin/env python3
"""Build interaction files that exclude downstream temporal val/test events."""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def train_rows(rows):
    """Match prepare_gensar_lite per-user chronological split semantics."""
    rows = sorted(rows, key=lambda row: (int(row["timestamp"]), row.get("interaction_id", "")))
    if len(rows) >= 3:
        return rows[:-2]
    if len(rows) == 2:
        return rows[:1]
    return rows


def group_by_user(path):
    grouped = defaultdict(list)
    for row in read_jsonl(path):
        grouped[row["user_id"]].append(row)
    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in ["items.jsonl", "item_ids.jsonl", "queries.jsonl", "users.jsonl", "stats.json"]:
        shutil.copy2(args.source_dir / name, args.output_dir / name)

    rec_by_user = group_by_user(args.source_dir / "rec_interactions_chrono.jsonl")
    search_by_user = group_by_user(args.source_dir / "search_interactions.jsonl")

    rec_rows = []
    search_rows = []
    for user in sorted(rec_by_user):
        rec_rows.extend(train_rows(rec_by_user[user]))
    for user in sorted(search_by_user):
        for row in train_rows(search_by_user[user]):
            row = dict(row)
            row["source_split"] = row.get("split", "")
            row["split"] = "train"
            search_rows.append(row)

    rec_rows.sort(key=lambda row: (row["user_id"], int(row["timestamp"]), row.get("position", 0)))
    search_rows.sort(key=lambda row: (row["user_id"], int(row["timestamp"]), row.get("interaction_id", "")))
    rec_count = write_jsonl(args.output_dir / "rec_interactions_chrono.jsonl", rec_rows)
    search_count = write_jsonl(args.output_dir / "search_interactions.jsonl", search_rows)

    summary = {
        "source_dir": str(args.source_dir),
        "split_rule": "per-user chronological; >=3 keep all but last 2, 2 keep first, 1 keep only row",
        "rec": {
            "source_rows": sum(len(rows) for rows in rec_by_user.values()),
            "train_only_rows": rec_count,
            "removed_rows": sum(len(rows) for rows in rec_by_user.values()) - rec_count,
        },
        "search": {
            "source_rows": sum(len(rows) for rows in search_by_user.values()),
            "train_only_rows": search_count,
            "removed_rows": sum(len(rows) for rows in search_by_user.values()) - search_count,
        },
    }
    with (args.output_dir / "train_only_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
