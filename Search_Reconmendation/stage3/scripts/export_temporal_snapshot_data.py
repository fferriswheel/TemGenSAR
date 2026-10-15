#!/usr/bin/env python3
"""Export all revealed Search/Rec feedback up to a global time cutoff."""

import argparse
import json
import shutil
from pathlib import Path


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--end-exclusive", type=int, required=True)
    parser.add_argument("--rec-end-exclusive", type=int, default=0)
    parser.add_argument("--search-end-exclusive", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("items.jsonl", "item_ids.jsonl", "queries.jsonl", "users.jsonl", "stats.json"):
        shutil.copy2(args.source_dir / name, args.output_dir / name)
    rec_end = args.rec_end_exclusive or args.end_exclusive
    search_end = args.search_end_exclusive or args.end_exclusive
    rec = [
        row for row in read_jsonl(args.source_dir / "rec_interactions_chrono.jsonl")
        if int(row["timestamp"]) < rec_end
    ]
    search = [
        row for row in read_jsonl(args.source_dir / "search_interactions.jsonl")
        if int(row["timestamp"]) < search_end
    ]
    rec.sort(key=lambda row: (row["user_id"], int(row["timestamp"]), int(row.get("position", 0))))
    search.sort(key=lambda row: (row["user_id"], int(row["timestamp"]), str(row["interaction_id"])))
    write_jsonl(args.output_dir / "rec_interactions_chrono.jsonl", rec)
    write_jsonl(args.output_dir / "search_interactions.jsonl", search)
    summary = {
        "source_dir": str(args.source_dir), "end_exclusive": args.end_exclusive,
        "rec_end_exclusive": rec_end, "search_end_exclusive": search_end,
        "policy": "all revealed behavior before the global cutoff",
        "rec_rows": len(rec), "search_rows": len(search),
        "rec_users": len({row["user_id"] for row in rec}),
        "search_users": len({row["user_id"] for row in search}),
        "rec_items": len({row["item_id"] for row in rec}),
        "search_items": len({row["item_id"] for row in search}),
    }
    (args.output_dir / "temporal_snapshot_stats.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
