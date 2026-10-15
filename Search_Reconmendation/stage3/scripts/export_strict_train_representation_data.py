#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


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
    parser.add_argument("--gensar-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in ("items.jsonl", "item_ids.jsonl", "queries.jsonl", "users.jsonl", "stats.json"):
        shutil.copy2(args.source_dir / name, args.output_dir / name)

    maps = json.loads((args.gensar_dir / "id_maps.json").read_text(encoding="utf-8"))
    int_to_asin = {int(key): value for key, value in maps["int_to_asin"].items()}
    int_to_user = {int(value): key for key, value in maps["user_to_int"].items()}
    query_text = {
        int(row["query_id"]): row["query"]
        for row in read_jsonl(args.source_dir / "queries.jsonl")
    }

    rec = pd.read_pickle(args.gensar_dir / "dataset" / "rec_train.pkl")
    src = pd.read_pickle(args.gensar_dir / "dataset" / "src_train.pkl")

    rec_rows = []
    for row in rec.itertuples():
        rec_rows.append(
            {
                "interaction_id": f"strict_rec_{len(rec_rows)}",
                "user_id": int_to_user[int(row.user_id)],
                "item_id": int_to_asin[int(row.item_id)],
                "timestamp": int(row.ts),
                "position": int(row.rec_his),
                "split": "train",
            }
        )
    search_rows = []
    for row in src.itertuples():
        search_rows.append(
            {
                "interaction_id": f"strict_src_{len(search_rows)}",
                "user_id": int_to_user[int(row.user_id)],
                "item_id": int_to_asin[int(row.item_id)],
                "query_id": int(row.query_id),
                "query": query_text[int(row.query_id)],
                "timestamp": int(row.ts),
                "split": "train",
            }
        )

    rec_rows.sort(key=lambda row: (row["user_id"], row["timestamp"], row["position"]))
    search_rows.sort(key=lambda row: (row["user_id"], row["timestamp"], row["interaction_id"]))
    write_jsonl(args.output_dir / "rec_interactions_chrono.jsonl", rec_rows)
    write_jsonl(args.output_dir / "search_interactions.jsonl", search_rows)

    summary = {
        "source_gensar_dir": str(args.gensar_dir),
        "policy": "exact downstream strict train targets only",
        "rec_rows": len(rec_rows),
        "search_rows": len(search_rows),
        "rec_users": len({row["user_id"] for row in rec_rows}),
        "search_users": len({row["user_id"] for row in search_rows}),
    }
    (args.output_dir / "strict_train_stats.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
