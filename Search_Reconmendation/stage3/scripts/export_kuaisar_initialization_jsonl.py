#!/usr/bin/env python3
"""Export a task-windowed KuaiSAR snapshot to the canonical JSONL format."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd


def parse_token_ids(value) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [int(token) for token in value]
    text = str(value).strip()
    if not text or text == "[]":
        return []
    parsed = ast.literal_eval(text)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"expected token list, got {type(parsed).__name__}")
    return [int(token) for token in parsed]


def token_text(value) -> str:
    return " ".join(f"tok{token}" for token in parse_token_ids(value))


def write_jsonl(path: Path, rows) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    value = str(value).strip()
    return "" if value.lower() == "nan" or value == "empty" else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initialization-end-window", type=int, default=2)
    parser.add_argument("--rec-end-window", type=int)
    parser.add_argument("--search-end-window", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rec_end = args.rec_end_window if args.rec_end_window is not None else args.initialization_end_window
    search_end = (
        args.search_end_window
        if args.search_end_window is not None
        else args.initialization_end_window
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{args.output_dir} is not empty; pass --force to replace files")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = pd.read_parquet(args.source_dir / "items.parquet").sort_values("catalog_index")
    rec = pd.read_parquet(
        args.source_dir / "rec_positive_events.parquet",
        filters=[("window_id", "<=", rec_end)],
    ).sort_values(["user_id", "timestamp", "item_id"])
    search = pd.read_parquet(
        args.source_dir / "search_positive_events.parquet",
        filters=[("window_id", "<=", search_end)],
    ).sort_values(["user_id", "timestamp", "search_session_id", "item_id"])
    query_table = pd.read_parquet(args.source_dir / "queries.parquet")
    query_lookup = dict(zip(query_table["query_index"], query_table["keyword"]))
    used_queries = sorted(int(value) for value in search["query_index"].unique())

    def item_rows():
        for row in items.itertuples(index=False):
            categories = [
                clean(getattr(row, f"{level}_level_category_name_en"))
                for level in ["first", "second", "third", "fourth"]
            ]
            categories = [category for category in categories if category]
            caption = token_text(row.caption_tokens)
            yield {
                "item_id": str(int(row.item_id)),
                "title": " > ".join(categories),
                "description": caption,
                "brand": "",
                "categories": categories,
                "item_type": clean(row.item_type),
                "item_text": " ".join(part for part in [caption, *categories] if part),
            }

    def rec_rows():
        positions: dict[int, int] = {}
        for row in rec.itertuples(index=False):
            user = int(row.user_id)
            position = positions.get(user, 0)
            positions[user] = position + 1
            yield {
                "user_id": str(user),
                "item_id": str(int(row.item_id)),
                "timestamp": int(row.timestamp),
                "position": position,
                "window_id": int(row.window_id),
            }

    def query_rows():
        for query_id in used_queries:
            yield {
                "query_id": query_id,
                "query": token_text(query_lookup[query_id]),
                "raw_keyword": str(query_lookup[query_id]),
            }

    def search_rows():
        for idx, row in enumerate(search.itertuples(index=False), start=1):
            query = token_text(row.keyword)
            yield {
                "interaction_id": f"ks-init-{idx}",
                "user_id": str(int(row.user_id)),
                "item_id": str(int(row.item_id)),
                "query_id": int(row.query_index),
                "query": query,
                "timestamp": int(row.timestamp),
                "window_id": int(row.window_id),
                "search_session_id": int(row.search_session_id),
                "split": "train",
            }

    item_count = write_jsonl(args.output_dir / "items.jsonl", item_rows())
    write_jsonl(
        args.output_dir / "item_ids.jsonl",
        ({"item_id": str(int(item_id))} for item_id in items["item_id"]),
    )
    query_count = write_jsonl(args.output_dir / "queries.jsonl", query_rows())
    rec_count = write_jsonl(args.output_dir / "rec_interactions_chrono.jsonl", rec_rows())
    search_count = write_jsonl(args.output_dir / "search_interactions.jsonl", search_rows())
    users = sorted(set(rec["user_id"].astype(int)) | set(search["user_id"].astype(int)))
    write_jsonl(
        args.output_dir / "users.jsonl",
        ({"user_id": str(user)} for user in users),
    )
    summary = {
        "source_dir": str(args.source_dir.resolve()),
        "windows": {"rec": [0, rec_end], "search": [0, search_end]},
        "future_behavior_used": False,
        "items": item_count,
        "users": len(users),
        "queries": query_count,
        "rec_rows": rec_count,
        "search_rows": search_count,
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
