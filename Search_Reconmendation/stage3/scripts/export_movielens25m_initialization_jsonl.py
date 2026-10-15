#!/usr/bin/env python3
"""Export a revealed MovieLens snapshot for representation and SID training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def write_jsonl(path: Path, rows) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in value]
    except TypeError:
        return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--through-window",
        type=int,
        default=0,
        help="Include Rec feedback through this window; Search remains the t0 control.",
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} is not empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = pd.read_parquet(args.source_dir / "items.parquet").sort_values("catalog_index")
    queries = pd.read_parquet(args.source_dir / "queries.parquet").sort_values("query_index")
    item_count = write_jsonl(
        args.output_dir / "items.jsonl",
        (
            {
                "item_id": str(int(row.item_id)),
                "title": str(row.title),
                "description": ". ".join(
                    part
                    for part in [
                        "Genres: " + ", ".join(values(row.genres)),
                        "User tags: " + ", ".join(values(row.user_tags)[:20]),
                        "Attributes: " + ", ".join(values(row.genome_tags)[:20]),
                    ]
                    if not part.endswith(": ")
                ),
                "brand": "",
                "categories": values(row.genres),
                "item_type": "movie",
            }
            for row in items.itertuples(index=False)
        ),
    )
    write_jsonl(
        args.output_dir / "item_ids.jsonl",
        ({"item_id": str(int(item_id))} for item_id in items["item_id"]),
    )
    query_count = write_jsonl(
        args.output_dir / "queries.jsonl",
        (
            {"query_id": int(row.query_index), "query": str(row.query)}
            for row in queries.itertuples(index=False)
        ),
    )

    connection = duckdb.connect()
    connection.execute("SET TimeZone='UTC'")
    rec_path = sql_path(args.source_dir / "rec_positive_events.parquet")
    search_path = sql_path(args.source_dir / "search_positive_events.parquet")
    output = sql_path(args.output_dir)
    connection.execute(
        f"""
        COPY (
          SELECT user_id::VARCHAR AS user_id, item_id::VARCHAR AS item_id,
                 timestamp::BIGINT AS timestamp,
                 (row_number() OVER (
                   PARTITION BY user_id ORDER BY timestamp, item_id
                 ) - 1)::BIGINT AS position,
                 window_id::INTEGER AS window_id
          FROM read_parquet('{rec_path}')
          WHERE window_id <= {args.through_window}
          ORDER BY user_id, timestamp, item_id
        ) TO '{output}/rec_interactions_chrono.jsonl' (FORMAT JSON, ARRAY false)
        """
    )
    connection.execute(
        f"""
        COPY (
          SELECT ('ml-init-' || search_session_id::VARCHAR) AS interaction_id,
                 user_id::VARCHAR AS user_id, item_id::VARCHAR AS item_id,
                 query_index::BIGINT AS query_id, keyword::VARCHAR AS query,
                 timestamp::BIGINT AS timestamp, 0::INTEGER AS window_id,
                 search_session_id::BIGINT AS search_session_id,
                 'train'::VARCHAR AS split
          FROM read_parquet('{search_path}')
          ORDER BY user_id, timestamp, item_id
        ) TO '{output}/search_interactions.jsonl' (FORMAT JSON, ARRAY false)
        """
    )
    connection.execute(
        f"""
        COPY (
          SELECT DISTINCT user_id::VARCHAR AS user_id
          FROM read_parquet('{rec_path}')
          WHERE window_id <= {args.through_window} ORDER BY user_id
        ) TO '{output}/users.jsonl' (FORMAT JSON, ARRAY false)
        """
    )
    counts = {
        "items": item_count,
        "queries": query_count,
        "rec_rows": int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet('{rec_path}') "
                f"WHERE window_id <= {args.through_window}"
            ).fetchone()[0]
        ),
        "search_rows": int(
            connection.execute(f"SELECT count(*) FROM read_parquet('{search_path}')").fetchone()[0]
        ),
        "through_window": args.through_window,
        "future_behavior_used": False,
        "search_policy": "fixed t0 control",
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(counts, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
