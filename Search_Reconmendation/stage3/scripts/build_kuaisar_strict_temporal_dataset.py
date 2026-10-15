#!/usr/bin/env python3
"""Materialize the strict fixed-catalog KuaiSAR temporal benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-hours", type=int, default=72)
    parser.add_argument("--window-count", type=int, default=6)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def scalar(connection: duckdb.DuckDBPyConnection, query: str):
    return connection.execute(query).fetchone()[0]


def copy_parquet(
    connection: duckdb.DuckDBPyConnection, query: str, output_path: Path
) -> None:
    connection.execute(
        f"COPY ({query}) TO '{sql_path(output_path)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def window_bounds(start_ms: int, window_hours: int, count: int) -> pd.DataFrame:
    width = window_hours * 3_600_000
    rows = []
    for window_id in range(count):
        left = start_ms + window_id * width
        right = left + width
        rows.append(
            {
                "window_id": window_id,
                "start_ms": left,
                "end_exclusive_ms": right,
                "start_time_asia_shanghai": str(
                    pd.to_datetime(left, unit="ms", utc=True).tz_convert("Asia/Shanghai")
                ),
                "end_time_asia_shanghai": str(
                    pd.to_datetime(right, unit="ms", utc=True).tz_convert("Asia/Shanghai")
                ),
                "role": (
                    "initialization"
                    if window_id <= 2
                    else "update"
                    if window_id in (3, 4)
                    else "final_holdout"
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluation_support(
    connection: duckdb.DuckDBPyConnection, eval_windows: tuple[int, ...]
) -> list[dict]:
    rows = []
    for window_id in eval_windows:
        for task, target_table in [
            ("rec", "rec_eval_targets"),
            ("search", "search_eval_targets"),
        ]:
            row = connection.execute(
                f"""
                SELECT '{task}' AS task, {window_id} AS window_id,
                       count(*) AS targets,
                       count(DISTINCT user_id) AS users,
                       sum((prior_rec_events > 0)::INTEGER) AS with_rec_history,
                       sum((prior_search_events > 0)::INTEGER) AS with_search_history,
                       sum((prior_rec_events > 0 AND prior_search_events > 0)::INTEGER)
                           AS with_both_histories
                FROM {target_table}
                WHERE window_id = {window_id}
                """
            ).fetchdf().iloc[0].to_dict()
            rows.append(
                {
                    key: int(value) if key not in {"task"} else value
                    for key, value in row.items()
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    if args.window_hours <= 0 or args.window_count < 6:
        raise ValueError("window-hours must be positive and window-count must be at least 6")
    if args.output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{args.output_dir} exists; pass --force to replace it")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    temp_dir = args.output_dir / "duckdb_tmp"
    temp_dir.mkdir()

    summary = json.loads(args.audit_summary.read_text(encoding="utf-8"))
    start_ms = int(summary["time"]["start_ms"])
    width_ms = args.window_hours * 3_600_000
    end_ms = start_ms + args.window_count * width_ms

    rec_path = args.raw_dir / "rec_inter.csv"
    search_path = args.raw_dir / "src_inter.csv"
    item_path = args.raw_dir / "item_features.csv"
    for path in [rec_path, search_path, item_path, args.catalog]:
        if not path.exists():
            raise FileNotFoundError(path)

    connection = duckdb.connect()
    connection.execute(f"SET threads={args.threads}")
    connection.execute(f"SET temp_directory='{sql_path(temp_dir)}'")
    connection.execute(
        f"CREATE TEMP TABLE catalog AS SELECT * FROM read_parquet('{sql_path(args.catalog)}')"
    )
    if scalar(connection, "SELECT count(*) != count(DISTINCT item_id) FROM catalog"):
        raise ValueError("catalog contains duplicate item IDs")
    connection.execute(
        f"CREATE TEMP VIEW rec_raw AS SELECT * FROM read_csv_auto('{sql_path(rec_path)}')"
    )
    connection.execute(
        f"CREATE TEMP VIEW search_raw AS SELECT * FROM read_csv_auto('{sql_path(search_path)}')"
    )
    connection.execute(
        f"CREATE TEMP VIEW item_raw AS SELECT * FROM read_csv_auto('{sql_path(item_path)}')"
    )

    connection.execute(
        f"""
        CREATE TEMP TABLE selected_rec AS
        SELECT r.user_id::BIGINT AS user_id, c.catalog_index::BIGINT AS item_index,
               r.item_id::BIGINT AS item_id, CAST(r.timestamp AS BIGINT) AS timestamp,
               floor((CAST(r.timestamp AS BIGINT) - {start_ms}) / {width_ms})::INTEGER
                   AS window_id,
               CAST(r.duration_ms AS DOUBLE) AS duration_ms,
               CAST(r.playing_time AS DOUBLE) AS playing_time,
               r.forward::INTEGER AS forward, r.like::INTEGER AS like,
               r.follow::INTEGER AS follow
        FROM rec_raw r JOIN catalog c USING (item_id)
        WHERE r.click = 1 AND CAST(r.timestamp AS BIGINT) >= {start_ms}
          AND CAST(r.timestamp AS BIGINT) < {end_ms}
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE selected_search AS
        SELECT s.user_id::BIGINT AS user_id,
               s.search_session_id::BIGINT AS search_session_id,
               CAST(s.search_session_timestamp AS BIGINT) AS timestamp,
               floor((CAST(s.search_session_timestamp AS BIGINT) - {start_ms}) / {width_ms})::INTEGER
                   AS window_id,
               s.keyword::VARCHAR AS keyword, c.catalog_index::BIGINT AS item_index,
               s.item_id::BIGINT AS item_id, s.click_cnt::INTEGER AS click_cnt,
               s.search_session_source::VARCHAR AS session_source,
               s.item_type::VARCHAR AS exposed_item_type
        FROM search_raw s JOIN catalog c USING (item_id)
        WHERE CAST(s.search_session_timestamp AS BIGINT) >= {start_ms}
          AND CAST(s.search_session_timestamp AS BIGINT) < {end_ms}
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE users AS
        WITH ids AS (
          SELECT user_id FROM selected_rec UNION SELECT user_id FROM selected_search
        )
        SELECT row_number() OVER (ORDER BY user_id)::BIGINT AS user_index, user_id
        FROM ids ORDER BY user_id
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE queries AS
        SELECT row_number() OVER (ORDER BY keyword)::BIGINT AS query_index, keyword
        FROM (SELECT DISTINCT keyword FROM selected_search) ORDER BY keyword
        """
    )

    copy_parquet(
        connection,
        """
        SELECT r.*, u.user_index
        FROM selected_rec r JOIN users u USING (user_id)
        ORDER BY timestamp, user_id, item_id
        """,
        args.output_dir / "rec_positive_events.parquet",
    )
    copy_parquet(
        connection,
        """
        SELECT s.*, u.user_index, q.query_index
        FROM selected_search s JOIN users u USING (user_id) JOIN queries q USING (keyword)
        ORDER BY timestamp, user_id, search_session_id, item_id
        """,
        args.output_dir / "search_impressions.parquet",
    )
    copy_parquet(
        connection,
        """
        SELECT s.*, u.user_index, q.query_index
        FROM selected_search s JOIN users u USING (user_id) JOIN queries q USING (keyword)
        WHERE click_cnt > 0
        ORDER BY timestamp, user_id, search_session_id, item_id
        """,
        args.output_dir / "search_positive_events.parquet",
    )
    copy_parquet(connection, "SELECT * FROM users ORDER BY user_index", args.output_dir / "users.parquet")
    copy_parquet(connection, "SELECT * FROM queries ORDER BY query_index", args.output_dir / "queries.parquet")
    copy_parquet(
        connection,
        """
        SELECT c.*, i.first_level_category_id, i.first_level_category_name_en,
               i.second_level_category_id, i.second_level_category_name_en,
               i.third_level_category_id, i.third_level_category_name_en,
               i.fourth_level_category_id, i.fourth_level_category_name_en,
               i.caption::VARCHAR AS caption_tokens, i.author_id, i.item_type,
               i.upload_time, i.upload_type
        FROM catalog c
        LEFT JOIN (
          SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (PARTITION BY item_id ORDER BY item_id) AS rn
            FROM item_raw
          ) WHERE rn = 1
        ) i USING (item_id)
        ORDER BY c.catalog_index
        """,
        args.output_dir / "items.parquet",
    )

    connection.execute(
        f"""
        CREATE TEMP TABLE rec_eval_targets AS
        WITH targets AS (
          SELECT *, row_number() OVER (
            PARTITION BY window_id, user_id ORDER BY timestamp, item_id
          ) AS rn
          FROM selected_rec WHERE window_id IN (4, 5)
        ), rec_history AS (
          SELECT t.window_id, t.user_id, count(r.item_id)::BIGINT AS prior_rec_events
          FROM (SELECT DISTINCT window_id, user_id FROM targets) t
          LEFT JOIN selected_rec r ON r.user_id = t.user_id
            AND r.timestamp < {start_ms} + t.window_id * {width_ms}
          GROUP BY t.window_id, t.user_id
        ), search_history AS (
          SELECT t.window_id, t.user_id,
                 count(*) FILTER (WHERE s.click_cnt > 0)::BIGINT AS prior_search_events
          FROM (SELECT DISTINCT window_id, user_id FROM targets) t
          LEFT JOIN selected_search s ON s.user_id = t.user_id
            AND s.timestamp < {start_ms} + t.window_id * {width_ms}
          GROUP BY t.window_id, t.user_id
        )
        SELECT t.user_id, u.user_index, t.item_id, t.item_index, t.timestamp,
               t.window_id, r.prior_rec_events, s.prior_search_events
        FROM targets t JOIN rec_history r USING (window_id, user_id)
        JOIN search_history s USING (window_id, user_id)
        JOIN users u USING (user_id)
        WHERE t.rn = 1
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE search_eval_targets AS
        WITH positives AS (
          SELECT *, row_number() OVER (
            PARTITION BY window_id, user_id
            ORDER BY timestamp, search_session_id, item_id
          ) AS rn
          FROM selected_search WHERE click_cnt > 0 AND window_id IN (4, 5)
        ), rec_history AS (
          SELECT t.window_id, t.user_id, count(r.item_id)::BIGINT AS prior_rec_events
          FROM (SELECT DISTINCT window_id, user_id FROM positives) t
          LEFT JOIN selected_rec r ON r.user_id = t.user_id
            AND r.timestamp < {start_ms} + t.window_id * {width_ms}
          GROUP BY t.window_id, t.user_id
        ), search_history AS (
          SELECT t.window_id, t.user_id,
                 count(*) FILTER (WHERE s.click_cnt > 0)::BIGINT AS prior_search_events
          FROM (SELECT DISTINCT window_id, user_id FROM positives) t
          LEFT JOIN selected_search s ON s.user_id = t.user_id
            AND s.timestamp < {start_ms} + t.window_id * {width_ms}
          GROUP BY t.window_id, t.user_id
        )
        SELECT t.user_id, u.user_index, t.item_id, t.item_index, t.keyword,
               q.query_index, t.search_session_id, t.timestamp, t.window_id,
               r.prior_rec_events, s.prior_search_events
        FROM positives t JOIN rec_history r USING (window_id, user_id)
        JOIN search_history s USING (window_id, user_id)
        JOIN users u USING (user_id) JOIN queries q USING (keyword)
        WHERE t.rn = 1
        """
    )
    copy_parquet(
        connection,
        "SELECT * FROM rec_eval_targets ORDER BY window_id, user_id",
        args.output_dir / "rec_eval_targets.parquet",
    )
    copy_parquet(
        connection,
        "SELECT * FROM search_eval_targets ORDER BY window_id, user_id",
        args.output_dir / "search_eval_targets.parquet",
    )

    counts = {
        "catalog_items": int(scalar(connection, "SELECT count(*) FROM catalog")),
        "users": int(scalar(connection, "SELECT count(*) FROM users")),
        "queries": int(scalar(connection, "SELECT count(*) FROM queries")),
        "rec_positive_events": int(scalar(connection, "SELECT count(*) FROM selected_rec")),
        "search_impressions": int(scalar(connection, "SELECT count(*) FROM selected_search")),
        "search_positive_events": int(
            scalar(connection, "SELECT count(*) FROM selected_search WHERE click_cnt > 0")
        ),
        "items_missing_metadata": int(
            scalar(
                connection,
                "SELECT count(*) FROM catalog c ANTI JOIN item_raw i USING (item_id)",
            )
        ),
    }
    windows = window_bounds(start_ms, args.window_hours, args.window_count)
    windows.to_csv(args.output_dir / "windows.csv", index=False)
    support = evaluation_support(connection, (4, 5))
    manifest = {
        "name": "KuaiSAR_Strict_Balanced_30K_W0_W5",
        "catalog_policy": {
            "source_windows": [0, 2],
            "selection": "top-15000 Rec-positive support union top-15000 Search-positive support",
            "fixed_after_initialization": True,
            "future_catalog_information": False,
            "catalog_sha256": sha256(args.catalog),
        },
        "event_policy": {
            "rec_positive": "click == 1",
            "search_positive": "click_cnt > 0",
            "search_impressions_preserved": True,
        },
        "temporal_protocol": {
            "window_hours": args.window_hours,
            "initialization": [0, 2],
            "first_update": 3,
            "development_evaluation": 4,
            "second_update_after_development": 4,
            "final_holdout": 5,
            "evaluation_history_cutoff": "strictly before target window start",
        },
        "counts": counts,
        "evaluation_support": support,
        "source": {
            "raw_dir": str(args.raw_dir.resolve()),
            "catalog": str(args.catalog.resolve()),
            "audit_summary": str(args.audit_summary.resolve()),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(args.catalog, args.output_dir / "catalog.parquet")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
