#!/usr/bin/env python3
"""Build the fixed-catalog MovieLens-25M temporal Search+Rec benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import duckdb
import pandas as pd


YEAR_RE = re.compile(r"\s*\((\d{4})\)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument("--queries-per-item", type=int, default=20)
    parser.add_argument("--train-queries-per-item", type=int, default=10)
    parser.add_argument("--val-queries-per-item", type=int, default=5)
    parser.add_argument("--max-search-events-per-user", type=int, default=8)
    parser.add_argument("--top-user-tags", type=int, default=20)
    parser.add_argument("--top-genome-tags", type=int, default=20)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_title(value: str) -> tuple[str, int | None]:
    title = str(value).strip()
    match = YEAR_RE.search(title)
    if match is None:
        return title, None
    return title[: match.start()].strip(), int(match.group(1))


def split_pipe(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def split_unit(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("\x1f") if part.strip()]


def normalize_query(value: str) -> str:
    return " ".join(str(value).lower().strip().split())


def item_queries(row: dict, count: int) -> list[str]:
    title = row["title"]
    year = row.get("release_year")
    genres = row["genres"]
    tags = row["user_tags"] + row["genome_tags"]
    candidates: list[str] = []

    def add(value: str) -> None:
        value = normalize_query(value)
        if value and value not in candidates:
            candidates.append(value)

    for tag in tags:
        add(tag)
    for genre in genres:
        add(f"{genre} movie")
    for tag in tags[:8]:
        add(f"movie about {tag}")
    for genre in genres[:4]:
        for tag in tags[:4]:
            add(f"{genre} movie {tag}")
    add(title)
    add(f"movie {title}")
    if year is not None:
        add(f"{title} {year}")
    for genre in genres[:4]:
        add(f"{genre} film like {title}")
    fallback = [
        f"find {title}",
        f"watch {title}",
        f"films similar to {title}",
        f"recommend a movie like {title}",
    ]
    for value in fallback:
        add(value)
    suffix = 1
    while len(candidates) < count:
        add(f"{title} movie recommendation {suffix}")
        suffix += 1
    return candidates[:count]


def copy_parquet(connection: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    connection.execute(
        f"COPY ({query}) TO '{sql_path(path)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def main() -> None:
    args = parse_args()
    if args.queries_per_item != args.train_queries_per_item + args.val_queries_per_item + 5:
        raise ValueError("queries-per-item must equal train + validation + 5 test queries")
    if args.start_year >= args.end_year:
        raise ValueError("start-year must be earlier than end-year")
    required = [
        args.raw_dir / name
        for name in ["ratings.csv", "movies.csv", "tags.csv", "genome-scores.csv", "genome-tags.csv"]
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing MovieLens files: {missing}")
    if args.output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{args.output_dir} exists; pass --force to replace it")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    temp_dir = args.output_dir / "duckdb_tmp"
    temp_dir.mkdir()

    start_ts = int(pd.Timestamp(f"{args.start_year}-01-01", tz="UTC").timestamp())
    end_ts = int(pd.Timestamp(f"{args.end_year}-01-01", tz="UTC").timestamp())
    start_half = args.start_year * 2
    window_count = (args.end_year - args.start_year) * 2

    connection = duckdb.connect()
    connection.execute(f"SET threads={args.threads}")
    connection.execute("SET TimeZone='UTC'")
    connection.execute(f"SET temp_directory='{sql_path(temp_dir)}'")
    connection.execute(
        f"CREATE TEMP VIEW ratings_raw AS SELECT * FROM read_csv_auto('{sql_path(args.raw_dir / 'ratings.csv')}')"
    )
    connection.execute(
        f"CREATE TEMP VIEW movies_raw AS SELECT * FROM read_csv_auto('{sql_path(args.raw_dir / 'movies.csv')}')"
    )
    connection.execute(
        f"CREATE TEMP VIEW tags_raw AS SELECT * FROM read_csv_auto('{sql_path(args.raw_dir / 'tags.csv')}')"
    )
    connection.execute(
        f"CREATE TEMP VIEW genome_scores_raw AS SELECT * FROM read_csv_auto('{sql_path(args.raw_dir / 'genome-scores.csv')}')"
    )
    connection.execute(
        f"CREATE TEMP VIEW genome_tags_raw AS SELECT * FROM read_csv_auto('{sql_path(args.raw_dir / 'genome-tags.csv')}')"
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE positive AS
        SELECT userId::BIGINT AS user_id, movieId::BIGINT AS item_id,
               rating::DOUBLE AS rating, timestamp::BIGINT AS timestamp,
               CASE WHEN timestamp < {start_ts} THEN 0 ELSE
                 1 + (year(to_timestamp(timestamp)) * 2
                      + CASE WHEN month(to_timestamp(timestamp)) > 6 THEN 1 ELSE 0 END
                      - {start_half}) END::INTEGER AS window_id
        FROM ratings_raw
        WHERE rating >= {args.min_rating} AND timestamp < {end_ts}
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE catalog AS
        WITH support AS (
          SELECT item_id, count(*)::BIGINT AS rec_support_t0
          FROM positive WHERE timestamp < {start_ts} GROUP BY item_id
        )
        SELECT row_number() OVER (ORDER BY item_id)::BIGINT AS catalog_index,
               item_id, rec_support_t0
        FROM support ORDER BY item_id
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE selected_rec AS
        SELECT p.*, c.catalog_index
        FROM positive p JOIN catalog c USING (item_id)
        ORDER BY timestamp, user_id, item_id
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE user_tag_ranked AS
        SELECT movieId::BIGINT AS item_id, lower(trim(tag)) AS tag, count(*)::BIGINT AS support,
               row_number() OVER (
                 PARTITION BY movieId ORDER BY count(*) DESC, lower(trim(tag))
               ) AS rank
        FROM tags_raw
        WHERE timestamp < {start_ts} AND tag IS NOT NULL AND trim(tag) <> ''
        GROUP BY movieId, lower(trim(tag))
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE genome_ranked AS
        SELECT s.movieId::BIGINT AS item_id, lower(trim(t.tag)) AS tag,
               s.relevance::DOUBLE AS relevance,
               row_number() OVER (
                 PARTITION BY s.movieId ORDER BY s.relevance DESC, s.tagId
               ) AS rank
        FROM genome_scores_raw s JOIN genome_tags_raw t USING (tagId)
        """
    )
    items = connection.execute(
        f"""
        WITH user_tags AS (
          SELECT item_id, string_agg(tag, chr(31) ORDER BY rank) AS tags
          FROM user_tag_ranked WHERE rank <= {args.top_user_tags} GROUP BY item_id
        ), genome_tags AS (
          SELECT item_id, string_agg(tag, chr(31) ORDER BY rank) AS tags
          FROM genome_ranked WHERE rank <= {args.top_genome_tags} GROUP BY item_id
        )
        SELECT c.catalog_index, c.item_id, c.rec_support_t0, m.title, m.genres,
               u.tags AS user_tags, g.tags AS genome_tags
        FROM catalog c JOIN movies_raw m ON c.item_id = m.movieId
        LEFT JOIN user_tags u USING (item_id)
        LEFT JOIN genome_tags g USING (item_id)
        ORDER BY c.catalog_index
        """
    ).fetchdf()
    item_rows = []
    bank_rows = []
    query_to_index: dict[str, int] = {}
    for row in items.itertuples(index=False):
        title, release_year = clean_title(row.title)
        value = {
            "catalog_index": int(row.catalog_index),
            "item_id": int(row.item_id),
            "rec_support_t0": int(row.rec_support_t0),
            "title": title,
            "release_year": release_year,
            "genres": split_pipe(row.genres),
            "user_tags": split_unit(row.user_tags),
            "genome_tags": split_unit(row.genome_tags),
        }
        item_rows.append(value)
        queries = item_queries(value, args.queries_per_item)
        for slot, query in enumerate(queries):
            query_index = query_to_index.setdefault(query, len(query_to_index) + 1)
            split = (
                "train"
                if slot < args.train_queries_per_item
                else "validation"
                if slot < args.train_queries_per_item + args.val_queries_per_item
                else "test"
            )
            bank_rows.append(
                {
                    "catalog_index": value["catalog_index"],
                    "item_id": value["item_id"],
                    "slot": slot,
                    "query_index": query_index,
                    "query": query,
                    "split": split,
                }
            )

    items_frame = pd.DataFrame(item_rows)
    bank_frame = pd.DataFrame(bank_rows)
    queries_frame = (
        bank_frame[["query_index", "query"]].drop_duplicates("query_index").sort_values("query_index")
    )
    connection.register("items_frame", items_frame)
    connection.register("bank_frame", bank_frame)
    connection.execute("CREATE TEMP TABLE item_metadata AS SELECT * FROM items_frame")
    connection.execute("CREATE TEMP TABLE query_bank AS SELECT * FROM bank_frame")

    copy_parquet(connection, "SELECT * FROM catalog ORDER BY catalog_index", args.output_dir / "catalog.parquet")
    copy_parquet(connection, "SELECT * FROM item_metadata ORDER BY catalog_index", args.output_dir / "items.parquet")
    copy_parquet(connection, "SELECT * FROM query_bank ORDER BY catalog_index, slot", args.output_dir / "query_bank.parquet")
    queries_frame.to_parquet(args.output_dir / "queries.parquet", index=False)
    copy_parquet(connection, "SELECT * FROM selected_rec", args.output_dir / "rec_positive_events.parquet")

    connection.execute(
        f"""
        CREATE TEMP TABLE search_events AS
        WITH sampled AS (
          SELECT *, row_number() OVER (
            PARTITION BY user_id ORDER BY hash(user_id, item_id, timestamp), timestamp, item_id
          ) AS sample_rank
          FROM selected_rec WHERE window_id = 0
        )
        SELECT row_number() OVER (ORDER BY s.timestamp, s.user_id, s.item_id)::BIGINT
                 AS search_session_id,
               s.user_id, s.item_id, s.catalog_index AS item_index, s.timestamp,
               0::INTEGER AS window_id, q.query_index, q.query AS keyword,
               1::INTEGER AS click_cnt
        FROM sampled s JOIN query_bank q
          ON s.item_id = q.item_id
         AND q.slot = (hash(s.user_id, s.item_id, s.timestamp) % {args.train_queries_per_item})
        WHERE s.sample_rank <= {args.max_search_events_per_user}
        ORDER BY s.timestamp, s.user_id, s.item_id
        """
    )
    copy_parquet(connection, "SELECT * FROM search_events", args.output_dir / "search_positive_events.parquet")

    connection.execute(
        """
        CREATE TEMP TABLE rec_eval_targets AS
        WITH targets AS (
          SELECT r.*, row_number() OVER (
            PARTITION BY window_id, user_id ORDER BY timestamp, item_id
          ) AS rn
          FROM selected_rec r WHERE window_id > 0
        ), history AS (
          SELECT t.window_id, t.user_id, count(h.item_id)::BIGINT AS prior_rec_events
          FROM (SELECT DISTINCT window_id, user_id FROM targets) t
          LEFT JOIN selected_rec h
            ON h.user_id = t.user_id AND h.window_id < t.window_id
          GROUP BY t.window_id, t.user_id
        )
        SELECT t.* EXCLUDE (rn), h.prior_rec_events
        FROM targets t JOIN history h USING (window_id, user_id)
        WHERE t.rn = 1 AND h.prior_rec_events >= 5
        """
    )
    copy_parquet(connection, "SELECT * FROM rec_eval_targets ORDER BY window_id, user_id", args.output_dir / "rec_eval_targets.parquet")

    windows = [{
        "window_id": 0,
        "period": f"before-{args.start_year}",
        "start_ts": int(connection.execute("SELECT min(timestamp) FROM selected_rec WHERE window_id = 0").fetchone()[0]),
        "end_exclusive_ts": start_ts,
        "role": "initialization",
    }]
    for index in range(window_count):
        start = pd.Timestamp(f"{args.start_year}-01-01", tz="UTC") + pd.DateOffset(months=6 * index)
        end = start + pd.DateOffset(months=6)
        windows.append({
            "window_id": index + 1,
            "period": f"{start.year}-H{1 if start.month == 1 else 2}",
            "start_ts": int(start.timestamp()),
            "end_exclusive_ts": int(end.timestamp()),
            "role": "development" if index < 4 else "pretest" if index < 6 else "test",
        })
    pd.DataFrame(windows).to_csv(args.output_dir / "windows.csv", index=False)

    counts = {
        "catalog_items": int(len(items_frame)),
        "users_t0": int(connection.execute("SELECT count(DISTINCT user_id) FROM selected_rec WHERE window_id = 0").fetchone()[0]),
        "rec_events_t0": int(connection.execute("SELECT count(*) FROM selected_rec WHERE window_id = 0").fetchone()[0]),
        "rec_events_future": int(connection.execute("SELECT count(*) FROM selected_rec WHERE window_id > 0").fetchone()[0]),
        "search_events_t0": int(connection.execute("SELECT count(*) FROM search_events").fetchone()[0]),
        "unique_queries": int(len(queries_frame)),
        "query_item_pairs": int(len(bank_frame)),
        "rec_eval_targets": int(connection.execute("SELECT count(*) FROM rec_eval_targets").fetchone()[0]),
    }
    per_window = connection.execute(
        """
        SELECT window_id, count(*) AS rec_events, count(DISTINCT user_id) AS users,
               count(DISTINCT item_id) AS items
        FROM selected_rec GROUP BY window_id ORDER BY window_id
        """
    ).fetchdf().to_dict(orient="records")
    eval_per_window = connection.execute(
        "SELECT window_id, count(*) AS targets FROM rec_eval_targets GROUP BY window_id ORDER BY window_id"
    ).fetchdf().to_dict(orient="records")
    manifest = {
        "name": "MovieLens25M_18K_FixedCatalog_Temporal_SearchRec_v1",
        "source": str(args.raw_dir.resolve()),
        "positive_definition": f"rating >= {args.min_rating}",
        "catalog_policy": f"items with a positive event before {args.start_year}-01-01",
        "search_policy": {
            "type": "stationary synthetic control",
            "metadata": "title, genres, pre-t0 user tags, static genome tags",
            "queries_per_item": args.queries_per_item,
            "split": [args.train_queries_per_item, args.val_queries_per_item, 5],
            "max_t0_events_per_user": args.max_search_events_per_user,
        },
        "temporal_policy": {
            "initialization": f"before {args.start_year}-01-01",
            "windows": window_count,
            "window_width": "six months",
            "future_behavior_used_for_t0_representation": False,
        },
        "counts": counts,
        "rec_windows": per_window,
        "rec_eval_windows": eval_per_window,
        "raw_sha256": {path.name: sha256(path) for path in required},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    shutil.rmtree(temp_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
