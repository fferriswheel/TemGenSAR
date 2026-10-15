#!/usr/bin/env python3
"""Audit MovieLens-25M for a long-horizon temporal Search+Rec benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--window-start-year", type=int, default=2015)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def half_label(half_id: int) -> str:
    year, half = divmod(int(half_id), 2)
    return f"{year}-H{half + 1}"


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    left /= left.sum()
    right /= right.sum()
    mean = 0.5 * (left + right)
    left_mask = left > 0
    right_mask = right > 0
    value = 0.5 * np.sum(left[left_mask] * np.log2(left[left_mask] / mean[left_mask]))
    value += 0.5 * np.sum(right[right_mask] * np.log2(right[right_mask] / mean[right_mask]))
    return float(value)


def create_views(connection: duckdb.DuckDBPyConnection, raw_dir: Path, min_rating: float) -> None:
    required = ["ratings.csv", "tags.csv", "movies.csv", "genome-scores.csv"]
    missing = [name for name in required if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing MovieLens files: {missing}")

    connection.execute(
        f"""
        CREATE TEMP VIEW ratings_raw AS
        SELECT * FROM read_csv_auto('{sql_path(raw_dir / 'ratings.csv')}')
        """
    )
    connection.execute(
        f"""
        CREATE TEMP VIEW tags_raw AS
        SELECT * FROM read_csv_auto('{sql_path(raw_dir / 'tags.csv')}')
        """
    )
    connection.execute(
        f"""
        CREATE TEMP VIEW movies_raw AS
        SELECT * FROM read_csv_auto('{sql_path(raw_dir / 'movies.csv')}')
        """
    )
    connection.execute(
        f"""
        CREATE TEMP VIEW genome_raw AS
        SELECT * FROM read_csv_auto('{sql_path(raw_dir / 'genome-scores.csv')}')
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE positive_ratings AS
        SELECT userId AS user_id, movieId AS movie_id, rating,
               CAST(timestamp AS BIGINT) AS ts,
               year(to_timestamp(timestamp))::INTEGER AS event_year,
               (year(to_timestamp(timestamp)) * 2
                + CASE WHEN month(to_timestamp(timestamp)) > 6 THEN 1 ELSE 0 END
               )::INTEGER AS half_id,
               row_number() OVER (
                   PARTITION BY userId ORDER BY timestamp, movieId
               )::INTEGER AS user_event_index
        FROM ratings_raw
        WHERE rating >= {float(min_rating)}
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE tag_events AS
        SELECT userId AS user_id, movieId AS movie_id,
               lower(trim(tag)) AS tag, CAST(timestamp AS BIGINT) AS ts,
               year(to_timestamp(timestamp))::INTEGER AS event_year,
               (year(to_timestamp(timestamp)) * 2
                + CASE WHEN month(to_timestamp(timestamp)) > 6 THEN 1 ELSE 0 END
               )::INTEGER AS half_id
        FROM tags_raw
        WHERE tag IS NOT NULL AND trim(tag) <> ''
        """
    )


def dataset_summary(connection: duckdb.DuckDBPyConnection, min_rating: float) -> dict:
    raw = connection.execute(
        """
        SELECT count(*) AS events, count(DISTINCT userId) AS users,
               count(DISTINCT movieId) AS items,
               min(timestamp) AS min_ts, max(timestamp) AS max_ts
        FROM ratings_raw
        """
    ).fetchdf().iloc[0].to_dict()
    positive = connection.execute(
        """
        SELECT count(*) AS events, count(DISTINCT user_id) AS users,
               count(DISTINCT movie_id) AS items, min(ts) AS min_ts, max(ts) AS max_ts
        FROM positive_ratings
        """
    ).fetchdf().iloc[0].to_dict()
    tags = connection.execute(
        """
        SELECT count(*) AS events, count(DISTINCT user_id) AS users,
               count(DISTINCT movie_id) AS items, count(DISTINCT tag) AS unique_tags,
               min(ts) AS min_ts, max(ts) AS max_ts
        FROM tag_events
        """
    ).fetchdf().iloc[0].to_dict()
    metadata = connection.execute(
        """
        SELECT
          (SELECT count(*) FROM movies_raw) AS movies,
          (SELECT count(DISTINCT movieId) FROM genome_raw) AS movies_with_genome,
          (SELECT count(DISTINCT movie_id) FROM tag_events) AS movies_with_user_tags,
          (SELECT count(*) FROM movies_raw WHERE genres IS NULL OR genres = '(no genres listed)')
              AS movies_without_genres
        """
    ).fetchdf().iloc[0].to_dict()
    return {
        "rating_filter": f"rating >= {min_rating}",
        "ratings_raw": raw,
        "ratings_positive": positive,
        "tag_events": tags,
        "metadata": metadata,
    }


def rating_activity(connection: duckdb.DuckDBPyConnection, grain: str) -> pd.DataFrame:
    if grain == "year":
        key = "event_year"
    elif grain == "halfyear":
        key = "half_id"
    else:
        raise ValueError(grain)
    frame = connection.execute(
        f"""
        SELECT {key} AS period_id, count(*) AS events,
               count(DISTINCT user_id) AS users,
               count(DISTINCT movie_id) AS items,
               sum(CASE WHEN user_event_index >= 3 THEN 1 ELSE 0 END) AS events_history_ge2,
               sum(CASE WHEN user_event_index >= 6 THEN 1 ELSE 0 END) AS events_history_ge5
        FROM positive_ratings
        GROUP BY {key}
        ORDER BY {key}
        """
    ).fetchdf()
    if grain == "halfyear":
        frame.insert(1, "period", frame["period_id"].map(half_label))
    else:
        frame.insert(1, "period", frame["period_id"].astype(str))
    frame["history_ge2_rate"] = frame["events_history_ge2"] / frame["events"]
    frame["history_ge5_rate"] = frame["events_history_ge5"] / frame["events"]
    return frame


def tag_activity(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    frame = connection.execute(
        """
        WITH user_support AS (
          SELECT half_id, user_id, count(*) AS events
          FROM tag_events GROUP BY half_id, user_id
        ), concentration AS (
          SELECT half_id, max(events) / sum(events) AS top_user_event_share,
                 sum(CASE WHEN user_rank <= 10 THEN events ELSE 0 END) / sum(events)
                     AS top10_user_event_share
          FROM (
            SELECT *, row_number() OVER (PARTITION BY half_id ORDER BY events DESC) AS user_rank
            FROM user_support
          ) ranked
          GROUP BY half_id
        ), activity AS (
          SELECT half_id, count(*) AS tag_events,
                 count(DISTINCT user_id) AS users,
                 count(DISTINCT movie_id) AS items,
                 count(DISTINCT tag) AS unique_tags,
                 count(DISTINCT (tag, movie_id)) AS query_item_pairs
          FROM tag_events
          GROUP BY half_id
        )
        SELECT activity.half_id AS period_id, tag_events, users, items,
               unique_tags, query_item_pairs, top_user_event_share, top10_user_event_share
        FROM activity JOIN concentration USING (half_id)
        ORDER BY half_id
        """
    ).fetchdf()
    frame.insert(1, "period", frame["period_id"].map(half_label))
    return frame


def candidate_window_stats(
    connection: duckdb.DuckDBPyConnection, start_year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_half = start_year * 2
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE candidate_item_support AS
        SELECT half_id, movie_id, count(*)::BIGINT AS events
        FROM positive_ratings
        WHERE half_id >= {start_half}
        GROUP BY half_id, movie_id
        """
    )
    windows = connection.execute(
        f"""
        WITH fixed_catalog AS (
          SELECT DISTINCT movie_id FROM positive_ratings WHERE half_id < {start_half}
        ), prior_users AS (
          SELECT p.half_id, p.user_id
          FROM (SELECT DISTINCT half_id, user_id FROM positive_ratings
                WHERE half_id >= {start_half}) p
          WHERE EXISTS (
            SELECT 1 FROM positive_ratings h
            WHERE h.user_id = p.user_id AND h.half_id < p.half_id
          )
        )
        SELECT p.half_id AS period_id,
               count(*) AS events,
               count(DISTINCT p.user_id) AS users,
               count(DISTINCT p.movie_id) AS items,
               sum(CASE WHEN f.movie_id IS NOT NULL THEN 1 ELSE 0 END) AS fixed_catalog_events,
               count(DISTINCT CASE WHEN f.movie_id IS NOT NULL THEN p.movie_id END)
                   AS fixed_catalog_items,
               count(DISTINCT CASE WHEN u.user_id IS NOT NULL THEN p.user_id END)
                   AS returning_users,
               sum(CASE WHEN u.user_id IS NOT NULL THEN 1 ELSE 0 END)
                   AS returning_user_events,
               sum(CASE WHEN p.user_event_index >= 3 THEN 1 ELSE 0 END) AS events_history_ge2,
               sum(CASE WHEN p.user_event_index >= 6 THEN 1 ELSE 0 END) AS events_history_ge5
        FROM positive_ratings p
        LEFT JOIN fixed_catalog f USING (movie_id)
        LEFT JOIN prior_users u ON p.half_id = u.half_id AND p.user_id = u.user_id
        WHERE p.half_id >= {start_half}
        GROUP BY p.half_id
        ORDER BY p.half_id
        """
    ).fetchdf()
    overlap = connection.execute(
        """
        SELECT curr.half_id AS period_id,
               count(*) AS items_seen_in_previous_window,
               sum(curr.events) AS events_on_previous_window_items,
               sum(CASE WHEN curr.events >= 5 AND prev.events >= 5 THEN 1 ELSE 0 END)
                   AS items_support_ge5_both,
               sum(CASE WHEN curr.events >= 10 AND prev.events >= 10 THEN 1 ELSE 0 END)
                   AS items_support_ge10_both
        FROM candidate_item_support curr
        JOIN candidate_item_support prev
          ON curr.half_id = prev.half_id + 1 AND curr.movie_id = prev.movie_id
        GROUP BY curr.half_id
        ORDER BY curr.half_id
        """
    ).fetchdf()
    windows = windows.merge(overlap, on="period_id", how="left").fillna(0)
    windows.insert(1, "period", windows["period_id"].map(half_label))
    windows["fixed_catalog_event_rate"] = windows["fixed_catalog_events"] / windows["events"]
    windows["fixed_catalog_item_rate"] = windows["fixed_catalog_items"] / windows["items"]
    windows["returning_user_rate"] = windows["returning_users"] / windows["users"]
    windows["returning_user_event_rate"] = windows["returning_user_events"] / windows["events"]
    windows["history_ge2_rate"] = windows["events_history_ge2"] / windows["events"]
    windows["history_ge5_rate"] = windows["events_history_ge5"] / windows["events"]
    windows["adjacent_item_overlap_rate"] = (
        windows["items_seen_in_previous_window"] / windows["items"]
    )
    windows["adjacent_event_coverage"] = (
        windows["events_on_previous_window_items"] / windows["events"]
    )

    counts = connection.execute(
        """
        SELECT half_id, movie_id, events FROM candidate_item_support ORDER BY half_id, movie_id
        """
    ).fetchdf()
    pair_rows = []
    periods = sorted(counts["half_id"].unique().tolist())
    grouped = {period: group.set_index("movie_id")["events"] for period, group in counts.groupby("half_id")}
    for left_id, right_id in zip(periods, periods[1:]):
        if right_id != left_id + 1:
            continue
        pair = pd.concat([grouped[left_id], grouped[right_id]], axis=1).fillna(0.0)
        pair.columns = ["left", "right"]
        pair_rows.append(
            {
                "left_period_id": int(left_id),
                "right_period_id": int(right_id),
                "left_period": half_label(left_id),
                "right_period": half_label(right_id),
                "popularity_js_divergence": js_divergence(
                    pair["left"].to_numpy(), pair["right"].to_numpy()
                ),
            }
        )
    return windows, pd.DataFrame(pair_rows)


def tag_window_overlap(connection: duckdb.DuckDBPyConnection, start_year: int) -> pd.DataFrame:
    start_half = start_year * 2
    return connection.execute(
        f"""
        WITH support AS (
          SELECT half_id, movie_id, tag, count(*) AS events
          FROM tag_events WHERE half_id >= {start_half}
          GROUP BY half_id, movie_id, tag
        ), totals AS (
          SELECT half_id, count(*) AS pairs, count(DISTINCT movie_id) AS items,
                 sum(events) AS events
          FROM support GROUP BY half_id
        ), adjacent AS (
          SELECT curr.half_id,
                 count(*) AS repeated_query_item_pairs,
                 count(DISTINCT curr.movie_id) AS repeated_items,
                 sum(curr.events) AS events_on_repeated_pairs
          FROM support curr JOIN support prev
            ON curr.half_id = prev.half_id + 1
           AND curr.movie_id = prev.movie_id AND curr.tag = prev.tag
          GROUP BY curr.half_id
        )
        SELECT totals.half_id AS period_id, totals.pairs, totals.items, totals.events,
               coalesce(repeated_query_item_pairs, 0) AS repeated_query_item_pairs,
               coalesce(repeated_items, 0) AS repeated_items,
               coalesce(events_on_repeated_pairs, 0) AS events_on_repeated_pairs
        FROM totals LEFT JOIN adjacent USING (half_id)
        ORDER BY totals.half_id
        """
    ).fetchdf().assign(
        period=lambda frame: frame["period_id"].map(half_label),
        repeated_pair_rate=lambda frame: frame["repeated_query_item_pairs"] / frame["pairs"],
        repeated_item_rate=lambda frame: frame["repeated_items"] / frame["items"],
        repeated_event_rate=lambda frame: frame["events_on_repeated_pairs"] / frame["events"],
    )[
        [
            "period_id", "period", "pairs", "items", "events",
            "repeated_query_item_pairs", "repeated_items", "events_on_repeated_pairs",
            "repeated_pair_rate", "repeated_item_rate", "repeated_event_rate",
        ]
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={args.threads}")
    connection.execute("PRAGMA temp_directory='/tmp/duckdb_movielens_audit'")
    create_views(connection, args.raw_dir, args.min_rating)

    summary = dataset_summary(connection, args.min_rating)
    yearly = rating_activity(connection, "year")
    halfyear = rating_activity(connection, "halfyear")
    tags = tag_activity(connection)
    windows, drift = candidate_window_stats(connection, args.window_start_year)
    tag_overlap = tag_window_overlap(connection, args.window_start_year)

    yearly.to_csv(args.output_dir / "ratings_yearly.csv", index=False)
    halfyear.to_csv(args.output_dir / "ratings_halfyear.csv", index=False)
    tags.to_csv(args.output_dir / "tags_halfyear.csv", index=False)
    windows.to_csv(args.output_dir / "candidate_rec_windows.csv", index=False)
    drift.to_csv(args.output_dir / "candidate_rec_adjacent_drift.csv", index=False)
    tag_overlap.to_csv(args.output_dir / "candidate_tag_windows.csv", index=False)
    report = {
        "dataset": summary,
        "candidate_protocol": {
            "initialization": f"all positive ratings before {args.window_start_year}-01-01",
            "update_eval_windows": (
                f"consecutive half-year windows from {args.window_start_year}-H1"
            ),
            "positive_definition": f"rating >= {args.min_rating}",
        },
        "candidate_window_summary": {
            "windows": int(len(windows)),
            "median_events": float(windows["events"].median()),
            "median_users": float(windows["users"].median()),
            "median_items": float(windows["items"].median()),
            "median_fixed_catalog_event_rate": float(windows["fixed_catalog_event_rate"].median()),
            "median_adjacent_item_overlap_rate": float(windows["adjacent_item_overlap_rate"].iloc[1:].median()),
            "median_adjacent_event_coverage": float(windows["adjacent_event_coverage"].iloc[1:].median()),
            "median_items_support_ge5_both": float(windows["items_support_ge5_both"].iloc[1:].median()),
            "median_history_ge5_rate": float(windows["history_ge5_rate"].median()),
            "median_popularity_js_divergence": float(drift["popularity_js_divergence"].median()),
        },
        "tag_proxy_summary": {
            "median_events": float(tag_overlap["events"].median()),
            "median_items": float(tag_overlap["items"].median()),
            "median_repeated_pair_rate": float(tag_overlap["repeated_pair_rate"].iloc[1:].median()),
            "median_repeated_event_rate": float(tag_overlap["repeated_event_rate"].iloc[1:].median()),
        },
        "candidate_rec_windows": records(windows),
        "candidate_rec_adjacent_drift": records(drift),
        "candidate_tag_windows": records(tag_overlap),
    }
    with (args.output_dir / "audit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["candidate_window_summary"], indent=2))
    print(json.dumps(report["tag_proxy_summary"], indent=2))


if __name__ == "__main__":
    main()
