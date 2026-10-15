#!/usr/bin/env python3
"""Audit whether KuaiSAR can support a natural temporal Search+Rec benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import duckdb
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--catalog-per-task-top-k", type=int, default=15_000)
    parser.add_argument("--initialization-end-window", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def scalar(connection: duckdb.DuckDBPyConnection, query: str):
    return connection.execute(query).fetchone()[0]


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def complete_window_count(span_hours: float, window_hours: int) -> int:
    exact = span_hours / window_hours
    whole = math.floor(exact)
    return whole + int(exact - whole >= 0.95)


def create_tables(connection: duckdb.DuckDBPyConnection, raw_dir: Path) -> None:
    rec_path = (raw_dir / "rec_inter.csv").as_posix().replace("'", "''")
    search_path = (raw_dir / "src_inter.csv").as_posix().replace("'", "''")
    item_path = (raw_dir / "item_features.csv").as_posix().replace("'", "''")
    for path in [rec_path, search_path, item_path]:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    connection.execute(
        f"CREATE TEMP VIEW rec_raw AS SELECT * FROM read_csv_auto('{rec_path}')"
    )
    connection.execute(
        f"CREATE TEMP VIEW search_raw AS SELECT * FROM read_csv_auto('{search_path}')"
    )
    connection.execute(
        f"CREATE TEMP VIEW item_raw AS SELECT * FROM read_csv_auto('{item_path}')"
    )
    connection.execute(
        """
        CREATE TEMP TABLE rec_positive AS
        SELECT user_id, item_id, CAST(timestamp AS BIGINT) AS ts
        FROM rec_raw WHERE click = 1
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE search_positive AS
        SELECT user_id, item_id, search_session_id, keyword,
               search_session_timestamp AS ts
        FROM search_raw WHERE click_cnt > 0
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE search_sessions AS
        SELECT user_id, search_session_id, any_value(keyword) AS keyword,
               min(search_session_timestamp) AS ts
        FROM search_raw
        GROUP BY user_id, search_session_id
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE item_category AS
        SELECT item_id,
               coalesce(nullif(first_level_category_name_en, ''), '__MISSING__') AS category
        FROM item_raw
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE joint_positive_catalog AS
        SELECT DISTINCT r.item_id
        FROM rec_positive r
        SEMI JOIN search_positive s USING (item_id)
        """
    )


def create_balanced_train_catalog(
    connection: duckdb.DuckDBPyConnection,
    start_ms: int,
    window_hours: int,
    initialization_end_window: int,
    per_task_top_k: int,
) -> None:
    """Freeze a task-balanced catalog using initialization-period behavior only."""
    width = window_hours * 3_600_000
    end_ms = start_ms + (initialization_end_window + 1) * width
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE balanced_train_catalog AS
        WITH rec_support AS (
          SELECT item_id, count(*)::BIGINT AS rec_support
          FROM rec_positive
          WHERE ts >= {start_ms} AND ts < {end_ms}
          GROUP BY item_id
        ), search_support AS (
          SELECT item_id, count(*)::BIGINT AS search_support
          FROM search_positive
          WHERE ts >= {start_ms} AND ts < {end_ms}
          GROUP BY item_id
        ), rec_ranked AS (
          SELECT item_id, rec_support,
                 row_number() OVER (ORDER BY rec_support DESC, item_id) AS rec_rank
          FROM rec_support
        ), search_ranked AS (
          SELECT item_id, search_support,
                 row_number() OVER (ORDER BY search_support DESC, item_id) AS search_rank
          FROM search_support
        ), selected AS (
          SELECT item_id FROM rec_ranked WHERE rec_rank <= {per_task_top_k}
          UNION
          SELECT item_id FROM search_ranked WHERE search_rank <= {per_task_top_k}
        )
        SELECT s.item_id,
               coalesce(r.rec_support, 0)::BIGINT AS rec_support_w0_w2,
               coalesce(q.search_support, 0)::BIGINT AS search_support_w0_w2,
               r.rec_rank,
               q.search_rank,
               coalesce(r.rec_rank <= {per_task_top_k}, false) AS selected_by_rec,
               coalesce(q.search_rank <= {per_task_top_k}, false) AS selected_by_search,
               CASE
                 WHEN r.rec_rank <= {per_task_top_k}
                  AND q.search_rank <= {per_task_top_k} THEN 'both'
                 WHEN r.rec_rank <= {per_task_top_k} THEN 'rec'
                 ELSE 'search'
               END AS selected_by
        FROM selected s
        LEFT JOIN rec_ranked r USING (item_id)
        LEFT JOIN search_ranked q USING (item_id)
        ORDER BY s.item_id
        """
    )


def dataset_summary(connection: duckdb.DuckDBPyConnection) -> dict:
    rec = connection.execute(
        """
        SELECT count(*) AS rows, count(DISTINCT user_id) AS users,
               count(DISTINCT item_id) AS items,
               min(CAST(timestamp AS BIGINT)) AS min_ts,
               max(CAST(timestamp AS BIGINT)) AS max_ts,
               sum(click) AS positive_rows
        FROM rec_raw
        """
    ).fetchdf().iloc[0].to_dict()
    search = connection.execute(
        """
        SELECT count(*) AS rows, count(DISTINCT user_id) AS users,
               count(DISTINCT item_id) AS items,
               count(DISTINCT (user_id, search_session_id)) AS sessions,
               min(search_session_timestamp) AS min_ts,
               max(search_session_timestamp) AS max_ts,
               sum(CASE WHEN click_cnt > 0 THEN 1 ELSE 0 END) AS positive_rows
        FROM search_raw
        """
    ).fetchdf().iloc[0].to_dict()
    overlap = connection.execute(
        """
        WITH rec_all AS (SELECT DISTINCT item_id FROM rec_raw),
             search_all AS (SELECT DISTINCT item_id FROM search_raw),
             rec_pos AS (SELECT DISTINCT item_id FROM rec_positive),
             search_pos AS (SELECT DISTINCT item_id FROM search_positive)
        SELECT
          (SELECT count(*) FROM rec_all JOIN search_all USING (item_id)) AS all_overlap,
          (SELECT count(*) FROM rec_pos) AS rec_positive_items,
          (SELECT count(*) FROM search_pos) AS search_positive_items,
          (SELECT count(*) FROM rec_pos JOIN search_pos USING (item_id)) AS positive_overlap
        """
    ).fetchdf().iloc[0].to_dict()
    item = connection.execute(
        """
        SELECT count(*) AS rows, count(DISTINCT item_id) AS items,
               sum(CASE WHEN caption IS NULL OR caption IN ('', '[]') THEN 1 ELSE 0 END)
                   AS empty_caption_rows,
               count(DISTINCT first_level_category_id) AS first_level_categories,
               sum(CASE WHEN first_level_category_name_en IS NULL
                         OR first_level_category_name_en = '' THEN 1 ELSE 0 END)
                   AS missing_english_category_rows
        FROM item_raw
        """
    ).fetchdf().iloc[0].to_dict()
    start_ms = int(min(rec["min_ts"], search["min_ts"]))
    end_ms = int(max(rec["max_ts"], search["max_ts"]))
    return {
        "rec": rec,
        "search": search,
        "item_features": item,
        "item_overlap": overlap,
        "time": {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "span_hours": (end_ms - start_ms) / 3_600_000,
        },
        "text_availability": {
            "caption_format": "anonymized token-id list",
            "query_format": "anonymized token-id list",
            "readable_item_text": "English category names only",
            "compatible_with_existing_bge_title_description": False,
        },
    }


def activity_counts(
    connection: duckdb.DuckDBPyConnection, start_ms: int, window_hours: int
) -> pd.DataFrame:
    width = window_hours * 3_600_000
    query = f"""
    WITH rec AS (
      SELECT floor((ts - {start_ms}) / {width})::INTEGER AS window_id,
             count(*) AS rec_events, count(DISTINCT user_id) AS rec_users,
             count(DISTINCT item_id) AS rec_items
      FROM rec_positive GROUP BY window_id
    ), search AS (
      SELECT floor((ts - {start_ms}) / {width})::INTEGER AS window_id,
             count(*) AS search_events, count(DISTINCT user_id) AS search_users,
             count(DISTINCT item_id) AS search_items,
             count(DISTINCT (user_id, search_session_id)) AS search_sessions
      FROM search_positive GROUP BY window_id
    )
    SELECT coalesce(rec.window_id, search.window_id) AS window_id,
           rec_events, rec_users, rec_items,
           search_events, search_users, search_items, search_sessions
    FROM rec FULL JOIN search USING (window_id)
    ORDER BY window_id
    """
    frame = connection.execute(query).fetchdf()
    frame.insert(1, "window_hours", window_hours)
    frame["start_time"] = (
        pd.to_datetime(start_ms + frame["window_id"] * width, unit="ms", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    frame["end_time"] = (
        pd.to_datetime(start_ms + (frame["window_id"] + 1) * width, unit="ms", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    return frame


def catalog_activity(
    connection: duckdb.DuckDBPyConnection,
    start_ms: int,
    window_hours: int,
    window_count: int,
    catalog_sql: str,
    catalog_name: str,
) -> pd.DataFrame:
    width = window_hours * 3_600_000
    query = f"""
    WITH catalog AS ({catalog_sql}),
    rec AS (
      SELECT floor((ts - {start_ms}) / {width})::INTEGER AS window_id,
             count(*) AS rec_events, count(DISTINCT item_id) AS rec_items
      FROM rec_positive SEMI JOIN catalog USING (item_id)
      GROUP BY window_id
    ), search AS (
      SELECT floor((ts - {start_ms}) / {width})::INTEGER AS window_id,
             count(*) AS search_events, count(DISTINCT item_id) AS search_items
      FROM search_positive SEMI JOIN catalog USING (item_id)
      GROUP BY window_id
    )
    SELECT coalesce(rec.window_id, search.window_id) AS window_id,
           rec_events, rec_items, search_events, search_items,
           (SELECT count(*) FROM catalog) AS catalog_items
    FROM rec FULL JOIN search USING (window_id)
    WHERE coalesce(rec.window_id, search.window_id) BETWEEN 0 AND {window_count - 1}
    ORDER BY window_id
    """
    frame = connection.execute(query).fetchdf()
    frame.insert(0, "catalog", catalog_name)
    return frame


def add_catalog_coverage(
    catalog_activity_frame: pd.DataFrame, all_activity_frame: pd.DataFrame
) -> pd.DataFrame:
    totals = all_activity_frame[
        ["window_id", "rec_events", "rec_items", "search_events", "search_items"]
    ].rename(
        columns={
            "rec_events": "all_rec_events",
            "rec_items": "all_rec_items",
            "search_events": "all_search_events",
            "search_items": "all_search_items",
        }
    )
    frame = catalog_activity_frame.merge(totals, on="window_id", how="left")
    for task in ["rec", "search"]:
        frame[f"{task}_event_coverage"] = (
            frame[f"{task}_events"] / frame[f"all_{task}_events"]
        )
        frame[f"{task}_item_coverage"] = (
            frame[f"{task}_items"] / frame[f"all_{task}_items"]
        )
    return frame


def event_sources(catalog_table: str | None = None) -> dict[str, str]:
    catalog_join = (
        f" SEMI JOIN {catalog_table} catalog USING (item_id)"
        if catalog_table
        else ""
    )
    return {
        "rec_item": (
            "SELECT e.item_id::VARCHAR AS key, e.user_id, e.ts "
            f"FROM rec_positive e{catalog_join}"
        ),
        "search_item": (
            "SELECT e.item_id::VARCHAR AS key, e.user_id, e.ts "
            f"FROM search_positive e{catalog_join}"
        ),
        "rec_category": (
            "SELECT c.category AS key, e.user_id, e.ts FROM rec_positive e "
            "JOIN item_category c USING (item_id)"
        ),
        "search_category": (
            "SELECT c.category AS key, e.user_id, e.ts FROM search_positive e "
            "JOIN item_category c USING (item_id)"
        ),
        "search_query": (
            "SELECT keyword AS key, user_id, ts FROM search_sessions"
        ),
    }


def pair_metrics(
    connection: duckdb.DuckDBPyConnection,
    source_sql: str,
    start_ms: int,
    window_hours: int,
    left: int,
    right: int,
    min_support: int,
    top_k: int,
    persistent_users: bool,
) -> dict:
    width = window_hours * 3_600_000
    user_filter = ""
    if persistent_users:
        user_filter = """
        , persistent AS (
          SELECT DISTINCT l.user_id
          FROM windowed l
          SEMI JOIN windowed r
            ON l.user_id = r.user_id AND r.window_id = {right}
          WHERE l.window_id = {left}
        )
        """.format(left=left, right=right)
    filtered_source = (
        "SELECT * FROM windowed SEMI JOIN persistent USING (user_id)"
        if persistent_users
        else "SELECT * FROM windowed"
    )
    query = f"""
    WITH source AS ({source_sql}),
    windowed AS (
      SELECT key, user_id,
             floor((ts - {start_ms}) / {width})::INTEGER AS window_id
      FROM source
    )
    {user_filter},
    filtered AS ({filtered_source}),
    left_counts AS (
      SELECT key, count(*)::DOUBLE AS n FROM filtered
      WHERE window_id = {left} GROUP BY key
    ),
    right_counts AS (
      SELECT key, count(*)::DOUBLE AS n FROM filtered
      WHERE window_id = {right} GROUP BY key
    ),
    joined AS (
      SELECT coalesce(l.key, r.key) AS key,
             coalesce(l.n, 0) AS left_n, coalesce(r.n, 0) AS right_n
      FROM left_counts l FULL JOIN right_counts r USING (key)
    ),
    totals AS (
      SELECT sum(left_n) AS left_total, sum(right_n) AS right_total FROM joined
    ),
    probabilities AS (
      SELECT *, left_n / nullif(left_total, 0) AS p,
                right_n / nullif(right_total, 0) AS q
      FROM joined CROSS JOIN totals
    ),
    ranked_left AS (
      SELECT key FROM left_counts ORDER BY n DESC, key LIMIT {top_k}
    ),
    ranked_right AS (
      SELECT key FROM right_counts ORDER BY n DESC, key LIMIT {top_k}
    ),
    top_stats AS (
      SELECT count(*) FILTER (WHERE l.key IS NOT NULL AND r.key IS NOT NULL) AS inter_n,
             count(*) AS union_n
      FROM ranked_left l FULL JOIN ranked_right r USING (key)
    )
    SELECT
      max(left_total) AS left_events,
      max(right_total) AS right_events,
      count(*) FILTER (WHERE left_n > 0) AS left_keys,
      count(*) FILTER (WHERE right_n > 0) AS right_keys,
      count(*) FILTER (WHERE left_n > 0 AND right_n > 0) AS repeat_keys,
      count(*) FILTER (WHERE left_n >= {min_support} AND right_n >= {min_support})
          AS supported_repeat_keys,
      count(*) AS union_keys,
      sum(CASE WHEN left_n = 0 THEN right_n ELSE 0 END) / nullif(max(right_total), 0)
          AS new_event_share,
      sum(CASE WHEN right_n = 0 THEN left_n ELSE 0 END) / nullif(max(left_total), 0)
          AS disappeared_event_share,
      sum(
        0.5 * CASE WHEN p > 0 THEN p * log2(p / ((p + q) / 2)) ELSE 0 END
        + 0.5 * CASE WHEN q > 0 THEN q * log2(q / ((p + q) / 2)) ELSE 0 END
      ) AS js_divergence,
      corr(ln(left_n + 1), ln(right_n + 1))
          FILTER (WHERE left_n >= {min_support} AND right_n >= {min_support})
          AS supported_log_count_correlation,
      (SELECT inter_n::DOUBLE / nullif(union_n, 0) FROM top_stats) AS top_k_jaccard
    FROM probabilities
    """
    row = connection.execute(query).fetchdf().iloc[0].to_dict()
    row.update(
        {
            "left_window": left,
            "right_window": right,
            "persistent_users_only": persistent_users,
        }
    )
    return row


def temporal_metrics(
    connection: duckdb.DuckDBPyConnection,
    start_ms: int,
    window_hours: int,
    window_count: int,
    min_support: int,
    top_k: int,
) -> pd.DataFrame:
    rows = []
    source_groups = [
        ("all_items", event_sources()),
        ("joint_catalog", event_sources("joint_positive_catalog")),
        ("balanced_train_catalog", event_sources("balanced_train_catalog")),
    ]
    for catalog_scope, sources in source_groups:
        selected = (
            {"rec_item": sources["rec_item"], "search_item": sources["search_item"]}
            if catalog_scope != "all_items"
            else sources
        )
        for distribution, source_sql in selected.items():
            for left in range(window_count - 1):
                for persistent in [False, True]:
                    row = pair_metrics(
                        connection,
                        source_sql,
                        start_ms,
                        window_hours,
                        left,
                        left + 1,
                        min_support,
                        top_k,
                        persistent,
                    )
                    row.update(
                        {
                            "window_hours": window_hours,
                            "catalog_scope": catalog_scope,
                            "distribution": distribution,
                        }
                    )
                    rows.append(row)
    columns = [
        "window_hours",
        "catalog_scope",
        "distribution",
        "persistent_users_only",
        "left_window",
        "right_window",
        "left_events",
        "right_events",
        "left_keys",
        "right_keys",
        "repeat_keys",
        "supported_repeat_keys",
        "union_keys",
        "new_event_share",
        "disappeared_event_share",
        "js_divergence",
        "supported_log_count_correlation",
        "top_k_jaccard",
    ]
    return pd.DataFrame(rows)[columns]


def build_decision(
    summary: dict,
    daily: pd.DataFrame,
    metrics_72h: pd.DataFrame,
    joint_activity: pd.DataFrame,
    strict_t0_activity: pd.DataFrame,
    initialization_activity: pd.DataFrame,
    balanced_activity: pd.DataFrame,
    balanced_catalog_stats: dict,
) -> dict:
    joint_items = int(summary["item_overlap"]["positive_overlap"])
    search_events = daily["search_events"].dropna().astype(float)
    traffic_ratio = float(search_events.min() / max(search_events.median(), 1.0))
    key_rows = metrics_72h[
        (metrics_72h["catalog_scope"] == "balanced_train_catalog")
        & (metrics_72h["distribution"].isin(["rec_item", "search_item"]))
        & (~metrics_72h["persistent_users_only"])
    ]
    min_supported = {
        name: int(group["supported_repeat_keys"].min())
        for name, group in key_rows.groupby("distribution")
    }
    direct_migration_gates = {
        "balanced_train_catalog_at_least_5000":
            int(balanced_catalog_stats["catalog_items"]) >= 5000,
        "at_least_7_daily_windows": len(daily) >= 7,
        "search_daily_traffic_min_to_median_at_least_0p5": traffic_ratio >= 0.5,
        "existing_bge_text_representation_reusable": False,
        "balanced_catalog_supported_rec_items_per_72h_pair_at_least_100":
            min_supported.get("rec_item", 0) >= 100,
        "balanced_catalog_supported_search_items_per_72h_pair_at_least_100":
            min_supported.get("search_item", 0) >= 100,
    }
    rec_rows = key_rows[key_rows["distribution"] == "rec_item"]
    rec_pilot_gates = {
        "at_least_6_complete_72h_windows": len(balanced_activity) >= 6,
        "balanced_train_catalog_at_least_5000":
            int(balanced_catalog_stats["catalog_items"]) >= 5000,
        "balanced_catalog_rec_events_per_72h_window_at_least_10000":
            int(balanced_activity["rec_events"].min()) >= 10_000,
        "balanced_catalog_search_eval_events_per_72h_window_at_least_1000":
            int(balanced_activity["search_events"].min()) >= 1_000,
        "balanced_catalog_supported_rec_items_per_pair_at_least_100":
            min_supported.get("rec_item", 0) >= 100,
        "median_rec_item_js_at_least_0p1": float(rec_rows["js_divergence"].median()) >= 0.1,
    }
    rec_ready = all(rec_pilot_gates.values())
    direct_blocking = [
        name for name, passed in direct_migration_gates.items() if not passed
    ]
    return {
        "status": (
            "ready_for_strict_train_only_rec_drift_pilot" if rec_ready else "not_ready"
        ),
        "interpretation": (
            "A fixed W0-W2 task-balanced union catalog supports a strict Rec-drift pilot "
            "with Search as a frozen stability task. Direct Amazon text-representation "
            "reuse and item-level Search-suffix drift are not ready."
        ),
        "rec_drift_pilot_gates": rec_pilot_gates,
        "direct_amazon_pipeline_migration_gates": direct_migration_gates,
        "diagnostics": {
            "joint_positive_items": joint_items,
            "daily_search_min_to_median_ratio": traffic_ratio,
            "minimum_supported_joint_catalog_items_per_72h_pair": min_supported,
            "joint_catalog_min_72h_search_events": int(
                joint_activity["search_events"].min()
            ),
            "balanced_train_catalog": balanced_catalog_stats,
            "balanced_catalog_min_72h_rec_events": int(
                balanced_activity["rec_events"].min()
            ),
            "balanced_catalog_min_72h_search_events": int(
                balanced_activity["search_events"].min()
            ),
            "strict_t0_catalog_items": int(strict_t0_activity["catalog_items"].iloc[0]),
            "strict_t0_catalog_min_72h_search_events": int(
                strict_t0_activity["search_events"].min()
            ),
            "initialization_w0_w2_catalog_items": int(
                initialization_activity["catalog_items"].iloc[0]
            ),
            "initialization_w0_w2_future_search_events": {
                f"W{int(row.window_id)}": int(row.search_events)
                for row in initialization_activity.itertuples()
                if int(row.window_id) >= 3
            },
        },
        "direct_migration_blocking_gates": direct_blocking,
        "catalog_policy": {
            "pilot": "W0-W2 TopK Rec-positive union TopK Search-positive catalog",
            "behavior_representation": "strictly time-windowed; no future behavior",
            "future_catalog_information": False,
            "transductive_upper_bound": (
                "all-time Rec-positive intersection Search-positive item catalog"
            ),
            "limitation": "future out-of-catalog items are excluded from the warm-item pilot",
        },
        "next_action": (
            "Build the strict balanced-catalog Rec-only update pilot. Use W0-W2 for "
            "initialization/calibration, W3 update and W4 evaluation, then W4 update and W5 "
            "holdout evaluation. Keep Search SID fixed and report Search stability."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.min_support <= 0 or args.top_k <= 0 or args.catalog_per_task_top_k <= 0:
        raise ValueError("support and top-k arguments must be positive")
    if args.initialization_end_window < 0:
        raise ValueError("initialization-end-window must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.output_dir / "duckdb_tmp"
    temp_dir.mkdir(exist_ok=True)
    connection = duckdb.connect()
    connection.execute(f"SET threads={args.threads}")
    escaped_temp = temp_dir.as_posix().replace("'", "''")
    connection.execute(f"SET temp_directory='{escaped_temp}'")
    create_tables(connection, args.raw_dir)

    summary = dataset_summary(connection)
    start_ms = int(summary["time"]["start_ms"])
    daily = activity_counts(connection, start_ms, 24)
    three_day = activity_counts(connection, start_ms, 72)
    daily_count = complete_window_count(summary["time"]["span_hours"], 24)
    three_day_count = complete_window_count(summary["time"]["span_hours"], 72)
    complete_daily = daily[daily["window_id"] < daily_count]
    complete_three_day = three_day[
        three_day["window_id"] < three_day_count
    ]
    create_balanced_train_catalog(
        connection,
        start_ms,
        72,
        args.initialization_end_window,
        args.catalog_per_task_top_k,
    )
    joint_activity = catalog_activity(
        connection,
        start_ms,
        72,
        len(complete_three_day),
        "SELECT item_id FROM joint_positive_catalog",
        "all_time_joint_positive",
    )
    width_72h = 72 * 3_600_000
    strict_t0_activity = catalog_activity(
        connection,
        start_ms,
        72,
        len(complete_three_day),
        f"""
        SELECT DISTINCT item_id FROM rec_positive
        WHERE floor((ts - {start_ms}) / {width_72h})::INTEGER = 0
        INTERSECT
        SELECT DISTINCT item_id FROM search_positive
        WHERE floor((ts - {start_ms}) / {width_72h})::INTEGER = 0
        """,
        "strict_t0_joint_positive",
    )
    initialization_activity = catalog_activity(
        connection,
        start_ms,
        72,
        len(complete_three_day),
        f"""
        SELECT DISTINCT item_id FROM rec_positive
        WHERE floor((ts - {start_ms}) / {width_72h})::INTEGER BETWEEN 0 AND 2
        INTERSECT
        SELECT DISTINCT item_id FROM search_positive
        WHERE floor((ts - {start_ms}) / {width_72h})::INTEGER BETWEEN 0 AND 2
        """,
        "initialization_w0_w2_joint_positive",
    )
    balanced_activity = catalog_activity(
        connection,
        start_ms,
        72,
        len(complete_three_day),
        "SELECT item_id FROM balanced_train_catalog",
        "strict_w0_w2_balanced_union",
    )
    balanced_catalog_stats = connection.execute(
        """
        SELECT count(*) AS catalog_items,
               sum(selected_by_rec::INTEGER) AS selected_by_rec_items,
               sum(selected_by_search::INTEGER) AS selected_by_search_items,
               sum((selected_by = 'both')::INTEGER) AS selected_by_both_items,
               sum((rec_support_w0_w2 > 0)::INTEGER) AS items_with_rec_support,
               sum((search_support_w0_w2 > 0)::INTEGER) AS items_with_search_support
        FROM balanced_train_catalog
        """
    ).fetchdf().iloc[0].to_dict()
    metrics_24h = temporal_metrics(
        connection,
        start_ms,
        24,
        len(complete_daily),
        args.min_support,
        args.top_k,
    )
    metrics_72h = temporal_metrics(
        connection,
        start_ms,
        72,
        len(complete_three_day),
        args.min_support,
        args.top_k,
    )
    decision = build_decision(
        summary,
        complete_daily,
        metrics_72h,
        joint_activity,
        strict_t0_activity,
        initialization_activity,
        balanced_activity,
        balanced_catalog_stats,
    )
    summary["protocol"] = {
        "positive_rec": "click == 1",
        "positive_search": "click_cnt > 0",
        "window_anchor_ms": start_ms,
        "complete_24h_windows": len(complete_daily),
        "complete_72h_windows": len(complete_three_day),
        "minimum_pair_support": args.min_support,
        "top_k": args.top_k,
        "catalog_per_task_top_k": args.catalog_per_task_top_k,
        "initialization_windows": [0, args.initialization_end_window],
    }

    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    complete_daily.to_csv(args.output_dir / "activity_24h.csv", index=False)
    complete_three_day.to_csv(args.output_dir / "activity_72h.csv", index=False)
    catalog_activities = pd.concat(
        [joint_activity, strict_t0_activity, initialization_activity, balanced_activity],
        ignore_index=True,
    )
    add_catalog_coverage(catalog_activities, complete_three_day).to_csv(
        args.output_dir / "catalog_activity_72h.csv", index=False
    )
    balanced_catalog = connection.execute(
        """
        SELECT row_number() OVER (ORDER BY item_id)::BIGINT AS catalog_index, *
        FROM balanced_train_catalog ORDER BY item_id
        """
    ).fetchdf()
    balanced_catalog.to_csv(args.output_dir / "balanced_train_catalog.csv", index=False)
    balanced_catalog.to_parquet(
        args.output_dir / "balanced_train_catalog.parquet", index=False
    )
    metrics_24h.to_csv(args.output_dir / "drift_metrics_24h.csv", index=False)
    metrics_72h.to_csv(args.output_dir / "drift_metrics_72h.csv", index=False)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "decision": decision}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
