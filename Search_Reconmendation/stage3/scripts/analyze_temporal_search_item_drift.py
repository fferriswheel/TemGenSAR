#!/usr/bin/env python3
"""Rank item-level Search drift from changes in associated query distributions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_temporal_item_drift import support_bucket
from run_temporal_drift_detection import counter_js, quarterly_protocol, write_json


def load_search(path: Path) -> pd.DataFrame:
    frame = pd.read_json(path, lines=True)
    if not np.issubdtype(frame["timestamp"].dtype, np.datetime64):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s")
    return frame


def aggregate(frame: pd.DataFrame) -> dict[str, Counter]:
    result: dict[str, Counter] = defaultdict(Counter)
    for row in frame.itertuples():
        result[str(row.item_id)][str(row.query)] += 1
    return result


def centroid_distance(
    left: Counter, right: Counter, query_to_index: dict[str, int], embeddings: np.ndarray
) -> float:
    def mean(counter: Counter) -> np.ndarray:
        total = max(float(sum(counter.values())), 1.0)
        value = sum(
            float(count) * embeddings[query_to_index[query]]
            for query, count in counter.items()
        ) / total
        return value / max(float(np.linalg.norm(value)), 1e-12)

    return float(1.0 - np.clip(mean(left) @ mean(right), -1.0, 1.0))


def percentile_rank(values: pd.Series) -> pd.Series:
    if len(values) <= 1:
        return pd.Series(np.ones(len(values)), index=values.index)
    return values.rank(method="average", pct=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--query-embedding-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-events", type=int, default=5)
    parser.add_argument("--top-fraction", type=float, default=0.30)
    parser.add_argument(
        "--enable-selection",
        action="store_true",
        help="Set only after the task-level Search detector has triggered.",
    )
    args = parser.parse_args()

    windows, _, pairs = quarterly_protocol()
    frame = load_search(args.data_dir / "search_interactions.jsonl")
    cache = np.load(args.query_embedding_cache, allow_pickle=True)
    queries = [str(value) for value in cache["queries"].tolist()]
    embeddings = cache["embeddings"].astype(np.float32)
    query_to_index = {query: index for index, query in enumerate(queries)}
    missing = set(frame["query"].astype(str)) - set(query_to_index)
    if missing:
        raise ValueError(f"Query embedding cache is missing {len(missing)} queries")

    aggregates = {}
    for name, (start, end) in windows.items():
        rows = frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)]
        aggregates[name] = aggregate(rows)

    output_rows = []
    pair_summaries = []
    for left_name, right_name in pairs:
        left, right = aggregates[left_name], aggregates[right_name]
        rows = []
        for item_id in sorted(set(left) | set(right)):
            left_events = sum(left[item_id].values())
            right_events = sum(right[item_id].values())
            support = min(left_events, right_events)
            eligible = support >= args.minimum_events
            query_js = counter_js(left[item_id], right[item_id]) if support else 0.0
            centroid = (
                centroid_distance(left[item_id], right[item_id], query_to_index, embeddings)
                if support
                else 0.0
            )
            rows.append(
                {
                    "item_id": item_id,
                    "left": left_name,
                    "right": right_name,
                    "left_events": left_events,
                    "right_events": right_events,
                    "event_support": support,
                    "support_bucket": support_bucket(support),
                    "query_js": query_js,
                    "query_centroid_shift": centroid,
                    "eligible": eligible,
                }
            )
        pair = pd.DataFrame(rows)
        eligible_mask = pair["eligible"]
        pair["drift_rank_score"] = 0.0
        pair.loc[eligible_mask, "drift_rank_score"] = (
            0.5 * percentile_rank(pair.loc[eligible_mask, "query_js"])
            + 0.5 * percentile_rank(pair.loc[eligible_mask, "query_centroid_shift"])
        )
        selected_count = (
            max(1, int(np.ceil(int(eligible_mask.sum()) * args.top_fraction)))
            if eligible_mask.any() and args.enable_selection
            else 0
        )
        selected_ids = set(
            pair.loc[eligible_mask]
            .sort_values(["drift_rank_score", "event_support", "item_id"], ascending=[False, False, True])
            .head(selected_count)["item_id"]
        )
        pair["selected_for_update"] = pair["item_id"].isin(selected_ids)
        output_rows.extend(pair.to_dict("records"))
        pair_summaries.append(
            {
                "left": left_name,
                "right": right_name,
                "union_items": len(pair),
                "eligible_items": int(eligible_mask.sum()),
                "selected_items": selected_count,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(args.output_dir / "search_item_drift.csv", index=False)
    summary = {
        "config": {
            "data_dir": str(args.data_dir),
            "query_embedding_cache": str(args.query_embedding_cache),
            "minimum_events": args.minimum_events,
            "top_fraction": args.top_fraction,
            "selection_enabled": args.enable_selection,
        },
        "pairs": pair_summaries,
    }
    write_json(args.output_dir / "search_item_drift_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
