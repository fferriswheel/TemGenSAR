#!/usr/bin/env python3
"""Analyze item-level temporal drift and produce selective update candidates."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from run_temporal_drift_detection import WINDOWS, counter_js, quarterly_protocol, write_json


PAIRS = [("B4", "W1"), ("W1", "W2"), ("W2", "W3")]


def load_rec(path):
    frame = pd.read_json(path, lines=True)
    if not np.issubdtype(frame["timestamp"].dtype, np.datetime64):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s")
    return frame.sort_values(["user_id", "timestamp", "position"])


def aggregate(frame):
    popularity = Counter(frame["item_id"].astype(str))
    transition = defaultdict(Counter)
    for _, group in frame.groupby("user_id", sort=False):
        items = group["item_id"].astype(str).tolist()
        for source, target in zip(items, items[1:]):
            transition[source][target] += 1
    return popularity, transition


def support_bucket(value):
    if value < 5:
        return "0-4"
    if value < 10:
        return "5-9"
    if value < 20:
        return "10-19"
    if value < 50:
        return "20-49"
    return "50+"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-events", type=int, default=5)
    parser.add_argument("--minimum-transitions", type=int, default=3)
    parser.add_argument("--top-fraction", type=float, default=0.30)
    parser.add_argument("--window-mode", choices=["halfyear", "quarterly"], default="halfyear")
    args = parser.parse_args()
    windows, pairs = (quarterly_protocol()[0], quarterly_protocol()[2]) \
        if args.window_mode == "quarterly" else (WINDOWS, PAIRS)
    frame = load_rec(args.data_dir / "rec_interactions_chrono.jsonl")
    aggregates = {}
    for name, (start, end) in windows.items():
        rows = frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)]
        aggregates[name] = aggregate(rows)

    report = {"config": {**vars(args), "data_dir": str(args.data_dir), "output_dir": str(args.output_dir)}, "pairs": []}
    output_rows = []
    for left_name, right_name in pairs:
        left_pop, left_trans = aggregates[left_name]
        right_pop, right_trans = aggregates[right_name]
        items = sorted(set(left_pop) | set(right_pop))
        rows = []
        for item in items:
            left_events = left_pop[item]
            right_events = right_pop[item]
            left_t = sum(left_trans[item].values())
            right_t = sum(right_trans[item].values())
            event_support = min(left_events, right_events)
            transition_support = min(left_t, right_t)
            pop_log_ratio = abs(np.log((right_events + 1.0) / (left_events + 1.0)))
            transition_js = counter_js(left_trans[item], right_trans[item]) if transition_support else 0.0
            eligible = event_support >= args.minimum_events and transition_support >= args.minimum_transitions
            # Ranking, not a calibrated detector: select the strongest supported items
            # after the task-level detector has already triggered.
            score = np.log1p(event_support) * (0.4 * pop_log_ratio + 0.6 * transition_js)
            rows.append({
                "item_id": item, "left": left_name, "right": right_name,
                "left_events": left_events, "right_events": right_events,
                "event_support": event_support, "support_bucket": support_bucket(event_support),
                "left_transitions": left_t, "right_transitions": right_t,
                "transition_support": transition_support,
                "popularity_log_ratio": float(pop_log_ratio),
                "transition_js": float(transition_js),
                "eligible": bool(eligible), "drift_rank_score": float(score),
            })
        eligible_rows = [row for row in rows if row["eligible"]]
        eligible_rows.sort(key=lambda row: row["drift_rank_score"], reverse=True)
        selected_count = max(1, int(np.ceil(len(eligible_rows) * args.top_fraction))) if eligible_rows else 0
        selected = {row["item_id"] for row in eligible_rows[:selected_count]}
        for row in rows:
            row["selected_for_update"] = row["item_id"] in selected
            output_rows.append(row)
        buckets = Counter(row["support_bucket"] for row in rows)
        report["pairs"].append({
            "left": left_name, "right": right_name, "union_items": len(rows),
            "eligible_items": len(eligible_rows), "selected_items": len(selected),
            "eligible_fraction_of_union": len(eligible_rows) / max(len(rows), 1),
            "selected_fraction_of_union": len(selected) / max(len(rows), 1),
            "support_buckets": dict(sorted(buckets.items())),
            "score_quantiles_eligible": {
                key: float(np.quantile([row["drift_rank_score"] for row in eligible_rows], q))
                if eligible_rows else 0.0
                for key, q in (("p50", 0.5), ("p75", 0.75), ("p90", 0.9), ("p95", 0.95))
            },
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(args.output_dir / "rec_item_drift.csv", index=False)
    write_json(args.output_dir / "rec_item_drift_summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
