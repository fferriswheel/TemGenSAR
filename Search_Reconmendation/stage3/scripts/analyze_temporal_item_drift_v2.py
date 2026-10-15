#!/usr/bin/env python3
"""Bidirectional item-drift detector with transition-counterpart propagation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from run_temporal_drift_detection import WINDOWS, counter_js, quarterly_protocol, write_json


PAIRS = [("B4", "W1"), ("W1", "W2"), ("W2", "W3")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-peak-events", type=int, default=5)
    parser.add_argument("--minimum-peak-transitions", type=int, default=3)
    parser.add_argument("--top-fraction", type=float, default=0.30)
    parser.add_argument(
        "--selection-budget",
        type=int,
        default=0,
        help="Fixed total selection budget per pair; zero uses top-fraction.",
    )
    parser.add_argument("--counterpart-fraction", type=float, default=0.30)
    parser.add_argument("--gain-fraction", type=float, default=0.50)
    parser.add_argument("--popularity-weight", type=float, default=0.40)
    parser.add_argument("--transition-weight", type=float, default=0.60)
    parser.add_argument("--max-popularity-log-ratio", type=float, default=3.0)
    parser.add_argument("--minimum-counterpart-similarity", type=float, default=0.05)
    parser.add_argument("--rate-smoothing", type=float, default=0.5)
    parser.add_argument("--window-mode", choices=["halfyear", "quarterly"], default="halfyear")
    return parser.parse_args()


def load_rec(path: Path) -> pd.DataFrame:
    frame = pd.read_json(path, lines=True)
    if not np.issubdtype(frame["timestamp"].dtype, np.datetime64):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s")
    sort_columns = [column for column in ["user_id", "timestamp", "position"] if column in frame]
    return frame.sort_values(sort_columns)


def aggregate(frame: pd.DataFrame) -> tuple[Counter, defaultdict, defaultdict]:
    popularity = Counter(frame["item_id"].astype(str))
    outgoing: defaultdict[str, Counter] = defaultdict(Counter)
    relation: defaultdict[str, Counter] = defaultdict(Counter)
    for _, group in frame.groupby("user_id", sort=False):
        items = group["item_id"].astype(str).tolist()
        for source, target in zip(items, items[1:]):
            if source == target:
                continue
            outgoing[source][target] += 1
            relation[source][target] += 1
            relation[target][source] += 1
    return popularity, outgoing, relation


def support_bucket(value: int) -> str:
    if value < 5:
        return "0-4"
    if value < 10:
        return "5-9"
    if value < 20:
        return "10-19"
    if value < 50:
        return "20-49"
    return "50+"


def one_sided_js(left: Counter, right: Counter) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not left_total and not right_total:
        return 0.0
    if not left_total or not right_total:
        return 1.0
    return counter_js(left, right)


def sparse_cosine(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(float(left[key]) * float(right[key]) for key in common)
    left_norm = np.sqrt(sum(float(value) ** 2 for value in left.values()))
    right_norm = np.sqrt(sum(float(value) ** 2 for value in right.values()))
    return float(numerator / max(left_norm * right_norm, 1e-12))


def sorted_rows(rows: list[dict], score: str = "drift_rank_score") -> list[dict]:
    return sorted(rows, key=lambda row: (-float(row[score]), str(row["item_id"])))


def take_balanced(
    rows: list[dict], budget: int, gain_fraction: float
) -> list[dict]:
    gain_budget = int(round(budget * gain_fraction))
    loss_budget = budget - gain_budget
    gains = sorted_rows([row for row in rows if row["drift_direction"] == "gain"])
    losses = sorted_rows([row for row in rows if row["drift_direction"] == "loss"])
    selected = gains[:gain_budget] + losses[:loss_budget]
    selected_ids = {row["item_id"] for row in selected}
    if len(selected) < budget:
        remaining = sorted_rows([row for row in rows if row["item_id"] not in selected_ids])
        selected.extend(remaining[: budget - len(selected)])
    return selected


def select_counterparts(
    core: list[dict],
    eligible: list[dict],
    left_relation: defaultdict[str, Counter],
    right_relation: defaultdict[str, Counter],
    budget: int,
    minimum_similarity: float,
) -> tuple[list[dict], dict[str, dict[str, float | str]]]:
    if budget <= 0:
        return [], {}
    core_ids = {row["item_id"] for row in core}
    candidate_info: dict[str, dict[str, float | str]] = {}
    for seed in core:
        seed_item = str(seed["item_id"])
        seed_is_gain = seed["drift_direction"] == "gain"
        seed_counter = right_relation[seed_item] if seed_is_gain else left_relation[seed_item]
        seed_mass = sum(seed_counter.values())
        if not seed_mass:
            continue
        for candidate in eligible:
            candidate_item = str(candidate["item_id"])
            if candidate_item in core_ids:
                continue
            candidate_is_gain = candidate["drift_direction"] == "gain"
            if candidate_is_gain == seed_is_gain:
                continue
            candidate_counter = (
                right_relation[candidate_item] if candidate_is_gain else left_relation[candidate_item]
            )
            candidate_mass = sum(candidate_counter.values())
            similarity = sparse_cosine(seed_counter, candidate_counter)
            if similarity < minimum_similarity or not candidate_mass:
                continue
            mass_match = min(seed_mass, candidate_mass) / max(seed_mass, candidate_mass)
            magnitude_match = min(
                abs(float(seed["signed_popularity_log_ratio"])),
                abs(float(candidate["signed_popularity_log_ratio"])),
            )
            score = similarity * np.sqrt(mass_match) * (1.0 + min(magnitude_match, 3.0))
            previous = candidate_info.get(candidate_item)
            if previous is None or score > float(previous["counterpart_score"]):
                candidate_info[candidate_item] = {
                    "counterpart_score": float(score),
                    "counterpart_similarity": float(similarity),
                    "counterpart_parent": seed_item,
                }

    ranked = sorted(
        (row for row in eligible if row["item_id"] in candidate_info),
        key=lambda row: (
            -float(candidate_info[str(row["item_id"])]["counterpart_score"]),
            -float(row["drift_rank_score"]),
            str(row["item_id"]),
        ),
    )
    selected = ranked[:budget]
    if len(selected) < budget:
        used = core_ids | {row["item_id"] for row in selected}
        fallback = sorted_rows([row for row in eligible if row["item_id"] not in used])
        selected.extend(fallback[: budget - len(selected)])
    return selected, candidate_info


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.counterpart_fraction < 1.0:
        raise ValueError("counterpart-fraction must be in [0, 1)")
    if not 0.0 <= args.gain_fraction <= 1.0:
        raise ValueError("gain-fraction must be in [0, 1]")
    if args.popularity_weight < 0 or args.transition_weight < 0:
        raise ValueError("score weights must be non-negative")
    weight_sum = args.popularity_weight + args.transition_weight
    if weight_sum <= 0:
        raise ValueError("at least one score weight must be positive")

    windows, pairs = (
        (quarterly_protocol()[0], quarterly_protocol()[2])
        if args.window_mode == "quarterly"
        else (WINDOWS, PAIRS)
    )
    frame = load_rec(args.data_dir / "rec_interactions_chrono.jsonl")
    aggregates = {}
    for name, (start, end) in windows.items():
        rows = frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)]
        aggregates[name] = aggregate(rows)

    report = {
        "method": "bidirectional-rate-change plus transition-counterpart propagation",
        "config": {
            **vars(args),
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
        },
        "pairs": [],
    }
    output_rows = []
    for left_name, right_name in pairs:
        left_pop, left_outgoing, left_relation = aggregates[left_name]
        right_pop, right_outgoing, right_relation = aggregates[right_name]
        items = sorted(set(left_pop) | set(right_pop))
        left_total = sum(left_pop.values())
        right_total = sum(right_pop.values())
        denominator_left = left_total + args.rate_smoothing * len(items)
        denominator_right = right_total + args.rate_smoothing * len(items)
        rows = []
        for item in items:
            left_events = int(left_pop[item])
            right_events = int(right_pop[item])
            event_min = min(left_events, right_events)
            event_peak = max(left_events, right_events)
            event_total = left_events + right_events
            left_transitions = int(sum(left_relation[item].values()))
            right_transitions = int(sum(right_relation[item].values()))
            transition_min = min(left_transitions, right_transitions)
            transition_peak = max(left_transitions, right_transitions)
            transition_total = left_transitions + right_transitions
            left_rate = (left_events + args.rate_smoothing) / denominator_left
            right_rate = (right_events + args.rate_smoothing) / denominator_right
            signed_popularity = float(np.log(right_rate / left_rate))
            clipped_popularity = min(abs(signed_popularity), args.max_popularity_log_ratio)
            transition_js = one_sided_js(left_relation[item], right_relation[item])
            eligible = (
                event_peak >= args.minimum_peak_events
                and transition_peak >= args.minimum_peak_transitions
            )
            confidence = float(np.log1p(event_total))
            score = confidence * (
                args.popularity_weight * clipped_popularity
                + args.transition_weight * transition_js
            ) / weight_sum
            rows.append({
                "item_id": item,
                "left": left_name,
                "right": right_name,
                "left_events": left_events,
                "right_events": right_events,
                "event_support": event_peak,
                "event_min_support": event_min,
                "event_total_support": event_total,
                "support_bucket": support_bucket(event_peak),
                "left_transitions": left_transitions,
                "right_transitions": right_transitions,
                "transition_support": transition_peak,
                "transition_min_support": transition_min,
                "transition_total_support": transition_total,
                "signed_popularity_log_ratio": signed_popularity,
                "popularity_log_ratio": abs(signed_popularity),
                "transition_js": float(transition_js),
                "eligible": bool(eligible),
                "drift_direction": "gain" if signed_popularity >= 0 else "loss",
                "drift_rank_score": float(score),
                "selected_by_core": False,
                "selected_by_counterpart": False,
                "counterpart_score": 0.0,
                "counterpart_similarity": 0.0,
                "counterpart_parent": "",
                "selected_for_update": False,
            })

        eligible_rows = [row for row in rows if row["eligible"]]
        total_budget = (
            min(args.selection_budget, len(eligible_rows))
            if args.selection_budget > 0
            else min(
                len(eligible_rows),
                max(1, int(np.ceil(len(eligible_rows) * args.top_fraction)))
                if eligible_rows else 0,
            )
        )
        counterpart_budget = int(round(total_budget * args.counterpart_fraction))
        core_budget = total_budget - counterpart_budget
        core = take_balanced(eligible_rows, core_budget, args.gain_fraction)
        counterparts, counterpart_info = select_counterparts(
            core,
            eligible_rows,
            left_relation,
            right_relation,
            counterpart_budget,
            args.minimum_counterpart_similarity,
        )
        core_ids = {str(row["item_id"]) for row in core}
        counterpart_ids = {str(row["item_id"]) for row in counterparts} - core_ids
        selected = core_ids | counterpart_ids
        for row in rows:
            item = str(row["item_id"])
            row["selected_by_core"] = item in core_ids
            row["selected_by_counterpart"] = item in counterpart_ids
            row["selected_for_update"] = item in selected
            if item in counterpart_info:
                row.update(counterpart_info[item])
            output_rows.append(row)

        selected_rows = [row for row in rows if row["selected_for_update"]]
        support_control = sorted(
            eligible_rows,
            key=lambda row: (
                -int(row["event_total_support"]),
                -int(row["transition_total_support"]),
                str(row["item_id"]),
            ),
        )[:total_budget]
        support_control_ids = {row["item_id"] for row in support_control}
        for row in output_rows[-len(rows):]:
            row["selected_for_support_control"] = row["item_id"] in support_control_ids

        report["pairs"].append({
            "left": left_name,
            "right": right_name,
            "union_items": len(rows),
            "left_events": left_total,
            "right_events": right_total,
            "eligible_items": len(eligible_rows),
            "eligible_gain_items": sum(row["drift_direction"] == "gain" for row in eligible_rows),
            "eligible_loss_items": sum(row["drift_direction"] == "loss" for row in eligible_rows),
            "selection_budget": total_budget,
            "core_budget": core_budget,
            "counterpart_budget": counterpart_budget,
            "selected_items": len(selected),
            "selected_gain_items": sum(row["drift_direction"] == "gain" for row in selected_rows),
            "selected_loss_items": sum(row["drift_direction"] == "loss" for row in selected_rows),
            "selected_core_items": len(core_ids),
            "selected_counterpart_items": len(counterpart_ids),
            "support_control_items": len(support_control_ids),
            "counterpart_similarity": {
                key: float(np.quantile([
                    float(row["counterpart_similarity"]) for row in selected_rows
                    if row["selected_by_counterpart"]
                ], quantile)) if counterpart_ids else 0.0
                for key, quantile in (("min", 0.0), ("p50", 0.5), ("p90", 0.9), ("max", 1.0))
            },
            "score_quantiles_eligible": {
                key: float(np.quantile(
                    [row["drift_rank_score"] for row in eligible_rows], quantile
                )) if eligible_rows else 0.0
                for key, quantile in (("p50", 0.5), ("p75", 0.75), ("p90", 0.9), ("p95", 0.95))
            },
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(args.output_dir / "rec_item_drift.csv", index=False)
    write_json(args.output_dir / "rec_item_drift_summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
