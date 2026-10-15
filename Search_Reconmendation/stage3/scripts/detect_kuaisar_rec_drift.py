#!/usr/bin/env python3
"""Rank natural KuaiSAR Rec drift items under fixed update budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transitions", default="0:1,1:2,2:3,3:4")
    parser.add_argument("--budgets", default="75,150,300")
    parser.add_argument("--minimum-side-support", type=int, default=5)
    parser.add_argument("--random-seeds", default="2024,2025,2026")
    return parser.parse_args()


def percentile_rank(counts: pd.Series) -> pd.Series:
    active = counts > 0
    result = pd.Series(np.ones(len(counts), dtype=np.float64), index=counts.index)
    if active.any():
        result.loc[active] = counts.loc[active].rank(
            method="average", ascending=False, pct=True
        )
    return result


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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(
        args.source_dir / "rec_positive_events.parquet",
        columns=["item_id", "window_id"],
    )
    catalog = pd.read_parquet(
        args.source_dir / "catalog.parquet", columns=["item_id"]
    )
    item_ids = catalog["item_id"].astype(str)
    budgets = [int(value) for value in args.budgets.split(",") if value]
    seeds = [int(value) for value in args.random_seeds.split(",") if value]
    transitions = [
        tuple(int(part) for part in value.split(":"))
        for value in args.transitions.split(",")
        if value
    ]

    counts = (
        events.groupby(["window_id", "item_id"]).size().rename("events").reset_index()
    )
    counts["item_id"] = counts["item_id"].astype(str)
    summary = {"score": "abs(rank_left-rank_right)*sqrt(right_events)", "transitions": {}}
    for left, right in transitions:
        left_counts = counts[counts["window_id"] == left].set_index("item_id")["events"]
        right_counts = counts[counts["window_id"] == right].set_index("item_id")["events"]
        frame = pd.DataFrame({"item_id": item_ids})
        frame["left"] = f"W{left}"
        frame["right"] = f"W{right}"
        frame["left_events"] = frame["item_id"].map(left_counts).fillna(0).astype(int)
        frame["right_events"] = frame["item_id"].map(right_counts).fillna(0).astype(int)
        frame["event_support"] = frame["left_events"] + frame["right_events"]
        frame["support_bucket"] = frame["event_support"].map(support_bucket)
        frame["rank_left"] = percentile_rank(frame["left_events"])
        frame["rank_right"] = percentile_rank(frame["right_events"])
        frame["drift_rank_score"] = (frame["rank_left"] - frame["rank_right"]).abs()
        frame["detector_score"] = frame["drift_rank_score"] * np.sqrt(frame["right_events"])
        frame["eligible"] = (
            (frame["left_events"] >= args.minimum_side_support)
            & (frame["right_events"] >= args.minimum_side_support)
        )
        eligible = frame[frame["eligible"]].copy()
        transition_key = f"W{left}_W{right}"
        transition_dir = args.output_dir / transition_key
        transition_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(transition_dir / "scores.csv", index=False)
        summary["transitions"][transition_key] = {
            "eligible_items": len(eligible),
            "left_events": int(frame["left_events"].sum()),
            "right_events": int(frame["right_events"].sum()),
            "budgets": {},
        }
        for budget in budgets:
            if budget > len(eligible):
                raise ValueError(
                    f"Budget {budget} exceeds {len(eligible)} eligible items for {transition_key}"
                )
            strategy_outputs = {}
            detector = eligible.sort_values(
                ["detector_score", "right_events", "item_id"],
                ascending=[False, False, True],
            ).head(budget)
            support = eligible.sort_values(
                ["right_events", "detector_score", "item_id"],
                ascending=[False, False, True],
            ).head(budget)
            selections = {"detector": set(detector["item_id"]), "support": set(support["item_id"])}
            for seed in seeds:
                sampled = eligible.sample(n=budget, random_state=seed)
                selections[f"random_seed{seed}"] = set(sampled["item_id"])
            for strategy, selected in selections.items():
                output = frame.copy()
                output["selected_for_update"] = output["item_id"].isin(selected)
                path = transition_dir / f"{strategy}_k{budget}.csv"
                output.to_csv(path, index=False)
                strategy_outputs[strategy] = {
                    "path": str(path.resolve()),
                    "selected": int(output["selected_for_update"].sum()),
                    "mean_detector_score": float(
                        output.loc[output["selected_for_update"], "detector_score"].mean()
                    ),
                    "right_events": int(
                        output.loc[output["selected_for_update"], "right_events"].sum()
                    ),
                }
            summary["transitions"][transition_key]["budgets"][str(budget)] = strategy_outputs

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
