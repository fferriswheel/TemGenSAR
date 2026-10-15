#!/usr/bin/env python3
"""Materialize a fixed-budget selection policy in a temporal drift CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument(
        "--policy",
        choices=["score_topk", "impact_topk", "support_topk", "column"],
        required=True,
    )
    parser.add_argument("--column", default="selected_for_support_control")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--support-power", type=float, default=0.5)
    parser.add_argument("--minimum-current-events", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, dtype={"item_id": str})
    pair_mask = (frame["left"] == args.left) & (frame["right"] == args.right)
    pair = frame.loc[pair_mask].copy()
    if args.policy == "score_topk":
        candidates = pair[pair["eligible"]].sort_values(
            ["drift_rank_score", "event_support", "item_id"],
            ascending=[False, False, True],
        )
        selected = set(candidates.head(args.budget)["item_id"])
    elif args.policy == "impact_topk":
        if args.support_power < 0:
            raise ValueError("support-power must be non-negative")
        candidates = pair[
            pair["eligible"] & (pair["right_events"] >= args.minimum_current_events)
        ].copy()
        candidates["selection_utility_score"] = candidates["drift_rank_score"] * (
            candidates["right_events"].astype(float) ** args.support_power
        )
        candidates = candidates.sort_values(
            ["selection_utility_score", "right_events", "drift_rank_score", "item_id"],
            ascending=[False, False, False, True],
        )
        selected = set(candidates.head(args.budget)["item_id"])
        frame.loc[pair_mask, "selection_utility_score"] = frame.loc[
            pair_mask, "drift_rank_score"
        ] * (frame.loc[pair_mask, "right_events"].astype(float) ** args.support_power)
    elif args.policy == "support_topk":
        candidates = pair[pair["eligible"]].sort_values(
            ["event_total_support", "transition_total_support", "item_id"],
            ascending=[False, False, True],
        )
        selected = set(candidates.head(args.budget)["item_id"])
    else:
        if args.column not in pair:
            raise ValueError(f"Missing selection column: {args.column}")
        candidates = pair[pair[args.column].astype(bool)]
        if len(candidates) != args.budget:
            raise ValueError(
                f"Column {args.column} selects {len(candidates)} items; expected {args.budget}"
            )
        selected = set(candidates["item_id"])
    if len(selected) != args.budget:
        raise ValueError(f"Selected {len(selected)} items; expected {args.budget}")

    frame.loc[pair_mask, "selected_for_update"] = frame.loc[
        pair_mask, "item_id"
    ].isin(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    summary = {
        "source": str(args.input),
        "left": args.left,
        "right": args.right,
        "policy": args.policy,
        "column": args.column if args.policy == "column" else None,
        "support_power": args.support_power if args.policy == "impact_topk" else None,
        "minimum_current_events": (
            args.minimum_current_events if args.policy == "impact_topk" else None
        ),
        "budget": args.budget,
        "selected_items": sorted(selected),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "selected_items"}, indent=2))


if __name__ == "__main__":
    main()
