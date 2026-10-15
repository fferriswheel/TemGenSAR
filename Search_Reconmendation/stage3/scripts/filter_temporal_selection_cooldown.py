#!/usr/bin/env python3
"""Suppress recently recoded items from a temporal drift selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-csv", type=Path, required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--task", choices=["search", "rec"], required=True)
    parser.add_argument(
        "--history-mapping", type=Path, action="append", required=True,
        help="Chronological mappings ending at the operational mapping before this update.",
    )
    parser.add_argument("--cooldown-windows", type=int, default=2)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def codes(mapping: dict, item: str, task: str) -> tuple[int, ...]:
    return tuple(int(value) for value in mapping[item][f"{task}_suffix_codes"])


def main() -> None:
    args = parse_args()
    if len(args.history_mapping) < 1:
        raise ValueError("At least one history mapping is required")
    mappings = [json.loads(path.read_text(encoding="utf-8")) for path in args.history_mapping]
    universe = set(mappings[-1])
    if any(set(mapping) != universe for mapping in mappings):
        raise ValueError("History mapping universes differ")

    recent_changes: set[str] = set()
    transitions = list(zip(mappings, mappings[1:]))[-args.cooldown_windows:]
    for old, new in transitions:
        recent_changes.update(
            item for item in universe
            if codes(old, item, args.task) != codes(new, item, args.task)
        )

    drift = pd.read_csv(args.drift_csv, dtype={"item_id": str})
    pair_mask = (drift["left"] == args.left) & (drift["right"] == args.right)
    if not pair_mask.any():
        raise ValueError(f"No drift rows for {args.left}->{args.right}")
    pair = drift.loc[pair_mask].copy()
    pair["selected_before_cooldown"] = pair["selected_for_update"].astype(bool)
    pair["cooldown_blocked"] = (
        pair["selected_before_cooldown"] & pair["item_id"].isin(recent_changes)
    )
    pair["selected_for_update"] = (
        pair["selected_before_cooldown"] & ~pair["cooldown_blocked"]
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    pair.to_csv(args.output_csv, index=False)
    summary = {
        "task": args.task,
        "drift_pair": [args.left, args.right],
        "cooldown_windows": args.cooldown_windows,
        "history_mappings": [str(path) for path in args.history_mapping],
        "recently_changed_items": len(recent_changes),
        "selected_before": int(pair["selected_before_cooldown"].sum()),
        "blocked": int(pair["cooldown_blocked"].sum()),
        "selected_after": int(pair["selected_for_update"].sum()),
        "blocked_item_ids": sorted(pair.loc[pair["cooldown_blocked"], "item_id"]),
    }
    args.output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
