#!/usr/bin/env python3
"""Evaluate a Rec detector against controlled labels without using them for selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from run_temporal_drift_detection import quarterly_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-drift-csv", type=Path, required=True)
    parser.add_argument("--injected-drift-csv", type=Path, required=True)
    parser.add_argument("--controlled-data-dir", type=Path, required=True)
    parser.add_argument("--pair", action="append", nargs=2, metavar=("LEFT", "RIGHT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def event_coverage(counter: Counter, items: set[str]) -> dict[str, int | float]:
    total = sum(counter.values())
    covered = sum(value for item, value in counter.items() if item in items)
    return {
        "unique_items": len(counter),
        "covered_items": len(set(counter) & items),
        "item_coverage": len(set(counter) & items) / max(len(counter), 1),
        "events": total,
        "covered_events": covered,
        "event_coverage": covered / max(total, 1),
    }


def main() -> None:
    args = parse_args()
    clean = pd.read_csv(args.clean_drift_csv, dtype={"item_id": str})
    injected = pd.read_csv(args.injected_drift_csv, dtype={"item_id": str})
    windows = quarterly_protocol()[0]
    requested = {(left, right) for left, right in args.pair}
    events = {pair: (Counter(), Counter()) for pair in requested}
    next_target_events = {pair: Counter() for pair in requested}

    with (args.controlled_data_dir / "rec_interactions_chrono.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("controlled_rec_drift"):
                continue
            timestamp = pd.to_datetime(int(row["timestamp"]), unit="s")
            source = str(row["controlled_rec_original_item"])
            target = str(row["item_id"])
            for pair in requested:
                _, right = pair
                start, end = (pd.Timestamp(value) for value in windows[right])
                if start <= timestamp < end:
                    events[pair][0][source] += 1
                    events[pair][1][target] += 1
                if right.startswith("W"):
                    next_name = f"W{int(right[1:]) + 1}"
                    if next_name in windows:
                        next_start, next_end = (pd.Timestamp(value) for value in windows[next_name])
                        if next_start <= timestamp < next_end:
                            next_target_events[pair][target] += 1

    rows = []
    for left, right in args.pair:
        clean_pair = clean[(clean["left"] == left) & (clean["right"] == right)]
        injected_pair = injected[(injected["left"] == left) & (injected["right"] == right)]
        clean_selected = set(clean_pair.loc[clean_pair["selected_for_update"], "item_id"])
        selected = set(injected_pair.loc[injected_pair["selected_for_update"], "item_id"])
        eligible = set(injected_pair.loc[injected_pair["eligible"], "item_id"])
        support_control = (
            set(injected_pair.loc[injected_pair["selected_for_support_control"], "item_id"])
            if "selected_for_support_control" in injected_pair else set()
        )
        source, target = events[(left, right)]
        union = clean_selected | selected
        row = {
            "left": left,
            "right": right,
            "eligible_items": len(eligible),
            "selected_items": len(selected),
            "clean_selected_items": len(clean_selected),
            "selection_intersection": len(clean_selected & selected),
            "selection_jaccard": len(clean_selected & selected) / max(len(union), 1),
            "injected_only_selected": len(selected - clean_selected),
            "clean_only_selected": len(clean_selected - selected),
            "source_eligible": event_coverage(source, eligible),
            "target_eligible": event_coverage(target, eligible),
            "source_selected": event_coverage(source, selected),
            "target_selected": event_coverage(target, selected),
            "next_window_target_selected": event_coverage(next_target_events[(left, right)], selected),
        }
        if support_control:
            row["source_support_control"] = event_coverage(source, support_control)
            row["target_support_control"] = event_coverage(target, support_control)
            row["next_window_target_support_control"] = event_coverage(
                next_target_events[(left, right)], support_control
            )
        rows.append(row)

    payload = {"pairs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
