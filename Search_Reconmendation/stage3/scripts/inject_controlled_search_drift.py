#!/usr/bin/env python3
"""Inject a persistent, t0-defined query-item concept drift into Search events."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def stable_uniform(seed: int, value: str) -> float:
    digest = hashlib.sha1(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--base-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-start", type=int, default=1317427200)  # 2011-10-01
    parser.add_argument("--drift-start", type=int, default=1325376000)  # 2012-01-01
    parser.add_argument(
        "--drift-end", type=int, default=0,
        help="Exclusive end timestamp; 0 keeps the changed relation persistent.",
    )
    parser.add_argument("--minimum-base-events", type=int, default=5)
    parser.add_argument("--item-fraction", type=float, default=0.30)
    parser.add_argument("--event-rate", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mapping = json.loads(args.base_mapping.read_text(encoding="utf-8"))
    query_rows = list(read_jsonl(args.source_dir / "queries.jsonl"))
    query_id = {str(row["query"]): int(row["query_id"]) for row in query_rows}
    search_rows = list(read_jsonl(args.source_dir / "search_interactions.jsonl"))

    baseline: dict[str, Counter] = defaultdict(Counter)
    for row in search_rows:
        timestamp = int(row["timestamp"])
        if args.calibration_start <= timestamp < args.drift_start and row.get("split") == "train":
            baseline[str(row["item_id"])][str(row["query"])] += 1

    eligible = [
        item_id for item_id, counts in baseline.items()
        if sum(counts.values()) >= args.minimum_base_events and item_id in mapping
    ]
    eligible.sort(key=lambda item: (-sum(baseline[item].values()), item))
    selected_count = max(1, round(len(eligible) * args.item_fraction))
    selected = eligible[:selected_count]

    groups: dict[int, list[str]] = defaultdict(list)
    for item_id in selected:
        groups[int(mapping[item_id]["prefix_codes"][0])].append(item_id)
    donor_query = {}
    dropped_singletons = []
    for _, items in sorted(groups.items()):
        items.sort()
        dominant = [baseline[item].most_common(1)[0][0] for item in items]
        if len(set(dominant)) < 2:
            dropped_singletons.extend(items)
            continue
        for offset, item_id in enumerate(items):
            replacement = None
            for step in range(1, len(items)):
                candidate = dominant[(offset + step) % len(items)]
                if candidate != dominant[offset]:
                    replacement = candidate
                    break
            if replacement is not None:
                donor_query[item_id] = replacement
    selected = sorted(donor_query)

    changed_events = 0
    changed_by_window = Counter()
    with (args.output_dir / "search_interactions.jsonl").open("w", encoding="utf-8") as handle:
        for row in search_rows:
            timestamp = int(row["timestamp"])
            item_id = str(row["item_id"])
            in_period = timestamp >= args.drift_start and (
                args.drift_end <= 0 or timestamp < args.drift_end
            )
            draw = stable_uniform(args.seed, str(row["interaction_id"]))
            if in_period and item_id in donor_query and draw < args.event_rate:
                replacement = donor_query[item_id]
                row = dict(row)
                row["query"] = replacement
                row["query_id"] = query_id[replacement]
                row["controlled_drift"] = True
                changed_events += 1
                period = str(pd.Period(pd.to_datetime(timestamp, unit="s"), freq="Q"))
                changed_by_window[period] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for name in (
        "items.jsonl", "item_ids.jsonl", "queries.jsonl", "users.jsonl", "stats.json",
        "rec_interactions_chrono.jsonl",
    ):
        shutil.copy2(args.source_dir / name, args.output_dir / name)
    manifest = {
        "type": "persistent controlled Search query-item drift",
        "selection_uses": "C8 only",
        "calibration_start": args.calibration_start,
        "drift_start": args.drift_start,
        "drift_end": args.drift_end,
        "eligible_items": len(eligible),
        "requested_selected_items": selected_count,
        "effective_selected_items": len(selected),
        "dropped_same-query_groups": len(dropped_singletons),
        "event_rate": args.event_rate,
        "changed_events": changed_events,
        "changed_events_by_quarter": dict(sorted(changed_by_window.items())),
        "selected_items": selected,
        "donor_query": donor_query,
        "seed": args.seed,
    }
    (args.output_dir / "controlled_drift_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key not in {"selected_items", "donor_query"}}, indent=2))


if __name__ == "__main__":
    main()
