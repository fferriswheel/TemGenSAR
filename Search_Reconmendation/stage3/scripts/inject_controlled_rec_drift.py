#!/usr/bin/env python3
"""Inject persistent cohort-level Rec transition drift within semantic categories."""

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


def canonical_category(row: dict) -> tuple[str, str] | None:
    paths = [tuple(str(part) for part in path) for path in row.get("categories", []) if path]
    if not paths:
        return None
    path = sorted(paths, key=lambda value: (-len(value), value))[0]
    return path[0], path[-1]


def window_start(name: str) -> int:
    number = int(name.removeprefix("W"))
    period = pd.Period("2012Q1", freq="Q") + number - 1
    return int(period.start_time.timestamp())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drift-windows", default="W3,W6,W9")
    parser.add_argument("--event-rates", default="0.25,0.50,0.75")
    parser.add_argument("--user-fraction", type=float, default=0.30)
    parser.add_argument("--minimum-user-events", type=int, default=5)
    parser.add_argument("--minimum-group-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    windows = [value.strip() for value in args.drift_windows.split(",")]
    rates = [float(value) for value in args.event_rates.split(",")]
    if len(windows) != len(rates) or rates != sorted(rates):
        raise ValueError("drift windows and monotonically increasing event rates must align")
    stages = sorted(zip((window_start(name) for name in windows), windows, rates))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    item_rows = list(read_jsonl(args.source_dir / "items.jsonl"))
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in item_rows:
        category = canonical_category(row)
        if category is not None:
            groups[category].append(str(row["item_id"]))
    eligible_groups = {
        category: sorted(items)
        for category, items in groups.items()
        if len(items) >= args.minimum_group_size
    }
    item_group = {
        item: category for category, items in eligible_groups.items() for item in items
    }

    rec_rows = list(read_jsonl(args.source_dir / "rec_interactions_chrono.jsonl"))
    first_start = stages[0][0]
    user_events = Counter(str(row["user_id"]) for row in rec_rows)
    future_events = Counter(
        str(row["user_id"]) for row in rec_rows if int(row["timestamp"]) >= first_start
    )
    eligible_users = sorted(
        user for user, count in user_events.items()
        if count >= args.minimum_user_events and future_events[user] > 0
    )
    affected_users = {
        user for user in eligible_users
        if stable_uniform(args.seed, f"user:{user}") < args.user_fraction
    }

    donor_maps = []
    for stage_index in range(len(stages)):
        donor = {}
        for items in eligible_groups.values():
            shift = stage_index + 1
            for index, item in enumerate(items):
                donor[item] = items[(index + shift) % len(items)]
        donor_maps.append(donor)

    changed_by_window = Counter()
    changed_by_stage = Counter()
    changed_users = set()
    changed_items = set()
    total_eligible_events = 0
    with (args.output_dir / "rec_interactions_chrono.jsonl").open("w", encoding="utf-8") as handle:
        for row in rec_rows:
            timestamp = int(row["timestamp"])
            stage_index = max(
                (index for index, (start, _, _) in enumerate(stages) if timestamp >= start),
                default=-1,
            )
            user = str(row["user_id"])
            item = str(row["item_id"])
            if stage_index >= 0 and user in affected_users and item in item_group:
                total_eligible_events += 1
                event_id = str(row.get("review_line_id", f"{user}:{timestamp}:{row.get('position', 0)}"))
                rate = stages[stage_index][2]
                if stable_uniform(args.seed, f"event:{event_id}") < rate:
                    replacement = donor_maps[stage_index][item]
                    row = dict(row)
                    row["item_id"] = replacement
                    row["controlled_rec_drift"] = True
                    row["controlled_rec_original_item"] = item
                    row["controlled_rec_stage"] = stages[stage_index][1]
                    changed_users.add(user)
                    changed_items.update((item, replacement))
                    changed_by_stage[stages[stage_index][1]] += 1
                    quarter = str(pd.Period(pd.to_datetime(timestamp, unit="s"), freq="Q"))
                    changed_by_window[quarter] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for name in (
        "items.jsonl", "item_ids.jsonl", "queries.jsonl", "users.jsonl", "stats.json",
        "search_interactions.jsonl",
    ):
        shutil.copy2(args.source_dir / name, args.output_dir / name)

    manifest = {
        "type": "persistent cohort-level Rec transition drift",
        "semantic_constraint": "one-to-one item rotation within metadata root+leaf category",
        "search_unchanged": True,
        "drift_stages": [
            {"window": name, "start": start, "event_rate": rate}
            for start, name, rate in stages
        ],
        "user_fraction": args.user_fraction,
        "eligible_users": len(eligible_users),
        "affected_users_requested": len(affected_users),
        "affected_users_realized": len(changed_users),
        "eligible_category_groups": len(eligible_groups),
        "items_in_eligible_groups": len(item_group),
        "eligible_post_drift_events": total_eligible_events,
        "changed_events": sum(changed_by_stage.values()),
        "changed_items": len(changed_items),
        "changed_events_by_stage": dict(changed_by_stage),
        "changed_events_by_quarter": dict(sorted(changed_by_window.items())),
        "seed": args.seed,
    }
    (args.output_dir / "controlled_rec_drift_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
