#!/usr/bin/env python3
"""Build fixed-budget Search/Rec alignment schedules with repeated recoded items."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-pool", type=Path, required=True)
    parser.add_argument("--base-mapping", type=Path, required=True)
    parser.add_argument("--current-mapping", type=Path, required=True)
    parser.add_argument("--protected-update-stats", type=Path, action="append", default=[])
    parser.add_argument("--repeat-factor", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def stable_key(item_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()


def main() -> None:
    args = parse_args()
    if args.repeat_factor < 1:
        raise ValueError("repeat-factor must be at least 1")

    base_pool = [str(value) for value in json.loads(args.base_pool.read_text(encoding="utf-8"))]
    if len(base_pool) != len(set(base_pool)):
        raise ValueError("base alignment pool must contain unique items")
    base = json.loads(args.base_mapping.read_text(encoding="utf-8"))
    current = json.loads(args.current_mapping.read_text(encoding="utf-8"))
    if set(base) != set(current):
        raise ValueError("base and current mappings have different item universes")

    recoded = sorted(
        item_id
        for item_id in base
        if base[item_id]["rec_suffix_codes"] != current[item_id]["rec_suffix_codes"]
    )
    missing = set(recoded) - set(base_pool)
    if missing:
        raise ValueError(f"recoded items missing from base pool: {sorted(missing)[:5]}")

    protected = set(recoded)
    for path in args.protected_update_stats:
        stats = json.loads(path.read_text(encoding="utf-8"))
        protected.update(str(value) for value in stats["selected_item_ids"])
    protected &= set(base_pool)

    extra_occurrences = (args.repeat_factor - 1) * len(recoded)
    removable = sorted(
        set(base_pool) - protected,
        key=lambda item_id: stable_key(item_id, args.seed),
    )
    if extra_occurrences > len(removable):
        raise ValueError(
            f"Need {extra_occurrences} removable items but only {len(removable)} are available"
        )
    removed = set(removable[:extra_occurrences])
    rec_schedule = [item_id for item_id in base_pool if item_id not in removed]
    for _ in range(args.repeat_factor - 1):
        rec_schedule.extend(recoded)
    if len(rec_schedule) != len(base_pool):
        raise RuntimeError("fixed-budget Rec schedule length changed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    search_path = args.output_dir / "search_alignment_schedule.json"
    rec_path = args.output_dir / "rec_alignment_schedule.json"
    search_path.write_text(json.dumps(base_pool, indent=2), encoding="utf-8")
    rec_path.write_text(json.dumps(rec_schedule, indent=2), encoding="utf-8")
    summary = {
        "base_pool": str(args.base_pool),
        "base_mapping": str(args.base_mapping),
        "current_mapping": str(args.current_mapping),
        "repeat_factor": args.repeat_factor,
        "budget": len(base_pool),
        "recoded_items": len(recoded),
        "protected_items": len(protected),
        "removed_fill_items": len(removed),
        "search_occurrences": len(base_pool),
        "search_unique_items": len(set(base_pool)),
        "rec_occurrences": len(rec_schedule),
        "rec_unique_items": len(set(rec_schedule)),
        "recoded_item_occurrences_each": args.repeat_factor,
        "seed": args.seed,
        "search_schedule": str(search_path),
        "rec_schedule": str(rec_path),
        "recoded_item_ids": recoded,
        "removed_item_ids": sorted(removed),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if not key.endswith("_ids")}, indent=2))


if __name__ == "__main__":
    main()
