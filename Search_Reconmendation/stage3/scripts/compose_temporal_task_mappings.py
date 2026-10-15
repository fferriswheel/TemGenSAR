#!/usr/bin/env python3
"""Compose task-local Search and Rec SID updates over an operational mapping."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from build_joint_temporal_sid_mappings import normalized_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-mapping", type=Path, required=True)
    parser.add_argument("--search-mapping", type=Path, required=True)
    parser.add_argument("--rec-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--project-task-fields",
        action="store_true",
        help="Ignore opposite-task suffix differences and project only each mapping's owned task.",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def collision(mapping: dict[str, dict], field: str) -> dict[str, int | float]:
    counts = Counter(tuple(row[field]) for row in mapping.values())
    total = len(mapping)
    return {
        "unique": len(counts),
        "duplicate_fraction": (total - len(counts)) / total,
        "groups": sum(count > 1 for count in counts.values()),
        "max_bucket": max(counts.values()),
    }


def main() -> None:
    args = parse_args()
    base, search, rec = map(load, [args.base_mapping, args.search_mapping, args.rec_mapping])
    if not (set(base) == set(search) == set(rec)):
        raise ValueError("Mappings have different item universes")
    result = {}
    search_changed, rec_changed = set(), set()
    for item, old in base.items():
        search_row, rec_row = search[item], rec[item]
        if search_row["prefix_codes"] != old["prefix_codes"]:
            raise ValueError(f"Search mapping changed prefix for {item}")
        if rec_row["prefix_codes"] != old["prefix_codes"]:
            raise ValueError(f"Rec mapping changed prefix for {item}")
        if not args.project_task_fields and search_row["rec_suffix_codes"] != old["rec_suffix_codes"]:
            raise ValueError(f"Search mapping changed Rec suffix for {item}")
        if not args.project_task_fields and rec_row["search_suffix_codes"] != old["search_suffix_codes"]:
            raise ValueError(f"Rec mapping changed Search suffix for {item}")
        search_codes = list(search_row["search_suffix_codes"])
        rec_codes = list(rec_row["rec_suffix_codes"])
        result[item] = normalized_row(list(old["prefix_codes"]), search_codes, rec_codes)
        if search_codes != old["search_suffix_codes"]:
            search_changed.add(item)
        if rec_codes != old["rec_suffix_codes"]:
            rec_changed.add(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    stats = {
        "base_mapping": str(args.base_mapping),
        "search_mapping": str(args.search_mapping),
        "rec_mapping": str(args.rec_mapping),
        "items": len(result),
        "search_changed": len(search_changed),
        "rec_changed": len(rec_changed),
        "changed_union": len(search_changed | rec_changed),
        "changed_intersection": len(search_changed & rec_changed),
        "search_changed_items": sorted(search_changed),
        "rec_changed_items": sorted(rec_changed),
        "prefix_exactly_preserved": True,
        "project_task_fields": args.project_task_fields,
        "search_collision": collision(result, "search_sid"),
        "rec_collision": collision(result, "rec_sid"),
    }
    args.output.with_suffix(".stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
