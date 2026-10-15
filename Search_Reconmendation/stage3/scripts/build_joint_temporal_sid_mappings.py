#!/usr/bin/env python3
"""Compose independently updated Search and Rec suffix mappings."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-mapping", type=Path, required=True)
    parser.add_argument("--search-mapping", type=Path, required=True)
    parser.add_argument("--rec-mapping", type=Path, required=True)
    parser.add_argument("--search-random-mapping", type=Path, required=True)
    parser.add_argument("--rec-random-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_row(prefix: list[int], search: list[int], rec: list[int]) -> dict:
    return {
        "prefix": [f"<P{level}_{code}>" for level, code in enumerate(prefix, start=1)],
        "search_suffix": [f"<S{level}_{code}>" for level, code in enumerate(search, start=1)],
        "rec_suffix": [f"<R{level}_{code}>" for level, code in enumerate(rec, start=1)],
        "search_sid": [
            *[f"<P{level}_{code}>" for level, code in enumerate(prefix, start=1)],
            *[f"<S{level}_{code}>" for level, code in enumerate(search, start=1)],
        ],
        "rec_sid": [
            *[f"<P{level}_{code}>" for level, code in enumerate(prefix, start=1)],
            *[f"<R{level}_{code}>" for level, code in enumerate(rec, start=1)],
        ],
        "prefix_codes": prefix,
        "search_suffix_codes": search,
        "rec_suffix_codes": rec,
    }


def validate_task_mapping(base: dict[str, dict], update: dict[str, dict], task: str) -> None:
    if set(base) != set(update):
        raise ValueError(f"{task} item universe differs from base")
    other = "rec_suffix_codes" if task == "search" else "search_suffix_codes"
    for item, base_row in base.items():
        row = update[item]
        if row["prefix_codes"] != base_row["prefix_codes"]:
            raise ValueError(f"{task} mapping changed prefix for {item}")
        if row[other] != base_row[other]:
            raise ValueError(f"{task} mapping changed non-target suffix for {item}")


def compose(base: dict[str, dict], search: dict[str, dict], rec: dict[str, dict]) -> dict[str, dict]:
    return {
        item: normalized_row(
            list(base[item]["prefix_codes"]),
            list(search[item]["search_suffix_codes"]),
            list(rec[item]["rec_suffix_codes"]),
        )
        for item in base
    }


def collision(mapping: dict[str, dict], field: str) -> dict[str, int | float]:
    counts = Counter(tuple(row[field]) for row in mapping.values())
    total = len(mapping)
    collided = sum(count for count in counts.values() if count > 1)
    return {
        "unique": len(counts),
        "duplicate_fraction": (total - len(counts)) / total,
        "collided_item_fraction": collided / total,
        "groups": sum(count > 1 for count in counts.values()),
        "max_bucket": max(counts.values()),
    }


def changed(base: dict[str, dict], mapping: dict[str, dict], field: str) -> set[str]:
    return {item for item in base if base[item][field] != mapping[item][field]}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = load(args.base_mapping)
    search = load(args.search_mapping)
    rec = load(args.rec_mapping)
    search_random = load(args.search_random_mapping)
    rec_random = load(args.rec_random_mapping)
    for mapping, task in [
        (search, "search"), (rec, "rec"),
        (search_random, "search-random"), (rec_random, "rec-random"),
    ]:
        validate_task_mapping(base, mapping, task.split("-")[0])

    mappings = {
        "J0_fixed": base,
        "JS_search": compose(base, search, base),
        "JR_rec": compose(base, base, rec),
        "JB_joint": compose(base, search, rec),
        "JRandom_joint": compose(base, search_random, rec_random),
    }
    summary: dict[str, dict] = {}
    all_changed: set[str] = set()
    all_search_changed: set[str] = set()
    all_rec_changed: set[str] = set()
    for name, mapping in mappings.items():
        path = args.output_dir / f"{name}_sid_mapping.json"
        path.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
        search_changed = changed(base, mapping, "search_suffix_codes")
        rec_changed = changed(base, mapping, "rec_suffix_codes")
        all_changed.update(search_changed | rec_changed)
        all_search_changed.update(search_changed)
        all_rec_changed.update(rec_changed)
        summary[name] = {
            "mapping": str(path),
            "search_changed": len(search_changed),
            "rec_changed": len(rec_changed),
            "changed_union": len(search_changed | rec_changed),
            "changed_intersection": len(search_changed & rec_changed),
            "search_collision": collision(mapping, "search_sid"),
            "rec_collision": collision(mapping, "rec_sid"),
        }
    (args.output_dir / "shared_alignment_items.json").write_text(
        json.dumps(sorted(all_changed), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "shared_search_alignment_items.json").write_text(
        json.dumps(sorted(all_search_changed), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "shared_rec_alignment_items.json").write_text(
        json.dumps(sorted(all_rec_changed), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["shared_alignment"] = {
        "items": len(all_changed),
        "path": str(args.output_dir / "shared_alignment_items.json"),
        "definition": "union of task-specific SID changes across all compared mappings",
        "search_items": len(all_search_changed),
        "search_path": str(args.output_dir / "shared_search_alignment_items.json"),
        "rec_items": len(all_rec_changed),
        "rec_path": str(args.output_dir / "shared_rec_alignment_items.json"),
    }
    (args.output_dir / "joint_mapping_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
