#!/usr/bin/env python3
"""Build a deterministic fixed-size alignment pool containing priority items."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-ids", type=Path, required=True)
    parser.add_argument("--priority-json", type=Path, action="append", default=[])
    parser.add_argument("--priority-update-stats", type=Path, action="append", default=[])
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def stable_key(item_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()


def main() -> None:
    args = parse_args()
    universe = [
        str(json.loads(line)["item_id"])
        for line in args.item_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    universe_set = set(universe)
    priority: set[str] = set()
    for path in args.priority_json:
        priority.update(str(value) for value in json.loads(path.read_text(encoding="utf-8")))
    for path in args.priority_update_stats:
        stats = json.loads(path.read_text(encoding="utf-8"))
        priority.update(str(value) for value in stats["selected_item_ids"])
    unknown = priority - universe_set
    if unknown:
        raise ValueError(f"Unknown priority items: {sorted(unknown)[:5]}")
    if len(priority) > args.size:
        raise ValueError(f"{len(priority)} priority items exceed pool size {args.size}")

    remaining = sorted(universe_set - priority, key=lambda item: stable_key(item, args.seed))
    pool = sorted(priority) + remaining[: args.size - len(priority)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pool, indent=2), encoding="utf-8")
    summary = {
        "size": len(pool),
        "priority_items": len(priority),
        "fill_items": len(pool) - len(priority),
        "seed": args.seed,
        "output": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
