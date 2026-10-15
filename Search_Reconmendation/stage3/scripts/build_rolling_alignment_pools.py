#!/usr/bin/env python3
"""Build task-local alignment pools from independent rolling SID transitions."""

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transition",
        action="append",
        nargs=3,
        metavar=("NAME", "PREVIOUS_MAPPING", "CURRENT_MAPPING"),
        required=True,
        help="Repeat for each rolling arm that changes SID in this window.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    search_union = set()
    rec_union = set()
    summaries = []
    universe = None
    for name, previous_path, current_path in args.transition:
        previous_path = Path(previous_path)
        current_path = Path(current_path)
        previous = load(previous_path)
        current = load(current_path)
        if set(previous) != set(current):
            raise ValueError(f"Item universe differs for transition {name}")
        if universe is None:
            universe = set(previous)
        elif universe != set(previous):
            raise ValueError(f"Item universe differs across transitions at {name}")

        search_changed = []
        rec_changed = []
        for item_id in sorted(previous):
            old = previous[item_id]
            new = current[item_id]
            if old["search_sid"][:2] != new["search_sid"][:2]:
                raise ValueError(f"Search prefix changed for {name}: {item_id}")
            if old["rec_sid"][:2] != new["rec_sid"][:2]:
                raise ValueError(f"Rec prefix changed for {name}: {item_id}")
            if old["search_sid"] != new["search_sid"]:
                search_changed.append(item_id)
            if old["rec_sid"] != new["rec_sid"]:
                rec_changed.append(item_id)

        search_union.update(search_changed)
        rec_union.update(rec_changed)
        summaries.append(
            {
                "name": name,
                "previous_mapping": str(previous_path),
                "current_mapping": str(current_path),
                "search_changed": len(search_changed),
                "rec_changed": len(rec_changed),
                "changed_union": len(set(search_changed) | set(rec_changed)),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    search_items = sorted(search_union)
    rec_items = sorted(rec_union)
    (args.output_dir / "search_alignment_items.json").write_text(
        json.dumps(search_items, indent=2), encoding="utf-8"
    )
    (args.output_dir / "rec_alignment_items.json").write_text(
        json.dumps(rec_items, indent=2), encoding="utf-8"
    )
    summary = {
        "transitions": summaries,
        "search_alignment_items": len(search_items),
        "rec_alignment_items": len(rec_items),
        "alignment_union": len(set(search_items) | set(rec_items)),
        "alignment_intersection": len(set(search_items) & set(rec_items)),
    }
    (args.output_dir / "alignment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
