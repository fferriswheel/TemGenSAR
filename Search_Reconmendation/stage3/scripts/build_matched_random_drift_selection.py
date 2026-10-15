#!/usr/bin/env python3
"""Replace one temporal drift selection with a support-matched random control."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from materialize_v13b_temporal_rec_sid_sticky import matched_random_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, dtype={"item_id": str})
    pair_mask = (frame["left"] == args.left) & (frame["right"] == args.right)
    pair = frame.loc[pair_mask].copy()
    selected = pair[pair["selected_for_update"]].copy()
    random_items = matched_random_items(pair, selected, args.seed)
    if len(random_items) != len(selected):
        raise ValueError("Matched-random selection size mismatch")

    frame.loc[pair_mask, "selected_for_update"] = frame.loc[
        pair_mask, "item_id"
    ].isin(random_items)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    random_frame = pair[pair["item_id"].isin(random_items)]
    summary = {
        "left": args.left,
        "right": args.right,
        "seed": args.seed,
        "selected_items": len(selected),
        "matched_random_items": len(random_items),
        "source_support_buckets": dict(Counter(selected["support_bucket"])),
        "random_support_buckets": dict(Counter(random_frame["support_bucket"])),
        "source_item_ids": sorted(selected["item_id"].tolist()),
        "random_item_ids": sorted(random_items),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
