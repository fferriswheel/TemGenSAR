#!/usr/bin/env python3
"""Audit behavioral and direct-alignment supervision for changed Rec SIDs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarterly-dir", type=Path, required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument("--eval-window", required=True)
    parser.add_argument("--base-mapping", type=Path, required=True)
    parser.add_argument("--current-mapping", type=Path, required=True)
    parser.add_argument("--update-stats", type=Path, required=True)
    parser.add_argument("--alignment-repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def distribution(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "items": int(len(array)),
        "zero_items": int((array == 0).sum()),
        "mean": float(array.mean()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "max": int(array.max()),
        "total_events": int(array.sum()),
    }


def main() -> None:
    args = parse_args()
    maps = json.loads((args.quarterly_dir / "id_maps.json").read_text(encoding="utf-8"))
    asin_to_int = {str(key): int(value) for key, value in maps["asin_to_int"].items()}
    base = json.loads(args.base_mapping.read_text(encoding="utf-8"))
    current = json.loads(args.current_mapping.read_text(encoding="utf-8"))
    stats = json.loads(args.update_stats.read_text(encoding="utf-8"))
    selected = set(map(str, stats["selected_item_ids"]))
    recoded = {
        item_id
        for item_id in base
        if base[item_id]["rec_suffix_codes"] != current[item_id]["rec_suffix_codes"]
    }
    if not recoded <= selected:
        raise ValueError("Recoded items must be a subset of selected items")

    def target_counts(window: str, split: str) -> Counter[int]:
        frame = pd.read_pickle(args.quarterly_dir / "windows" / window / f"rec_{split}.pkl")
        return Counter(map(int, frame["item_id"]))

    update_counts = target_counts(args.window, "update")
    validation_counts = target_counts(args.window, "eval")
    test_counts = target_counts(args.eval_window, "eval")

    def summarize(items: set[str]) -> dict:
        integer_ids = [asin_to_int[item_id] for item_id in sorted(items)]
        return {
            "item_ids": sorted(items),
            "update_target_support": distribution([update_counts[item] for item in integer_ids]),
            "validation_target_support": distribution(
                [validation_counts[item] for item in integer_ids]
            ),
            "test_target_support": distribution([test_counts[item] for item in integer_ids]),
        }

    payload = {
        "window": args.window,
        "eval_window": args.eval_window,
        "selected": summarize(selected),
        "recoded": summarize(recoded),
        "direct_rec_alignment_per_item": {
            "Item2Index": args.alignment_repeat,
            "Index2Item": args.alignment_repeat,
            "total": 2 * args.alignment_repeat,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    printable = {key: value for key, value in payload.items() if key not in {"selected", "recoded"}}
    printable["selected"] = {
        key: value for key, value in payload["selected"].items() if key != "item_ids"
    }
    printable["recoded"] = {
        key: value for key, value in payload["recoded"].items() if key != "item_ids"
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
