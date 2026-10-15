#!/usr/bin/env python
"""Filter temporal evaluation targets to items seen by the same task at t0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_seen(path: Path, asin_to_int: dict[str, int]) -> set[int]:
    seen: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            asin = json.loads(line)["item_id"]
            if asin in asin_to_int:
                seen.add(int(asin_to_int[asin]))
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarterly-dir", type=Path, required=True)
    parser.add_argument("--t0-snapshot", type=Path, required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.quarterly_dir.resolve()
    window_dir = source / "windows" / args.window
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (source / "id_maps.json").open("r", encoding="utf-8") as handle:
        asin_to_int = json.load(handle)["asin_to_int"]

    report: dict[str, object] = {
        "window": args.window,
        "definition": "positive target item occurs in the same task's raw t0 representation signal",
    }
    signal_files = {
        "rec": args.t0_snapshot / "rec_interactions_chrono.jsonl",
        "src": args.t0_snapshot / "search_interactions.jsonl",
    }
    for task, signal_path in signal_files.items():
        seen = load_seen(signal_path, asin_to_int)
        frame = pd.read_pickle(window_dir / f"{task}_eval.pkl")
        filtered = frame[frame["item_id"].astype(int).isin(seen)].reset_index(drop=True)
        filtered.to_pickle(output / f"{task}_eval.pkl")
        report[task] = {
            "t0_seen_items": len(seen),
            "all_rows": len(frame),
            "seen_rows": len(filtered),
            "seen_fraction": len(filtered) / len(frame) if len(frame) else 0.0,
        }

    with (output / "seen_filter_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
