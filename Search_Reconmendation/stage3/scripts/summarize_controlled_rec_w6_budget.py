#!/usr/bin/env python3
"""Summarize the W6 detector/support alignment-budget curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from summarize_controlled_rec_w6 import TASK_FILES, read_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="POLICY:BUDGET=RUN_DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for spec in args.run:
        identity, raw_path = spec.split("=", 1)
        policy, raw_budget = identity.split(":", 1)
        run_dir = Path(raw_path)
        validation = read_validation(run_dir)
        row = {"policy": policy, "budget": int(raw_budget), "run_dir": str(run_dir)}
        for task, filename in TASK_FILES.items():
            payload = json.loads((run_dir / filename).read_text(encoding="utf-8"))
            test = payload["mean_results"]
            row[f"val_{task}_hit10"] = validation[task]["hit@10"]
            row[f"val_{task}_ndcg10"] = validation[task]["ndcg@10"]
            row[f"test_{task}_hit10"] = test["hit@10"]
            row[f"test_{task}_ndcg10"] = test["ndcg@10"]
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["policy", "budget"])
    fixed = frame[frame["policy"] == "Fixed"]
    if len(fixed) != 1:
        raise ValueError("Exactly one Fixed arm is required")
    fixed_row = fixed.iloc[0]
    metrics = [
        "test_rec_hit10",
        "test_rec_ndcg10",
        "test_search_hit10",
        "test_search_ndcg10",
    ]
    for metric in metrics:
        frame[f"{metric}_delta_fixed"] = frame[metric] - fixed_row[metric]

    pair_rows = []
    for budget in sorted(frame.loc[frame["policy"] != "Fixed", "budget"].unique()):
        current = frame[frame["budget"] == budget].set_index("policy")
        if not {"V3", "Support"}.issubset(current.index):
            raise ValueError(f"Missing V3/Support pair for budget {budget}")
        row = {"budget": int(budget)}
        for metric in metrics:
            row[f"{metric}_v3_minus_support"] = current.loc["V3", metric] - current.loc["Support", metric]
        pair_rows.append(row)
    pairs = pd.DataFrame(pair_rows).sort_values("budget")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "runs.csv", index=False)
    pairs.to_csv(args.output_dir / "paired_differences.csv", index=False)
    payload = {
        "runs": frame.to_dict("records"),
        "paired_differences": pairs.to_dict("records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False))
    print("\n" + pairs.to_string(index=False))


if __name__ == "__main__":
    main()
