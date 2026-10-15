#!/usr/bin/env python3
"""Summarize validation/test metrics from temporal alignment-budget runs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="BUDGET=run_directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def last_match(pattern: str, text: str) -> str:
    matches = re.findall(pattern, text)
    if not matches:
        raise ValueError(f"Pattern not found: {pattern}")
    return matches[-1]


def parse_run(budget: int, run_dir: Path) -> dict[str, object]:
    text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    metrics = [
        ast.literal_eval(raw)
        for raw in re.findall(r"Mean results:\s+(\{[^\n]+\})", text)
    ]
    if len(metrics) != 4:
        raise ValueError(f"Expected four aggregate results in {run_dir}, got {len(metrics)}")
    val_rec, val_search, test_rec, test_search = metrics
    row: dict[str, object] = {
        "configured_budget_per_task": budget,
        "actual_examples_per_alignment_task": int(
            last_match(r"task: Item2Index data_num: (\d+)", text)
        ),
        "total_train_examples": int(
            last_match(r"Num examples = ([\d,]+)", text).replace(",", "")
        ),
        "optimizer_steps": int(last_match(r"Total optimization steps = (\d+)", text)),
        "wall_time": last_match(
            r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([^\n]+)", text
        ),
        "run_dir": str(run_dir),
    }
    for split, result in [
        ("val_rec", val_rec),
        ("val_search", val_search),
        ("test_rec", test_rec),
        ("test_search", test_search),
    ]:
        row[f"{split}_hit10"] = result["hit@10"]
        row[f"{split}_ndcg5"] = result["ndcg@5"]
        row[f"{split}_ndcg10"] = result["ndcg@10"]
    row["val_mean_ndcg5"] = (val_rec["ndcg@5"] + val_search["ndcg@5"]) / 2
    row["val_mean_ndcg10"] = (val_rec["ndcg@10"] + val_search["ndcg@10"]) / 2
    return row


def main() -> None:
    args = parse_args()
    rows = []
    for spec in args.run:
        raw_budget, raw_dir = spec.split("=", 1)
        rows.append(parse_run(int(raw_budget), Path(raw_dir)))
    frame = pd.DataFrame(rows).sort_values("configured_budget_per_task")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "alignment_budget_curve.csv", index=False)
    payload = {
        "selection_metric": "mean W1 validation Rec/Search nDCG@5",
        "rows": frame.to_dict("records"),
    }
    (args.output_dir / "alignment_budget_curve.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
