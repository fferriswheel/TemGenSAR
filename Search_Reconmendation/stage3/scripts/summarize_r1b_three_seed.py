#!/usr/bin/env python3
"""Summarize matched temporal R1b runs over multiple downstream seeds."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="METHOD:SEED=run_dir")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_run(method: str, seed: int, run_dir: Path) -> dict[str, object]:
    text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    metrics = [
        ast.literal_eval(raw)
        for raw in re.findall(r"Mean results:\s+(\{[^\n]+\})", text)
    ]
    if len(metrics) != 4:
        raise ValueError(f"Expected four results in {run_dir}, got {len(metrics)}")
    val_rec, val_search, test_rec, test_search = metrics
    return {
        "method": method,
        "seed": seed,
        "val_mean_ndcg5": (val_rec["ndcg@5"] + val_search["ndcg@5"]) / 2,
        "rec_hit10": test_rec["hit@10"],
        "rec_ndcg10": test_rec["ndcg@10"],
        "search_hit10": test_search["hit@10"],
        "search_ndcg10": test_search["ndcg@10"],
        "run_dir": str(run_dir),
    }


def main() -> None:
    args = parse_args()
    rows = []
    for spec in args.run:
        identity, raw_dir = spec.split("=", 1)
        method, raw_seed = identity.split(":", 1)
        rows.append(parse_run(method, int(raw_seed), Path(raw_dir)))
    frame = pd.DataFrame(rows).sort_values(["method", "seed"])
    metric_columns = [
        "val_mean_ndcg5", "rec_hit10", "rec_ndcg10",
        "search_hit10", "search_ndcg10",
    ]
    summary = frame.groupby("method")[metric_columns].agg(["mean", "std"])
    pivot = frame.pivot(index="seed", columns="method", values=metric_columns)
    paired_rows = []
    for baseline in ["fixed", "random"]:
        for metric in metric_columns[1:]:
            difference = pivot[metric]["drift"] - pivot[metric][baseline]
            paired_rows.append({
                "comparison": f"drift_minus_{baseline}",
                "metric": metric,
                "mean_difference": difference.mean(),
                "positive_seeds": int((difference > 0).sum()),
                "zero_seeds": int((difference == 0).sum()),
                "negative_seeds": int((difference < 0).sum()),
                **{f"seed_{seed}": value for seed, value in difference.items()},
            })
    paired = pd.DataFrame(paired_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "three_seed_runs.csv", index=False)
    summary.to_csv(args.output_dir / "three_seed_summary.csv")
    paired.to_csv(args.output_dir / "three_seed_paired_differences.csv", index=False)
    payload = {
        "runs": frame.to_dict("records"),
        "summary": {
            method: {
                metric: {
                    "mean": float(group[metric].mean()),
                    "std": float(group[metric].std()),
                }
                for metric in metric_columns
            }
            for method, group in frame.groupby("method")
        },
        "paired_differences": paired.to_dict("records"),
    }
    (args.output_dir / "three_seed_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False))
    print("\n", summary.to_string())
    print("\n", paired.to_string(index=False))


if __name__ == "__main__":
    main()
