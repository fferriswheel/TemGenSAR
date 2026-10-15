#!/usr/bin/env python3
"""Summarize joint Search/Rec temporal experiments over downstream seeds."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd


METRICS = ["rec_hit10", "rec_ndcg10", "search_hit10", "search_ndcg10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="METHOD:SEED=run_dir")
    parser.add_argument(
        "--reference",
        help="Optional method used as the left-hand side of paired comparisons.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_run(method: str, seed: int, run_dir: Path) -> dict[str, object]:
    text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    values = [ast.literal_eval(raw) for raw in re.findall(r"Mean results:\s+(\{[^\n]+\})", text)]
    if len(values) != 4:
        raise ValueError(f"Expected four result blocks in {run_dir}; got {len(values)}")
    val_rec, val_search, test_rec, test_search = values
    return {
        "method": method,
        "seed": seed,
        "val_rec_ndcg5": val_rec["ndcg@5"],
        "val_search_ndcg5": val_search["ndcg@5"],
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
        identity, raw_path = spec.split("=", 1)
        method, raw_seed = identity.split(":", 1)
        rows.append(parse_run(method, int(raw_seed), Path(raw_path)))
    frame = pd.DataFrame(rows).sort_values(["method", "seed"])
    seeds_by_method = frame.groupby("method")["seed"].apply(set).to_dict()
    if len({frozenset(value) for value in seeds_by_method.values()}) != 1:
        raise ValueError(f"Methods do not share the same seed set: {seeds_by_method}")

    summary = frame.groupby("method")[METRICS].agg(["mean", "std"])
    pivot = frame.pivot(index="seed", columns="method", values=METRICS)
    comparisons = []
    methods = list(frame["method"].drop_duplicates())
    if args.reference:
        if args.reference not in methods:
            raise ValueError(f"Unknown reference method: {args.reference}")
        comparison_pairs = [
            (args.reference, baseline)
            for baseline in methods
            if baseline != args.reference
        ]
    elif "JB" in methods:
        comparison_pairs = [
            ("JB", baseline)
            for baseline in ["J0", "JS", "JR", "JRandom"]
            if baseline in methods
        ]
    else:
        comparison_pairs = []
    for reference, baseline in comparison_pairs:
        for metric in METRICS:
            diff = pivot[metric][reference] - pivot[metric][baseline]
            comparisons.append({
                "comparison": f"{reference}_minus_{baseline}",
                "metric": metric,
                "mean_difference": float(diff.mean()),
                "positive_seeds": int((diff > 0).sum()),
                "zero_seeds": int((diff == 0).sum()),
                "negative_seeds": int((diff < 0).sum()),
                **{f"seed_{seed}": float(value) for seed, value in diff.items()},
            })
    paired = pd.DataFrame(comparisons)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "runs.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv")
    paired.to_csv(args.output_dir / "paired_differences.csv", index=False)
    payload = {
        "runs": frame.to_dict("records"),
        "summary": {
            method: {
                metric: {"mean": float(group[metric].mean()), "std": float(group[metric].std())}
                for metric in METRICS
            }
            for method, group in frame.groupby("method")
        },
        "paired_differences": paired.to_dict("records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False))
    print("\n" + summary.to_string())
    print("\n" + paired.to_string(index=False))


if __name__ == "__main__":
    main()
