#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import math
import re
import statistics
from pathlib import Path


WINDOWS = (3, 5, 9)
SEEDS = (42, 2024, 2025)
METHODS = ("Fixed", "Random", "Selective", "R4C")
TASK_FILES = {
    "rec": "SeqRecWithSrcHis_sample_neg-999_beam-10.json",
    "search": "PerSrcWithRecHis_sample_neg-999_beam-10.json",
}
MEAN_RESULTS_RE = re.compile(r"^Mean results:\s+(\{.*\})$", re.MULTILINE)


def run_dir(root: Path, method: str, window: int, seed: int) -> Path:
    if seed != 2024:
        return root / f"R4S_W{window}_{method}_s{seed}"
    if method == "R4C":
        return root / f"R4C_C0_W{window}_s2024"
    return root / f"R3_W{window}_{method}_s2024"


def read_validation(path: Path) -> dict:
    text = (path / "train.log").read_text(errors="replace").replace("\r", "\n")
    matches = MEAN_RESULTS_RE.findall(text)
    if len(matches) < 2:
        raise RuntimeError(f"validation metrics missing: {path}")
    return {"rec": ast.literal_eval(matches[0]), "search": ast.literal_eval(matches[1])}


def read_test(path: Path) -> dict:
    result = {}
    for task, filename in TASK_FILES.items():
        with (path / filename).open() as handle:
            result[task] = json.load(handle)["mean_results"]
    return result


def mean_ci95(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, mean, mean
    # Student-t critical values for the small seed-level samples used here.
    critical = {2: 12.706, 3: 4.303}.get(len(values), 1.96)
    radius = critical * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - radius, mean + radius


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("stage3/runs/temporal_incremental"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("stage3/results/temporal_r4_three_seed"),
    )
    parser.add_argument("--gate-margin", type=float, default=0.0002)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    lookup = {}
    for seed in SEEDS:
        for window in WINDOWS:
            for method in METHODS:
                path = run_dir(args.runs_dir, method, window, seed)
                validation = read_validation(path)
                test = read_test(path)
                record = {
                    "seed": seed,
                    "update_window": f"W{window}",
                    "eval_window": f"W{window + 1}",
                    "method": method,
                }
                for split, metrics in (("val", validation), ("test", test)):
                    for task in TASK_FILES:
                        record[f"{split}_{task}_hit10"] = metrics[task]["hit@10"]
                        record[f"{split}_{task}_ndcg10"] = metrics[task]["ndcg@10"]
                    record[f"{split}_mean_ndcg10"] = statistics.mean(
                        metrics[task]["ndcg@10"] for task in TASK_FILES
                    )
                records.append(record)
                lookup[(seed, window, method)] = record

    with (args.output_dir / "all_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    method_summary = []
    for method in METHODS:
        for metric in ("test_rec_ndcg10", "test_search_ndcg10", "test_mean_ndcg10"):
            seed_means = [
                statistics.mean(lookup[(seed, window, method)][metric] for window in WINDOWS)
                for seed in SEEDS
            ]
            mean, low, high = mean_ci95(seed_means)
            method_summary.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": mean,
                    "seed_std": statistics.stdev(seed_means),
                    "ci95_low": low,
                    "ci95_high": high,
                    "seed_means": json.dumps(dict(zip(SEEDS, seed_means))),
                }
            )
    with (args.output_dir / "method_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(method_summary[0]))
        writer.writeheader()
        writer.writerows(method_summary)

    pairwise = []
    for candidate in ("Selective", "R4C"):
        baselines = ("Fixed", "Random", "Selective") if candidate == "R4C" else ("Fixed", "Random")
        for baseline in baselines:
            for metric in ("test_rec_ndcg10", "test_search_ndcg10", "test_mean_ndcg10"):
                deltas = []
                seed_means = []
                for seed in SEEDS:
                    current = []
                    for window in WINDOWS:
                        delta = lookup[(seed, window, candidate)][metric] - lookup[(seed, window, baseline)][metric]
                        deltas.append(delta)
                        current.append(delta)
                    seed_means.append(statistics.mean(current))
                mean, low, high = mean_ci95(seed_means)
                pairwise.append(
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": metric,
                        "mean_delta": mean,
                        "positive_cells": sum(delta > 0 for delta in deltas),
                        "total_cells": len(deltas),
                        "positive_seeds": sum(delta > 0 for delta in seed_means),
                        "ci95_low": low,
                        "ci95_high": high,
                        "seed_mean_deltas": json.dumps(dict(zip(SEEDS, seed_means))),
                    }
                )
    with (args.output_dir / "pairwise_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairwise[0]))
        writer.writeheader()
        writer.writerows(pairwise)

    gate_rows = []
    for seed in SEEDS:
        for window in WINDOWS:
            base = lookup[(seed, window, "Selective")]
            candidate = lookup[(seed, window, "R4C")]
            val_delta = candidate["val_mean_ndcg10"] - base["val_mean_ndcg10"]
            accepted = val_delta >= args.gate_margin
            chosen = candidate if accepted else base
            fixed = lookup[(seed, window, "Fixed")]
            random = lookup[(seed, window, "Random")]
            gate_rows.append(
                {
                    "seed": seed,
                    "update_window": f"W{window}",
                    "accepted_r4c": accepted,
                    "validation_delta": val_delta,
                    "test_delta_vs_selective": chosen["test_mean_ndcg10"] - base["test_mean_ndcg10"],
                    "rec_test_delta_vs_selective": chosen["test_rec_ndcg10"] - base["test_rec_ndcg10"],
                    "search_test_delta_vs_selective": chosen["test_search_ndcg10"] - base["test_search_ndcg10"],
                    "chosen_rec_test_ndcg10": chosen["test_rec_ndcg10"],
                    "chosen_search_test_ndcg10": chosen["test_search_ndcg10"],
                    "chosen_mean_test_ndcg10": chosen["test_mean_ndcg10"],
                    "mean_test_delta_vs_fixed": chosen["test_mean_ndcg10"] - fixed["test_mean_ndcg10"],
                    "rec_test_delta_vs_fixed": chosen["test_rec_ndcg10"] - fixed["test_rec_ndcg10"],
                    "mean_test_delta_vs_random": chosen["test_mean_ndcg10"] - random["test_mean_ndcg10"],
                }
            )
    with (args.output_dir / "gate_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gate_rows[0]))
        writer.writeheader()
        writer.writerows(gate_rows)

    summary = {
        "protocol": "incremental-training seeds; shared 2024 upstream trajectory",
        "seeds": list(SEEDS),
        "windows": list(WINDOWS),
        "gate_margin": args.gate_margin,
        "gate_accepts": sum(row["accepted_r4c"] for row in gate_rows),
        "gate_total": len(gate_rows),
        "gate_mean_test_delta_vs_selective": statistics.mean(
            row["test_delta_vs_selective"] for row in gate_rows
        ),
        "gate_mean_test_delta_vs_fixed": statistics.mean(
            row["mean_test_delta_vs_fixed"] for row in gate_rows
        ),
        "gate_rec_test_delta_vs_fixed": statistics.mean(
            row["rec_test_delta_vs_fixed"] for row in gate_rows
        ),
        "gate_mean_test_delta_vs_random": statistics.mean(
            row["mean_test_delta_vs_random"] for row in gate_rows
        ),
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
