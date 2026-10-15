#!/usr/bin/env python3
"""Evaluate temporal Search methods on drift-related example subgroups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--result", action="append", required=True, help="NAME=path/to/result.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("all_prompt_instance_results")
    if prompts is None or len(prompts) != 1:
        raise ValueError(f"Expected one prompt with instance results in {path}")
    hits = np.asarray(prompts[0], dtype=np.float64)
    hit10 = (hits[:, :10].sum(axis=1) > 0).astype(np.float64)
    ndcg10 = (hits[:, :10] / np.log2(np.arange(2, 12))[None, :]).sum(axis=1)
    return hit10, ndcg10


def bootstrap(diff: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float, float]:
    if len(diff) == 0:
        return np.nan, np.nan, np.nan
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 500):
        end = min(samples, start + 500)
        indices = rng.integers(0, len(diff), size=(end - start, len(diff)))
        means[start:end] = diff[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    p_value = min(1.0, 2 * min(float((means <= 0).mean()), float((means >= 0).mean())))
    return float(low), float(high), p_value


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.annotations).sort_values("row_index").reset_index(drop=True)
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for spec in args.result:
        name, raw_path = spec.split("=", 1)
        results[name] = load_metrics(Path(raw_path))
        if len(results[name][0]) != len(frame):
            raise ValueError(f"{name} result length does not match annotations")

    groups: list[tuple[str, np.ndarray]] = [("all", np.ones(len(frame), dtype=bool))]
    for column in [
        "injected", "drift_eligible", "drift_selected", "sid_changed",
        "selected_and_injected", "changed_and_injected",
    ]:
        groups.append((column, frame[column].astype(bool).to_numpy()))
    for value in ["changed", "selected_unchanged", "not_selected"]:
        groups.append((f"update_group:{value}", frame["update_group"].to_numpy() == value))

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for group, mask in groups:
        if not mask.any():
            continue
        row: dict[str, object] = {"group": group, "samples": int(mask.sum())}
        for name, metrics in results.items():
            row[f"{name}_hit10"] = float(metrics[0][mask].mean())
            row[f"{name}_ndcg10"] = float(metrics[1][mask].mean())
        for comparator in ["Fixed", "Random"]:
            if "Selective" not in results or comparator not in results:
                continue
            for metric_index, metric in enumerate(["hit10", "ndcg10"]):
                diff = results["Selective"][metric_index][mask] - results[comparator][metric_index][mask]
                low, high, p_value = bootstrap(diff, args.bootstrap_samples, rng)
                prefix = f"Selective_minus_{comparator}_{metric}"
                row[prefix] = float(diff.mean())
                row[f"{prefix}_ci_low"] = low
                row[f"{prefix}_ci_high"] = high
                row[f"{prefix}_p"] = p_value
        rows.append(row)

    output = pd.DataFrame(rows)
    output.to_csv(args.output_dir / "subgroup_metrics.csv", index=False)
    (args.output_dir / "subgroup_metrics.json").write_text(
        json.dumps(output.to_dict("records"), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
