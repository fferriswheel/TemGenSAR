#!/usr/bin/env python3
"""Paired evaluation for fixed and selectively refreshed temporal SID runs."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    return name, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rec-test", type=Path, required=True)
    parser.add_argument("--id-maps", type=Path, required=True)
    parser.add_argument("--base-mapping", type=Path, required=True)
    parser.add_argument("--mapping", type=parse_named_path, action="append", required=True)
    parser.add_argument("--result", type=parse_named_path, action="append", required=True)
    parser.add_argument("--detector-items", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload["all_prompt_instance_results"]
    if len(prompts) != 1:
        raise ValueError(f"Expected one prompt in {path}")
    ranks = np.asarray(prompts[0], dtype=np.float64)
    if ranks.ndim != 2 or ranks.shape[1] < 10:
        raise ValueError(f"Expected [N, >=10] hit matrix in {path}, got {ranks.shape}")
    hit10 = (ranks[:, :10].sum(axis=1) > 0).astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, 12))
    ndcg10 = (ranks[:, :10] * discounts[None, :]).sum(axis=1)
    return hit10, ndcg10


def changed_items(base: dict[str, dict], current_path: Path) -> set[str]:
    current = json.loads(current_path.read_text(encoding="utf-8"))
    if set(current) != set(base):
        raise ValueError(f"Catalog mismatch in {current_path}")
    return {
        item_id
        for item_id in base
        if current[item_id]["rec_suffix_codes"] != base[item_id]["rec_suffix_codes"]
    }


def paired_bootstrap(
    difference: np.ndarray, samples: int, rng: np.random.Generator
) -> dict[str, float]:
    if len(difference) == 0:
        return {"ci_low": math.nan, "ci_high": math.nan, "p_value": math.nan}
    means = np.empty(samples, dtype=np.float64)
    chunk_size = 500
    for start in range(0, samples, chunk_size):
        end = min(start + chunk_size, samples)
        indices = rng.integers(0, len(difference), size=(end - start, len(difference)))
        means[start:end] = difference[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(means, [0.025, 0.975])
    p_value = min(
        1.0,
        2.0 * min(float((means <= 0).mean()), float((means >= 0).mean())),
    )
    return {
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_pickle(args.rec_test).reset_index(drop=True)
    id_maps = json.loads(args.id_maps.read_text(encoding="utf-8"))
    targets = frame["item_id"].map(lambda value: str(id_maps["int_to_item"][str(value)]))

    base = json.loads(args.base_mapping.read_text(encoding="utf-8"))
    changed = {name: changed_items(base, path) for name, path in args.mapping}
    detector = set(json.loads(args.detector_items.read_text(encoding="utf-8")))
    changed_union = set().union(*changed.values())
    groups = {
        "all": np.ones(len(frame), dtype=bool),
        "detector_k150": targets.isin(detector).to_numpy(),
        "changed_union": targets.isin(changed_union).to_numpy(),
    }
    groups.update(
        {f"{name}_changed": targets.isin(items).to_numpy() for name, items in changed.items()}
    )

    results = {name: load_metrics(path) for name, path in args.result}
    for name, (hit10, _) in results.items():
        if len(hit10) != len(frame):
            raise ValueError(f"{name} has {len(hit10)} rows; test data has {len(frame)}")

    rng = np.random.default_rng(args.seed)
    metric_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for group_name, mask in groups.items():
        for model_name, (hit10, ndcg10) in results.items():
            metric_rows.append(
                {
                    "group": group_name,
                    "samples": int(mask.sum()),
                    "model": model_name,
                    "hit10": float(hit10[mask].mean()),
                    "ndcg10": float(ndcg10[mask].mean()),
                }
            )
        for left, right in itertools.combinations(results, 2):
            for metric_index, metric_name in enumerate(("hit10", "ndcg10")):
                difference = results[right][metric_index][mask] - results[left][metric_index][mask]
                comparison_rows.append(
                    {
                        "group": group_name,
                        "samples": int(mask.sum()),
                        "comparison": f"{right}_minus_{left}",
                        "metric": metric_name,
                        "delta": float(difference.mean()),
                        **paired_bootstrap(difference, args.bootstrap_samples, rng),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    comparisons = pd.DataFrame(comparison_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    comparisons.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    summary = {
        "rec_test_rows": len(frame),
        "changed_items": {name: len(items) for name, items in changed.items()},
        "detector_items": len(detector),
        "bootstrap_samples": args.bootstrap_samples,
        "metrics": metric_rows,
        "comparisons": comparison_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print("\nPaired comparisons:\n")
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
