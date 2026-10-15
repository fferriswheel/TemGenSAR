#!/usr/bin/env python3
"""Compute temporal R1 subgroup metrics from per-instance beam hit vectors."""

from __future__ import annotations

import argparse
import json
import math
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


def load_instance_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("all_prompt_instance_results")
    if prompts is None or len(prompts) != 1:
        raise ValueError(f"Expected one prompt with instance results in {path}")
    hits = np.asarray(prompts[0], dtype=np.float64)
    if hits.ndim != 2 or hits.shape[1] < 10:
        raise ValueError(f"Expected [N, >=10] hit vectors in {path}, got {hits.shape}")
    hit10 = (hits[:, :10].sum(axis=1) > 0).astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, 12))
    ndcg10 = (hits[:, :10] * discounts[None, :]).sum(axis=1)
    return hit10, ndcg10


def paired_bootstrap(diff: np.ndarray, samples: int, rng: np.random.Generator) -> dict[str, float]:
    if len(diff) == 0:
        return {"ci_low": math.nan, "ci_high": math.nan, "p_value": math.nan}
    means = np.empty(samples, dtype=np.float64)
    chunk = 500
    for start in range(0, samples, chunk):
        end = min(start + chunk, samples)
        indices = rng.integers(0, len(diff), size=(end - start, len(diff)))
        means[start:end] = diff[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    p_value = min(1.0, 2 * min(float((means <= 0).mean()), float((means >= 0).mean())))
    return {"ci_low": float(lower), "ci_high": float(upper), "p_value": p_value}


def masks(frame: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    t0_seen = frame["t0_rec_events"].to_numpy() > 0
    groups = [("all", "all", np.ones(len(frame), dtype=bool))]
    for column in ["update_group", "popularity_group"]:
        for value in sorted(frame[column].dropna().unique()):
            groups.append((column, str(value), frame[column].to_numpy() == value))
    for update in sorted(frame["update_group"].unique()):
        for popularity in ["head", "middle", "tail"]:
            mask = (frame["update_group"].to_numpy() == update) & (
                frame["popularity_group"].to_numpy() == popularity
            )
            groups.append(("update_x_popularity", f"{update}:{popularity}", mask))
        groups.append((
            "update_x_t0_rec_seen",
            f"{update}:seen",
            (frame["update_group"].to_numpy() == update) & t0_seen,
        ))
        groups.append((
            "update_x_t0_rec_seen",
            f"{update}:unseen",
            (frame["update_group"].to_numpy() == update) & ~t0_seen,
        ))
    for value in sorted(frame.loc[frame["changed"], "drift_quintile_changed"].unique()):
        groups.append((
            "changed_drift_quintile",
            str(value),
            frame["drift_quintile_changed"].to_numpy() == value,
        ))
    for column in ["drift_eligible", "drift_selected"]:
        values = frame[column].astype(bool).to_numpy()
        groups.append((column, "yes", values))
        groups.append((column, "no", ~values))
        groups.append((
            f"changed_x_{column}",
            "yes",
            frame["changed"].astype(bool).to_numpy() & values,
        ))
    return groups


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.annotations).sort_values("row_index").reset_index(drop=True)
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for spec in args.result:
        name, raw_path = spec.split("=", 1)
        results[name] = load_instance_metrics(Path(raw_path))
        if len(results[name][0]) != len(frame):
            raise ValueError(f"{name} has {len(results[name][0])} rows; annotations have {len(frame)}")

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed)
    for group_type, group, mask in masks(frame):
        n = int(mask.sum())
        if n == 0:
            continue
        row: dict[str, object] = {"group_type": group_type, "group": group, "samples": n}
        for name, (hit10, ndcg10) in results.items():
            row[f"{name}_hit10"] = float(hit10[mask].mean())
            row[f"{name}_ndcg10"] = float(ndcg10[mask].mean())
        if "E2" in results and "E3" in results:
            for metric_index, metric in enumerate(["hit10", "ndcg10"]):
                diff = results["E3"][metric_index][mask] - results["E2"][metric_index][mask]
                stats = paired_bootstrap(diff, args.bootstrap_samples, rng)
                row[f"E3_minus_E2_{metric}"] = float(diff.mean())
                row[f"E3_minus_E2_{metric}_ci_low"] = stats["ci_low"]
                row[f"E3_minus_E2_{metric}_ci_high"] = stats["ci_high"]
                row[f"E3_minus_E2_{metric}_p"] = stats["p_value"]
        rows.append(row)

    output = pd.DataFrame(rows)
    output.to_csv(args.output_dir / "subgroup_metrics.csv", index=False)
    payload = {
        "models": list(results),
        "bootstrap_samples": args.bootstrap_samples,
        "rows": output.to_dict("records"),
    }
    (args.output_dir / "subgroup_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
