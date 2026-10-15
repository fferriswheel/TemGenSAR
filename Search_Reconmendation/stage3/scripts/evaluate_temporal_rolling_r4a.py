#!/usr/bin/env python3
"""Aggregate per-instance R3 rolling metrics over drift and recoding subgroups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TASK_FILES = {
    "rec": "SeqRecWithSrcHis_sample_neg-999_beam-10.json",
    "search": "PerSrcWithRecHis_sample_neg-999_beam-10.json",
}
METHODS = ("Fixed", "Selective", "Random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_instance_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("all_prompt_instance_results")
    if prompts is None or len(prompts) != 1:
        raise ValueError(f"Missing single-prompt instance results in {path}")
    hits = np.asarray(prompts[0], dtype=np.float64)
    if hits.ndim != 2 or hits.shape[1] < 10:
        raise ValueError(f"Expected [N, >=10] hit vectors in {path}; got {hits.shape}")
    hit10 = (hits[:, :10].sum(axis=1) > 0).astype(np.float64)
    ndcg10 = (hits[:, :10] / np.log2(np.arange(2, 12))[None, :]).sum(axis=1)
    return hit10, ndcg10


def group_masks(frame: pd.DataFrame, task: str) -> list[tuple[str, np.ndarray]]:
    selected = frame["drift_selected"].astype(bool).to_numpy()
    changed = frame["sid_changed"].astype(bool).to_numpy()
    first = frame["first_change"].astype(bool).to_numpy()
    repeated = frame["repeat_change"].astype(bool).to_numpy()
    masks = [
        ("all", np.ones(len(frame), dtype=bool)),
        ("drift_selected", selected),
        ("sid_changed", changed),
        ("selected_unchanged", selected & ~changed),
        ("first_change", first),
        ("repeat_change", repeated),
        ("not_selected", ~selected),
        ("unchanged", ~changed),
    ]
    if task == "search":
        injected = frame["injected"].astype(bool).to_numpy()
        masks.extend([
            ("injected", injected),
            ("changed_and_injected", changed & injected),
        ])
    return [(name, mask) for name, mask in masks if mask.any()]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for window in range(1, 10):
        update_window = f"W{window}"
        eval_window = f"W{window + 1}"
        for task, filename in TASK_FILES.items():
            frame = pd.read_csv(
                args.annotations_dir / update_window / task / "example_annotations.csv"
            ).sort_values("row_index").reset_index(drop=True)
            results = {
                method: load_instance_metrics(
                    args.predictions_dir / f"{update_window}_{method}" / filename
                )
                for method in METHODS
            }
            for method, values in results.items():
                if len(values[0]) != len(frame):
                    raise ValueError(
                        f"{update_window} {task} {method}: {len(values[0])} results "
                        f"for {len(frame)} annotations"
                    )

            for group, mask in group_masks(frame, task):
                row: dict[str, object] = {
                    "update_window": update_window,
                    "eval_window": eval_window,
                    "task": task,
                    "group": group,
                    "samples": int(mask.sum()),
                    "distinct_items": int(frame.loc[mask, "item_id"].nunique()),
                }
                for method, (hit10, ndcg10) in results.items():
                    row[f"{method}_hit10"] = float(hit10[mask].mean())
                    row[f"{method}_ndcg10"] = float(ndcg10[mask].mean())
                for comparator in ("Fixed", "Random"):
                    for metric_index, metric in enumerate(("hit10", "ndcg10")):
                        delta = (
                            results["Selective"][metric_index][mask]
                            - results[comparator][metric_index][mask]
                        )
                        row[f"Selective_minus_{comparator}_{metric}"] = float(delta.mean())
                rows.append(row)

    window = pd.DataFrame(rows)
    window.to_csv(args.output_dir / "window_subgroup_metrics.csv", index=False)

    aggregate_rows: list[dict[str, object]] = []
    for (task, group), part in window.groupby(["task", "group"], sort=True):
        row = {
            "task": task,
            "group": group,
            "windows": len(part),
            "total_samples": int(part["samples"].sum()),
            "mean_samples_per_window": float(part["samples"].mean()),
        }
        for method in METHODS:
            for metric in ("hit10", "ndcg10"):
                column = f"{method}_{metric}"
                row[f"{column}_window_macro"] = float(part[column].mean())
                row[f"{column}_sample_weighted"] = float(
                    np.average(part[column], weights=part["samples"])
                )
        for comparator in ("Fixed", "Random"):
            for metric in ("hit10", "ndcg10"):
                column = f"Selective_minus_{comparator}_{metric}"
                row[f"{column}_window_macro"] = float(part[column].mean())
                row[f"{column}_sample_weighted"] = float(
                    np.average(part[column], weights=part["samples"])
                )
                row[f"{column}_positive_windows"] = int((part[column] > 0).sum())
        aggregate_rows.append(row)

    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(args.output_dir / "aggregate_subgroup_metrics.csv", index=False)
    payload = {
        "window_rows": len(window),
        "aggregate": aggregate.to_dict("records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    columns = [
        "task", "group", "windows", "total_samples",
        "Selective_minus_Fixed_ndcg10_window_macro",
        "Selective_minus_Fixed_ndcg10_positive_windows",
        "Selective_minus_Random_ndcg10_window_macro",
        "Selective_minus_Random_ndcg10_positive_windows",
    ]
    print(aggregate[columns].to_string(index=False))


if __name__ == "__main__":
    main()
