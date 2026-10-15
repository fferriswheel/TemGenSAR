#!/usr/bin/env python3
"""Summarize the W6 Fixed/Support/V3 comparison over downstream seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_controlled_rec_w6 import (
    TASK_FILES,
    injected_event_sources,
    instance_metrics,
    read_validation,
    recoded_items,
)


METHODS = ("Fixed", "Support", "V3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarterly-dir", type=Path, required=True)
    parser.add_argument("--controlled-data-dir", type=Path, required=True)
    parser.add_argument("--fixed-mapping", type=Path, required=True)
    parser.add_argument("--candidate-mapping", type=Path, required=True)
    parser.add_argument("--candidate-stats", type=Path, required=True)
    parser.add_argument("--drift-csv", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="METHOD:SEED=RUN_DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summarize(values: pd.Series) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    run_specs = {}
    for spec in args.run:
        identity, raw_path = spec.split("=", 1)
        method, raw_seed = identity.split(":", 1)
        run_specs[(method, int(raw_seed))] = Path(raw_path)
    seeds = sorted({seed for _, seed in run_specs})
    expected = {(method, seed) for method in METHODS for seed in seeds}
    if set(run_specs) != expected or len(seeds) < 2:
        raise ValueError(f"Expected a complete METHODS x SEEDS grid; got {set(run_specs)}")

    fixed = json.loads(args.fixed_mapping.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_mapping.read_text(encoding="utf-8"))
    candidate_selected = set(
        map(
            str,
            json.loads(args.candidate_stats.read_text(encoding="utf-8"))["selected_item_ids"],
        )
    )
    candidate_recoded = recoded_items(fixed, candidate)

    drift = pd.read_csv(args.drift_csv, dtype={"item_id": str})
    drift = drift[(drift["left"] == "W5") & (drift["right"] == "W6")]
    eligible = set(drift.loc[drift["eligible"], "item_id"])

    id_maps = json.loads((args.quarterly_dir / "id_maps.json").read_text(encoding="utf-8"))
    int_to_asin = {int(key): str(value) for key, value in id_maps["int_to_asin"].items()}
    rec_eval = pd.read_pickle(
        args.quarterly_dir / "windows" / "W7" / "rec_eval.pkl"
    ).reset_index(drop=True)
    target_asin = rec_eval["item_id"].map(int_to_asin).to_numpy()
    injected_sources = injected_event_sources(args.controlled_data_dir, args.quarterly_dir)
    event_keys = [
        (int(row.user_id), int(row.item_id), int(row.ts)) for row in rec_eval.itertuples()
    ]
    injected = np.asarray([key in injected_sources for key in event_keys])
    original_asin = np.asarray([injected_sources.get(key, "") for key in event_keys])
    source_selected = injected & np.isin(original_asin, list(candidate_selected))
    source_recoded = injected & np.isin(original_asin, list(candidate_recoded))
    masks = {
        "all": np.ones(len(rec_eval), dtype=bool),
        "injected_target": injected,
        "non_injected_target": ~injected,
        "drift_eligible_target": np.isin(target_asin, list(eligible)),
        "v3_selected_target": np.isin(target_asin, list(candidate_selected)),
        "v3_recoded_target": np.isin(target_asin, list(candidate_recoded)),
        "injected_v3_either_selected": injected
        & (np.isin(target_asin, list(candidate_selected)) | source_selected),
        "injected_v3_either_recoded": injected
        & (np.isin(target_asin, list(candidate_recoded)) | source_recoded),
    }

    run_rows = []
    subgroup_rows = []
    for (method, seed), run_dir in sorted(run_specs.items()):
        validation = read_validation(run_dir)
        task_instances = {}
        row = {"method": method, "seed": seed, "run_dir": str(run_dir)}
        for task, filename in TASK_FILES.items():
            path = run_dir / filename
            payload = json.loads(path.read_text(encoding="utf-8"))
            test = payload["mean_results"]
            task_instances[task] = instance_metrics(path)
            row[f"val_{task}_hit10"] = validation[task]["hit@10"]
            row[f"val_{task}_ndcg10"] = validation[task]["ndcg@10"]
            row[f"test_{task}_hit10"] = test["hit@10"]
            row[f"test_{task}_ndcg10"] = test["ndcg@10"]
        run_rows.append(row)
        rec_hit, rec_ndcg = task_instances["rec"]
        for group, mask in masks.items():
            if mask.any():
                subgroup_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "group": group,
                        "samples": int(mask.sum()),
                        "hit10": float(rec_hit[mask].mean()),
                        "ndcg10": float(rec_ndcg[mask].mean()),
                    }
                )

    runs = pd.DataFrame(run_rows).sort_values(["method", "seed"])
    subgroups = pd.DataFrame(subgroup_rows).sort_values(["group", "method", "seed"])
    metric_columns = [
        "test_rec_hit10",
        "test_rec_ndcg10",
        "test_search_hit10",
        "test_search_ndcg10",
    ]
    paired_rows = []
    pivot = runs.pivot(index="seed", columns="method", values=metric_columns)
    for baseline in ("Fixed", "Support"):
        for metric in metric_columns:
            difference = pivot[metric]["V3"] - pivot[metric][baseline]
            paired_rows.append(
                {
                    "comparison": f"V3_minus_{baseline}",
                    "metric": metric,
                    "mean_difference": float(difference.mean()),
                    "positive_seeds": int((difference > 0).sum()),
                    "zero_seeds": int((difference == 0).sum()),
                    "negative_seeds": int((difference < 0).sum()),
                    **{f"seed_{seed}": float(value) for seed, value in difference.items()},
                }
            )
    paired = pd.DataFrame(paired_rows)

    subgroup_paired_rows = []
    subgroup_pivot = subgroups.pivot(index="seed", columns=["group", "method"], values="ndcg10")
    for group in masks:
        if group not in subgroup_pivot.columns.get_level_values("group"):
            continue
        for baseline in ("Fixed", "Support"):
            difference = subgroup_pivot[(group, "V3")] - subgroup_pivot[(group, baseline)]
            subgroup_paired_rows.append(
                {
                    "group": group,
                    "comparison": f"V3_minus_{baseline}",
                    "mean_difference": float(difference.mean()),
                    "positive_seeds": int((difference > 0).sum()),
                    **{f"seed_{seed}": float(value) for seed, value in difference.items()},
                }
            )
    subgroup_paired = pd.DataFrame(subgroup_paired_rows)

    summary = {
        method: {
            metric: summarize(group[metric]) for metric in metric_columns
        }
        for method, group in runs.groupby("method")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "runs.csv", index=False)
    paired.to_csv(args.output_dir / "paired_differences.csv", index=False)
    subgroups.to_csv(args.output_dir / "subgroup_runs.csv", index=False)
    subgroup_paired.to_csv(args.output_dir / "subgroup_paired_differences.csv", index=False)
    payload = {
        "seeds": seeds,
        "summary": summary,
        "paired_differences": paired.to_dict("records"),
        "subgroup_paired_differences": subgroup_paired.to_dict("records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(runs.to_string(index=False))
    print("\n" + paired.to_string(index=False))
    print("\n" + subgroup_paired.to_string(index=False))


if __name__ == "__main__":
    main()
