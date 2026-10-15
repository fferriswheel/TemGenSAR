#!/usr/bin/env python3
"""Aggregate strict temporal SID refresh results across matched training seeds."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from evaluate_sid_refresh_strict import changed_items, load_metrics, paired_bootstrap


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    return name, Path(raw_path)


def parse_named_template(value: str) -> tuple[str, str]:
    name, separator, template = value.partition("=")
    if not separator or "{seed}" not in template:
        raise argparse.ArgumentTypeError("Expected NAME=PATH_WITH_{seed}")
    return name, template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--rec-test", type=Path, required=True)
    parser.add_argument("--id-maps", type=Path, required=True)
    parser.add_argument("--base-mapping", type=Path, required=True)
    parser.add_argument("--mapping", type=parse_named_path, action="append", required=True)
    parser.add_argument("--detector-items", type=Path, required=True)
    parser.add_argument("--rec-result", type=parse_named_template, action="append", required=True)
    parser.add_argument("--search-result", type=parse_named_template, action="append", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summarize_models(
    task: str,
    groups: dict[str, np.ndarray],
    results: dict[str, list[tuple[np.ndarray, np.ndarray]]],
) -> list[dict[str, object]]:
    rows = []
    for group_name, mask in groups.items():
        for model, seed_results in results.items():
            for metric_index, metric in enumerate(("hit10", "ndcg10")):
                values = np.asarray([result[metric_index][mask].mean() for result in seed_results])
                rows.append(
                    {
                        "task": task,
                        "group": group_name,
                        "samples": int(mask.sum()),
                        "model": model,
                        "metric": metric,
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)),
                        "seed_values": json.dumps(values.tolist()),
                    }
                )
    return rows


def summarize_comparisons(
    task: str,
    groups: dict[str, np.ndarray],
    results: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    rows = []
    for group_name, mask in groups.items():
        for left, right in itertools.combinations(results, 2):
            for metric_index, metric in enumerate(("hit10", "ndcg10")):
                per_seed_instances = np.stack(
                    [
                        results[right][seed][metric_index][mask]
                        - results[left][seed][metric_index][mask]
                        for seed in range(len(results[left]))
                    ]
                )
                seed_deltas = per_seed_instances.mean(axis=1)
                averaged_instance_delta = per_seed_instances.mean(axis=0)
                bootstrap = paired_bootstrap(averaged_instance_delta, bootstrap_samples, rng)
                if np.allclose(seed_deltas, 0):
                    t_statistic, t_p_value = 0.0, 1.0
                elif float(np.ptp(seed_deltas)) <= 1e-15:
                    # A zero-variance nonzero difference makes the t statistic undefined.
                    t_statistic, t_p_value = None, None
                else:
                    test = stats.ttest_1samp(seed_deltas, 0.0)
                    t_statistic, t_p_value = float(test.statistic), float(test.pvalue)
                rows.append(
                    {
                        "task": task,
                        "group": group_name,
                        "samples": int(mask.sum()),
                        "comparison": f"{right}_minus_{left}",
                        "metric": metric,
                        "mean_delta": float(seed_deltas.mean()),
                        "std_delta": float(seed_deltas.std(ddof=1)),
                        "positive_seeds": int((seed_deltas > 0).sum()),
                        "zero_seeds": int((seed_deltas == 0).sum()),
                        "seed_deltas": json.dumps(seed_deltas.tolist()),
                        "seed_paired_t": t_statistic,
                        "seed_paired_t_p": t_p_value,
                        "row_bootstrap_ci_low": bootstrap["ci_low"],
                        "row_bootstrap_ci_high": bootstrap["ci_high"],
                        "row_bootstrap_p": bootstrap["p_value"],
                    }
                )
    return rows


def load_seed_results(
    specifications: list[tuple[str, str]], seeds: list[int]
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    return {
        name: [load_metrics(Path(template.format(seed=seed))) for seed in seeds]
        for name, template in specifications
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)

    rec_frame = pd.read_pickle(args.rec_test).reset_index(drop=True)
    id_maps = json.loads(args.id_maps.read_text(encoding="utf-8"))
    targets = rec_frame["item_id"].map(lambda value: str(id_maps["int_to_item"][str(value)]))
    base = json.loads(args.base_mapping.read_text(encoding="utf-8"))
    changed = {name: changed_items(base, path) for name, path in args.mapping}
    detector = set(json.loads(args.detector_items.read_text(encoding="utf-8")))
    rec_groups = {
        "all": np.ones(len(rec_frame), dtype=bool),
        "detector_k150": targets.isin(detector).to_numpy(),
        "changed_union": targets.isin(set().union(*changed.values())).to_numpy(),
    }
    rec_groups.update(
        {f"{name}_changed": targets.isin(items).to_numpy() for name, items in changed.items()}
    )

    rec_results = load_seed_results(args.rec_result, args.seed)
    search_results = load_seed_results(args.search_result, args.seed)
    search_size = len(next(iter(search_results.values()))[0][0])
    search_groups = {"all": np.ones(search_size, dtype=bool)}

    model_rows = summarize_models("rec", rec_groups, rec_results)
    model_rows += summarize_models("search", search_groups, search_results)
    comparison_rows = summarize_comparisons(
        "rec", rec_groups, rec_results, args.bootstrap_samples, rng
    )
    comparison_rows += summarize_comparisons(
        "search", search_groups, search_results, args.bootstrap_samples, rng
    )

    models = pd.DataFrame(model_rows)
    comparisons = pd.DataFrame(comparison_rows)
    models.to_csv(args.output_dir / "model_metrics.csv", index=False)
    comparisons.to_csv(args.output_dir / "comparisons.csv", index=False)
    payload = {
        "seeds": args.seed,
        "changed_items": {name: len(items) for name, items in changed.items()},
        "bootstrap_samples": args.bootstrap_samples,
        "model_metrics": model_rows,
        "comparisons": comparison_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(models.to_string(index=False))
    print("\nComparisons:\n")
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
