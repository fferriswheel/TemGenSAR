#!/usr/bin/env python3
"""Summarize the controlled W3 Rec-drift SID and downstream experiment."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


TASK_FILES = {
    "rec": "SeqRecWithSrcHis_sample_neg-999_beam-10.json",
    "search": "PerSrcWithRecHis_sample_neg-999_beam-10.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarterly-dir", type=Path, required=True)
    parser.add_argument("--controlled-data-dir", type=Path, required=True)
    parser.add_argument("--fixed-mapping", type=Path, required=True)
    parser.add_argument("--random-mapping", type=Path, required=True)
    parser.add_argument("--selective-mapping", type=Path, required=True)
    parser.add_argument("--selective-stats", type=Path, required=True)
    parser.add_argument("--random-stats", type=Path, required=True)
    parser.add_argument("--drift-csv", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="METHOD=RUN_DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def instance_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("all_prompt_instance_results")
    if prompts is None or len(prompts) != 1:
        raise ValueError(f"Missing single-prompt instance results in {path}")
    hits = np.asarray(prompts[0], dtype=np.float64)
    hit10 = (hits[:, :10].sum(axis=1) > 0).astype(np.float64)
    ndcg10 = (hits[:, :10] / np.log2(np.arange(2, 12))[None, :]).sum(axis=1)
    return hit10, ndcg10


def read_validation(run_dir: Path) -> dict[str, dict[str, float]]:
    text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    values = [
        ast.literal_eval(value)
        for value in re.findall(r"Mean results:\s+(\{[^\n]+\})", text.replace("\r", "\n"))
    ]
    if len(values) != 4:
        raise ValueError(f"Expected four result blocks in {run_dir}; got {len(values)}")
    return {"rec": values[0], "search": values[1]}


def recoded_items(base: dict, current: dict) -> set[str]:
    return {
        item for item in base
        if base[item]["rec_suffix_codes"] != current[item]["rec_suffix_codes"]
    }


def injected_event_sources(
    controlled_dir: Path, quarterly_dir: Path
) -> dict[tuple[int, int, int], str]:
    maps = json.loads((quarterly_dir / "id_maps.json").read_text(encoding="utf-8"))
    user_to_int = maps["user_to_int"]
    asin_to_int = maps["asin_to_int"]
    sources = {}
    with (controlled_dir / "rec_interactions_chrono.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("controlled_rec_drift"):
                key = (
                    int(user_to_int[str(row["user_id"])]),
                    int(asin_to_int[str(row["item_id"])]),
                    int(row["timestamp"]),
                )
                sources[key] = str(row["controlled_rec_original_item"])
    return sources


def main() -> None:
    args = parse_args()
    runs = {}
    for spec in args.run:
        method, raw_path = spec.split("=", 1)
        runs[method] = Path(raw_path)
    expected = {"Fixed", "Random", "Selective"}
    if set(runs) != expected:
        raise ValueError(f"Expected runs {expected}; got {set(runs)}")

    base = json.loads(args.fixed_mapping.read_text(encoding="utf-8"))
    random_mapping = json.loads(args.random_mapping.read_text(encoding="utf-8"))
    selective_mapping = json.loads(args.selective_mapping.read_text(encoding="utf-8"))
    random_changed = recoded_items(base, random_mapping)
    selective_changed = recoded_items(base, selective_mapping)
    random_stats = json.loads(args.random_stats.read_text(encoding="utf-8"))
    selective_stats = json.loads(args.selective_stats.read_text(encoding="utf-8"))
    random_selected = set(random_stats["selected_item_ids"])
    selective_selected = set(selective_stats["selected_item_ids"])

    drift = pd.read_csv(args.drift_csv, dtype={"item_id": str})
    drift = drift[(drift["left"] == "W2") & (drift["right"] == "W3")]
    eligible = set(drift.loc[drift["eligible"], "item_id"])

    overall_rows = []
    per_method = {}
    for method, run_dir in runs.items():
        validation = read_validation(run_dir)
        per_method[method] = {}
        for task, filename in TASK_FILES.items():
            payload = json.loads((run_dir / filename).read_text(encoding="utf-8"))
            test = payload["mean_results"]
            per_method[method][task] = instance_metrics(run_dir / filename)
            overall_rows.append({
                "method": method,
                "task": task,
                "val_hit10": validation[task]["hit@10"],
                "val_ndcg10": validation[task]["ndcg@10"],
                "test_hit10": test["hit@10"],
                "test_ndcg10": test["ndcg@10"],
            })
    overall = pd.DataFrame(overall_rows).sort_values(["task", "method"])

    id_maps = json.loads((args.quarterly_dir / "id_maps.json").read_text(encoding="utf-8"))
    int_to_asin = {int(key): str(value) for key, value in id_maps["int_to_asin"].items()}
    rec_eval = pd.read_pickle(
        args.quarterly_dir / "windows" / "W4" / "rec_eval.pkl"
    ).reset_index(drop=True)
    target_asin = rec_eval["item_id"].map(int_to_asin).to_numpy()
    injected_sources = injected_event_sources(args.controlled_data_dir, args.quarterly_dir)
    event_keys = [
        (int(row.user_id), int(row.item_id), int(row.ts))
        for row in rec_eval.itertuples()
    ]
    injected = np.asarray([key in injected_sources for key in event_keys])
    original_asin = np.asarray([injected_sources.get(key, "") for key in event_keys])
    selective_source_selected = injected & np.isin(original_asin, list(selective_selected))
    selective_source_recoded = injected & np.isin(original_asin, list(selective_changed))
    masks = {
        "all": np.ones(len(rec_eval), dtype=bool),
        "injected_target": injected,
        "non_injected_target": ~injected,
        "drift_eligible_target": np.isin(target_asin, list(eligible)),
        "selective_selected_target": np.isin(target_asin, list(selective_selected)),
        "selective_recoded_target": np.isin(target_asin, list(selective_changed)),
        "injected_selective_source_selected": selective_source_selected,
        "injected_selective_source_recoded": selective_source_recoded,
        "injected_selective_either_selected": injected & (
            np.isin(target_asin, list(selective_selected)) | selective_source_selected
        ),
        "injected_selective_either_recoded": injected & (
            np.isin(target_asin, list(selective_changed)) | selective_source_recoded
        ),
        "random_selected_target": np.isin(target_asin, list(random_selected)),
        "random_recoded_target": np.isin(target_asin, list(random_changed)),
    }
    subgroup_rows = []
    for group, mask in masks.items():
        if not mask.any():
            continue
        row = {"group": group, "samples": int(mask.sum()), "distinct_items": int(len(set(target_asin[mask])))}
        for method in ("Fixed", "Random", "Selective"):
            hit10, ndcg10 = per_method[method]["rec"]
            row[f"{method}_hit10"] = float(hit10[mask].mean())
            row[f"{method}_ndcg10"] = float(ndcg10[mask].mean())
        row["Selective_minus_Fixed_ndcg10"] = row["Selective_ndcg10"] - row["Fixed_ndcg10"]
        row["Selective_minus_Random_ndcg10"] = row["Selective_ndcg10"] - row["Random_ndcg10"]
        subgroup_rows.append(row)
    subgroup = pd.DataFrame(subgroup_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.output_dir / "overall_metrics.csv", index=False)
    subgroup.to_csv(args.output_dir / "rec_subgroup_metrics.csv", index=False)
    summary = {
        "protocol": {
            "update_window": "W3",
            "validation_window": "W3",
            "test_window": "W4",
            "seed": 2024,
            "same_downstream_data": True,
            "same_initial_checkpoint": True,
            "same_alignment_pool": True,
        },
        "sid_update": {
            "selective_selected": len(selective_selected),
            "selective_recoded": len(selective_changed),
            "random_selected": len(random_selected),
            "random_recoded": len(random_changed),
        },
        "overall": overall.to_dict("records"),
        "rec_subgroups": subgroup.to_dict("records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overall.to_string(index=False))
    print("\n" + subgroup.to_string(index=False))


if __name__ == "__main__":
    main()
