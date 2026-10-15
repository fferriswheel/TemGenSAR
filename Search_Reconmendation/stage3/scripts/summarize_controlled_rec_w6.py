#!/usr/bin/env python3
"""Summarize the W6 controlled Rec-drift downstream smoke test."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


TASK_FILES = {
    "rec": "SeqRecWithSrcHis_sample_neg-999_beam-10.json",
    "search": "PerSrcWithRecHis_sample_neg-999_beam-10.json",
}
PREFERRED_METHOD_ORDER = ("Fixed", "Random", "Support", "V1", "V2", "V3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarterly-dir", type=Path, required=True)
    parser.add_argument("--controlled-data-dir", type=Path, required=True)
    parser.add_argument("--fixed-mapping", type=Path, required=True)
    parser.add_argument("--drift-csv", type=Path, required=True)
    parser.add_argument("--drift-left", default="W5")
    parser.add_argument("--drift-right", default="W6")
    parser.add_argument("--test-window", default="W7")
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="METHOD=MAPPING=UPDATE_STATS (for non-Fixed methods)",
    )
    parser.add_argument("--run", action="append", required=True, help="METHOD=RUN_DIR")
    parser.add_argument("--candidate", default="V2")
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
    if len(values) < 4:
        raise ValueError(f"Expected at least four result blocks in {run_dir}; got {len(values)}")
    return {"rec": values[0], "search": values[1]}


def recoded_items(base: dict, current: dict) -> set[str]:
    return {
        item
        for item in base
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
    runs = dict(spec.split("=", 1) for spec in args.run)
    if "Fixed" not in runs or "Support" not in runs or args.candidate not in runs:
        raise ValueError("Runs must include Fixed, Support, and the candidate method")
    methods = [method for method in PREFERRED_METHOD_ORDER if method in runs]
    methods.extend(sorted(set(runs) - set(methods)))

    arm_specs = {}
    for spec in args.arm:
        method, mapping, stats = spec.split("=", 2)
        arm_specs[method] = (Path(mapping), Path(stats))
    if set(arm_specs) != set(methods) - {"Fixed"}:
        raise ValueError(f"Missing non-Fixed arm specs: {set(arm_specs)}")

    base = json.loads(args.fixed_mapping.read_text(encoding="utf-8"))
    selected = {}
    recoded = {}
    for method, (mapping_path, stats_path) in arm_specs.items():
        current = json.loads(mapping_path.read_text(encoding="utf-8"))
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        selected[method] = set(map(str, stats["selected_item_ids"]))
        recoded[method] = recoded_items(base, current)

    drift = pd.read_csv(args.drift_csv, dtype={"item_id": str})
    drift = drift[
        (drift["left"] == args.drift_left) & (drift["right"] == args.drift_right)
    ]
    eligible = set(drift.loc[drift["eligible"], "item_id"])

    overall_rows = []
    per_method = {}
    for method in methods:
        run_dir = Path(runs[method])
        validation = read_validation(run_dir)
        per_method[method] = {}
        for task, filename in TASK_FILES.items():
            payload = json.loads((run_dir / filename).read_text(encoding="utf-8"))
            test = payload["mean_results"]
            per_method[method][task] = instance_metrics(run_dir / filename)
            overall_rows.append(
                {
                    "method": method,
                    "task": task,
                    "val_hit10": validation[task]["hit@10"],
                    "val_ndcg10": validation[task]["ndcg@10"],
                    "test_hit10": test["hit@10"],
                    "test_ndcg10": test["ndcg@10"],
                }
            )
    overall = pd.DataFrame(overall_rows).sort_values(["task", "method"])

    id_maps = json.loads((args.quarterly_dir / "id_maps.json").read_text(encoding="utf-8"))
    int_to_asin = {int(key): str(value) for key, value in id_maps["int_to_asin"].items()}
    rec_eval = pd.read_pickle(
        args.quarterly_dir / "windows" / args.test_window / "rec_eval.pkl"
    ).reset_index(drop=True)
    target_asin = rec_eval["item_id"].map(int_to_asin).to_numpy()
    injected_sources = injected_event_sources(args.controlled_data_dir, args.quarterly_dir)
    event_keys = [
        (int(row.user_id), int(row.item_id), int(row.ts)) for row in rec_eval.itertuples()
    ]
    injected = np.asarray([key in injected_sources for key in event_keys])
    original_asin = np.asarray([injected_sources.get(key, "") for key in event_keys])

    masks = {
        "all": np.ones(len(rec_eval), dtype=bool),
        "injected_target": injected,
        "non_injected_target": ~injected,
        "drift_eligible_target": np.isin(target_asin, list(eligible)),
    }
    for method in methods[1:]:
        key = method.lower()
        masks[f"{key}_selected_target"] = np.isin(target_asin, list(selected[method]))
        masks[f"{key}_recoded_target"] = np.isin(target_asin, list(recoded[method]))
    candidate_key = args.candidate.lower()
    candidate_source_selected = injected & np.isin(
        original_asin, list(selected[args.candidate])
    )
    candidate_source_recoded = injected & np.isin(
        original_asin, list(recoded[args.candidate])
    )
    masks.update(
        {
            f"injected_{candidate_key}_source_selected": candidate_source_selected,
            f"injected_{candidate_key}_source_recoded": candidate_source_recoded,
            f"injected_{candidate_key}_either_selected": injected
            & (
                np.isin(target_asin, list(selected[args.candidate]))
                | candidate_source_selected
            ),
            f"injected_{candidate_key}_either_recoded": injected
            & (
                np.isin(target_asin, list(recoded[args.candidate]))
                | candidate_source_recoded
            ),
        }
    )

    subgroup_rows = []
    for group, mask in masks.items():
        if not mask.any():
            continue
        row = {
            "group": group,
            "samples": int(mask.sum()),
            "distinct_items": int(len(set(target_asin[mask]))),
        }
        for method in methods:
            hit10, ndcg10 = per_method[method]["rec"]
            row[f"{method}_hit10"] = float(hit10[mask].mean())
            row[f"{method}_ndcg10"] = float(ndcg10[mask].mean())
        for baseline in (method for method in methods if method != args.candidate):
            row[f"{args.candidate}_minus_{baseline}_ndcg10"] = (
                row[f"{args.candidate}_ndcg10"] - row[f"{baseline}_ndcg10"]
            )
        subgroup_rows.append(row)
    subgroup = pd.DataFrame(subgroup_rows)

    metric = overall.set_index(["method", "task"])
    candidate_rec = float(metric.loc[(args.candidate, "rec"), "test_ndcg10"])
    fixed_rec = float(metric.loc[("Fixed", "rec"), "test_ndcg10"])
    candidate_search = float(metric.loc[(args.candidate, "search"), "test_ndcg10"])
    fixed_search = float(metric.loc[("Fixed", "search"), "test_ndcg10"])
    subgroup_index = subgroup.set_index("group")
    support_independent = any(
        group in subgroup_index.index
        and float(
            subgroup_index.loc[
                group, f"{args.candidate}_minus_Support_ndcg10"
            ]
        )
        > 0
        for group in (
            "injected_target",
            "drift_eligible_target",
            f"injected_{candidate_key}_either_recoded",
        )
    )
    gate = {
        "candidate": args.candidate,
        "rec_test_better_than_fixed": candidate_rec > fixed_rec,
        "search_relative_drop_le_1pct": candidate_search >= 0.99 * fixed_search,
        "subgroup_better_than_support": support_independent,
    }
    gate["passed"] = all(value for key, value in gate.items() if key != "candidate")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.output_dir / "overall_metrics.csv", index=False)
    subgroup.to_csv(args.output_dir / "rec_subgroup_metrics.csv", index=False)
    summary = {
        "protocol": {
            "update_window": args.drift_right,
            "validation_window": args.drift_right,
            "test_window": args.test_window,
            "seed": 2024,
            "same_downstream_data": True,
            "same_initial_checkpoint": True,
            "same_alignment_pool": True,
        },
        "sid_update": {
            method: {
                "selected": len(selected[method]),
                "recoded": len(recoded[method]),
            }
            for method in methods[1:]
        },
        "smoke_gate": gate,
        "overall": overall.to_dict("records"),
        "rec_subgroups": subgroup.to_dict("records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overall.to_string(index=False))
    print("\n" + subgroup.to_string(index=False))
    print("\nSmoke gate:", json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
