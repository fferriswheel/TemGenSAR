#!/usr/bin/env python3
"""Summarize the fixed-budget changed-item alignment repeat curve."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarterly-dir", type=Path, required=True)
    parser.add_argument("--base-mapping", type=Path, required=True)
    parser.add_argument("--current-mapping", type=Path, required=True)
    parser.add_argument("--test-window", default="W7")
    parser.add_argument("--run", action="append", required=True, help="FACTOR=RUN_DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validation_metrics(run_dir: Path) -> dict[str, dict[str, float]]:
    text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    blocks = [
        ast.literal_eval(value)
        for value in re.findall(r"Mean results:\s+(\{[^\n]+\})", text.replace("\r", "\n"))
    ]
    if len(blocks) < 4:
        raise ValueError(f"Expected validation and test blocks in {run_dir}; got {len(blocks)}")
    return {"rec": blocks[0], "search": blocks[1]}


def instance_metrics(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("all_prompt_instance_results")
    if prompts is None or len(prompts) != 1:
        raise ValueError(f"Missing single-prompt instance results in {path}")
    hits = np.asarray(prompts[0], dtype=np.float64)
    hit10 = (hits[:, :10].sum(axis=1) > 0).astype(np.float64)
    ndcg10 = (hits[:, :10] / np.log2(np.arange(2, 12))[None, :]).sum(axis=1)
    return hit10, ndcg10, payload["mean_results"]


def main() -> None:
    args = parse_args()
    runs = {int(factor): Path(path) for factor, path in (spec.split("=", 1) for spec in args.run)}
    if 1 not in runs:
        raise ValueError("Repeat factor 1 is required as the control")

    base = json.loads(args.base_mapping.read_text(encoding="utf-8"))
    current = json.loads(args.current_mapping.read_text(encoding="utf-8"))
    recoded = {
        item_id
        for item_id in base
        if base[item_id]["rec_suffix_codes"] != current[item_id]["rec_suffix_codes"]
    }

    id_maps = json.loads((args.quarterly_dir / "id_maps.json").read_text(encoding="utf-8"))
    int_to_asin = {int(key): str(value) for key, value in id_maps["int_to_asin"].items()}
    rec_eval = pd.read_pickle(
        args.quarterly_dir / "windows" / args.test_window / "rec_eval.pkl"
    ).reset_index(drop=True)
    target_asin = rec_eval["item_id"].map(int_to_asin).to_numpy()
    masks = {
        "all": np.ones(len(target_asin), dtype=bool),
        "recoded_target": np.isin(target_asin, list(recoded)),
        "unchanged_target": ~np.isin(target_asin, list(recoded)),
    }

    rows = []
    subgroup_rows = []
    metrics_by_factor: dict[int, dict[str, dict[str, float]]] = {}
    for factor in sorted(runs):
        run_dir = runs[factor]
        validation = validation_metrics(run_dir)
        metrics_by_factor[factor] = {}
        rec_instances = None
        for task, filename in TASK_FILES.items():
            hit10, ndcg10, test = instance_metrics(run_dir / filename)
            metrics_by_factor[factor][task] = test
            rows.append(
                {
                    "repeat_factor": factor,
                    "task": task,
                    "val_hit10": validation[task]["hit@10"],
                    "val_ndcg10": validation[task]["ndcg@10"],
                    "test_hit10": test["hit@10"],
                    "test_ndcg10": test["ndcg@10"],
                }
            )
            if task == "rec":
                rec_instances = (hit10, ndcg10)
        assert rec_instances is not None
        hit10, ndcg10 = rec_instances
        if len(hit10) != len(target_asin):
            raise ValueError(f"W7 Rec instance count mismatch for R{factor}")
        for group, mask in masks.items():
            subgroup_rows.append(
                {
                    "repeat_factor": factor,
                    "group": group,
                    "samples": int(mask.sum()),
                    "distinct_items": int(len(set(target_asin[mask]))),
                    "hit10": float(hit10[mask].mean()),
                    "ndcg10": float(ndcg10[mask].mean()),
                }
            )

    overall = pd.DataFrame(rows).sort_values(["task", "repeat_factor"])
    subgroups = pd.DataFrame(subgroup_rows).sort_values(["group", "repeat_factor"])
    base_rec = metrics_by_factor[1]["rec"]["ndcg@10"]
    base_search = metrics_by_factor[1]["search"]["ndcg@10"]
    base_recoded = float(
        subgroups[(subgroups.repeat_factor == 1) & (subgroups.group == "recoded_target")][
            "ndcg10"
        ].iloc[0]
    )
    decisions = []
    for factor in sorted(runs):
        recoded_ndcg = float(
            subgroups[
                (subgroups.repeat_factor == factor) & (subgroups.group == "recoded_target")
            ]["ndcg10"].iloc[0]
        )
        decision = {
            "repeat_factor": factor,
            "recoded_target_improves": recoded_ndcg > base_recoded,
            "overall_rec_not_below_control": metrics_by_factor[factor]["rec"]["ndcg@10"] >= base_rec,
            "search_relative_drop_le_1pct": metrics_by_factor[factor]["search"]["ndcg@10"] >= 0.99 * base_search,
        }
        decision["passed"] = factor > 1 and all(
            value for key, value in decision.items() if key not in {"repeat_factor", "passed"}
        )
        decisions.append(decision)
    passing = [row["repeat_factor"] for row in decisions if row["passed"]]
    selected = min(
        passing,
        key=lambda factor: (-metrics_by_factor[factor]["rec"]["ndcg@10"], factor),
    ) if passing else None

    summary = {
        "protocol": {
            "test_window": args.test_window,
            "seed": 2024,
            "fixed_alignment_occurrence_budget": 1800,
            "recoded_items": len(recoded),
        },
        "decisions": decisions,
        "selected_repeat_factor": selected,
        "bridge_recommended": selected is not None,
        "overall": overall.to_dict("records"),
        "rec_subgroups": subgroups.to_dict("records"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.output_dir / "overall_metrics.csv", index=False)
    subgroups.to_csv(args.output_dir / "rec_subgroup_metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overall.to_string(index=False))
    print("\n" + subgroups.to_string(index=False))
    print("\n" + json.dumps({"decisions": decisions, "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
