#!/usr/bin/env python3
"""Compare a W6 task-local Rec adapter with its frozen R1 base."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REC_FILE = "SeqRecWithSrcHis_sample_neg-999_beam-10.json"
SEARCH_FILE = "PerSrcWithRecHis_sample_neg-999_beam-10.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--adapter-run", type=Path, required=True)
    parser.add_argument("--quarterly-dir", type=Path, required=True)
    parser.add_argument("--base-mapping", type=Path, required=True)
    parser.add_argument("--current-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_instances(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    hits = np.asarray(payload["all_prompt_instance_results"][0], dtype=np.float64)
    hit10 = (hits[:, :10].sum(axis=1) > 0).astype(np.float64)
    ndcg10 = (hits[:, :10] / np.log2(np.arange(2, 12))[None, :]).sum(axis=1)
    return payload["mean_results"], hit10, ndcg10


def main() -> None:
    args = parse_args()
    base_mapping = json.loads(args.base_mapping.read_text(encoding="utf-8"))
    current_mapping = json.loads(args.current_mapping.read_text(encoding="utf-8"))
    recoded = {
        item_id
        for item_id in base_mapping
        if base_mapping[item_id]["rec_suffix_codes"]
        != current_mapping[item_id]["rec_suffix_codes"]
    }
    id_maps = json.loads((args.quarterly_dir / "id_maps.json").read_text(encoding="utf-8"))
    int_to_asin = {int(key): str(value) for key, value in id_maps["int_to_asin"].items()}
    rec_eval = pd.read_pickle(args.quarterly_dir / "windows/W7/rec_eval.pkl")
    targets = rec_eval["item_id"].map(int_to_asin).to_numpy()
    masks = {
        "all": np.ones(len(targets), dtype=bool),
        "recoded_target": np.isin(targets, list(recoded)),
        "unchanged_target": ~np.isin(targets, list(recoded)),
    }

    methods = {
        "R1": args.baseline_run,
        "TaskLocalRec": args.adapter_run,
    }
    overall = []
    subgroup = []
    rec_values = {}
    for method, run in methods.items():
        rec_metric, hit10, ndcg10 = load_instances(run / REC_FILE)
        search_metric = json.loads((run / SEARCH_FILE).read_text(encoding="utf-8"))["mean_results"]
        rec_values[method] = (rec_metric, hit10, ndcg10)
        overall.extend(
            [
                {"method": method, "task": "rec", "hit10": rec_metric["hit@10"], "ndcg10": rec_metric["ndcg@10"]},
                {"method": method, "task": "search", "hit10": search_metric["hit@10"], "ndcg10": search_metric["ndcg@10"]},
            ]
        )
        for group, mask in masks.items():
            subgroup.append(
                {
                    "method": method,
                    "group": group,
                    "samples": int(mask.sum()),
                    "distinct_items": int(len(set(targets[mask]))),
                    "hit10": float(hit10[mask].mean()),
                    "ndcg10": float(ndcg10[mask].mean()),
                }
            )

    search_identical = digest(args.baseline_run / SEARCH_FILE) == digest(args.adapter_run / SEARCH_FILE)
    subgroup_frame = pd.DataFrame(subgroup)
    indexed = subgroup_frame.set_index(["method", "group"])
    gate = {
        "overall_rec_improves": bool(rec_values["TaskLocalRec"][0]["ndcg@10"] > rec_values["R1"][0]["ndcg@10"]),
        "recoded_target_improves": bool(indexed.loc[("TaskLocalRec", "recoded_target"), "ndcg10"] > indexed.loc[("R1", "recoded_target"), "ndcg10"]),
        "search_result_byte_identical": search_identical,
    }
    gate["passed"] = all(gate.values())
    summary = {
        "protocol": {
            "base": str(args.baseline_run),
            "adapter": "rank-8 q/v LoRA over frozen R1",
            "trainable_parameters": 294912,
            "affected_rec_rows": 5344,
            "unchanged_replay_rows": 4000,
            "rec_alignment_occurrences": 1800,
            "seed": 2024,
        },
        "gate": gate,
        "overall": overall,
        "rec_subgroups": subgroup,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(overall).to_csv(args.output_dir / "overall_metrics.csv", index=False)
    subgroup_frame.to_csv(args.output_dir / "rec_subgroup_metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(overall).to_string(index=False))
    print("\n" + subgroup_frame.to_string(index=False))
    print("\nGate:", json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
