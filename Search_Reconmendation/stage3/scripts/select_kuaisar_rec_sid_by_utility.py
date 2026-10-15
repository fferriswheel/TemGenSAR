#!/usr/bin/env python3
"""Select candidate Rec SID changes using paired downstream utility on a gate set."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def instance_metrics(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    hits = np.asarray(payload["all_prompt_instance_results"][0], dtype=np.float64)
    hit10 = (hits[:, :10].sum(axis=1) > 0).astype(np.float64)
    ndcg10 = (hits[:, :10] / np.log2(np.arange(2, 12))[None, :]).sum(axis=1)
    return hit10, ndcg10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-pickle", type=Path, required=True)
    parser.add_argument("--fixed-result", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--base-sid", type=Path, required=True)
    parser.add_argument("--candidate-sid", type=Path, required=True)
    parser.add_argument("--item-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-support", type=int, action="append", default=[])
    parser.add_argument("--min-delta", type=float, default=0.0)
    args = parser.parse_args()

    supports = sorted(set(args.min_support or [1, 3]))
    gate = pd.read_pickle(args.gate_pickle).reset_index(drop=True)
    fixed_hit, fixed_ndcg = instance_metrics(args.fixed_result)
    candidate_hit, candidate_ndcg = instance_metrics(args.candidate_result)
    if len(gate) != len(fixed_hit) or len(gate) != len(candidate_hit):
        raise ValueError("Gate rows and instance-result rows do not match")

    base = json.loads(args.base_sid.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_sid.read_text(encoding="utf-8"))
    item_ids = [json.loads(line)["item_id"] for line in args.item_ids.read_text().splitlines()]
    changed = sorted(int(key) for key in base if base[key] != candidate[key])

    rows = []
    targets = gate["item_id"].astype(int).to_numpy()
    for item in changed:
        mask = targets == item
        support = int(mask.sum())
        row = {
            "item_index": item,
            "item_id": str(item_ids[item - 1]),
            "support": support,
            "fixed_hit10": float(fixed_hit[mask].mean()) if support else None,
            "candidate_hit10": float(candidate_hit[mask].mean()) if support else None,
            "delta_hit10": float((candidate_hit[mask] - fixed_hit[mask]).mean()) if support else None,
            "fixed_ndcg10": float(fixed_ndcg[mask].mean()) if support else None,
            "candidate_ndcg10": float(candidate_ndcg[mask].mean()) if support else None,
            "delta_ndcg10": float((candidate_ndcg[mask] - fixed_ndcg[mask]).mean()) if support else None,
        }
        rows.append(row)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "item_utility.csv", index=False)
    summaries = {}
    for support in supports:
        selected = frame[
            (frame["support"] >= support) & (frame["delta_ndcg10"] > args.min_delta)
        ]
        selected_indices = set(selected["item_index"].astype(int))
        gated = {
            key: candidate[key] if int(key) in selected_indices else value
            for key, value in base.items()
        }
        stem = f"support{support}"
        (output / f"collaborate_{stem}.json").write_text(
            json.dumps(gated, indent=2), encoding="utf-8"
        )
        (output / f"selected_internal_{stem}.json").write_text(
            json.dumps(sorted(selected_indices), indent=2), encoding="utf-8"
        )
        (output / f"selected_items_{stem}.json").write_text(
            json.dumps(selected["item_id"].tolist(), indent=2), encoding="utf-8"
        )
        summaries[stem] = {
            "min_support": support,
            "min_delta": args.min_delta,
            "selected_items": len(selected),
            "selected_gate_rows": int(selected["support"].sum()),
            "mean_item_delta_ndcg10": float(selected["delta_ndcg10"].mean()) if len(selected) else None,
        }

    summary = {
        "candidate_changes": len(changed),
        "gate_rows": len(gate),
        "covered_candidates": int((frame["support"] > 0).sum()),
        "variants": summaries,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
