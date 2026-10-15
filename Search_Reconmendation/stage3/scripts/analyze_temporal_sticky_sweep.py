#!/usr/bin/env python3
"""Compare topology and churn across sticky temporal SID sweep mappings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analyze_temporal_r1_sid_quality import (
    load_embedding,
    load_item_ids,
    neighbor_sid_metrics,
    rec_sid,
    topk_neighbors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-summary", type=Path, required=True)
    parser.add_argument("--item-ids", type=Path, required=True)
    parser.add_argument("--base-mapping", type=Path, required=True)
    parser.add_argument("--updated-embedding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:4")
    return parser.parse_args()


def mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if mask.any() else float("nan")


def main() -> None:
    args = parse_args()
    sweep = json.loads(args.sweep_summary.read_text(encoding="utf-8"))
    item_ids = load_item_ids(args.item_ids)
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    updated_embedding, _ = load_embedding(args.updated_embedding)
    neighbors = topk_neighbors(
        updated_embedding, args.top_k, args.batch_size, torch.device(args.device)
    )
    base_mapping = json.loads(args.base_mapping.read_text(encoding="utf-8"))
    base_sid = rec_sid(base_mapping, item_ids)
    base_route, base_suffix, base_lcp = neighbor_sid_metrics(base_sid, neighbors)
    selections = {
        "drift_selected": set(sweep["selection"]["drift_selected_items"]),
        "budget_random": set(sweep["selection"]["matched_random_items"]),
    }

    rows = []
    for run in sweep["runs"]:
        mapping = json.loads(Path(run["mapping"]).read_text(encoding="utf-8"))
        sid = rec_sid(mapping, item_ids)
        route, suffix, lcp = neighbor_sid_metrics(sid, neighbors)
        changed = np.any(sid != base_sid, axis=1)
        selected = np.zeros(len(item_ids), dtype=bool)
        selected[[item_to_index[item] for item in selections[run["policy"]]]] = True
        rows.append({
            "policy": run["policy"],
            "switch_ratio": run["switch_ratio"],
            "selected_items": int(selected.sum()),
            "changed_items": int(changed.sum()),
            "r1_changed_items": run["r1_changed_items"],
            "r2_changed_items": run["r2_changed_items"],
            "duplicate_fraction": run["collision"]["duplicate_fraction"],
            "selected_route_delta": mean(route - base_route, selected),
            "changed_route_delta": mean(route - base_route, changed),
            "selected_suffix_delta": mean(suffix - base_suffix, selected),
            "changed_suffix_delta": mean(suffix - base_suffix, changed),
            "selected_lcp_delta": mean(lcp - base_lcp, selected),
            "changed_lcp_delta": mean(lcp - base_lcp, changed),
            "changed_route_improved_fraction": mean((route - base_route) > 0, changed),
            "changed_route_worsened_fraction": mean((route - base_route) < 0, changed),
            "mapping": run["mapping"],
        })
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
