#!/usr/bin/env python3
"""Add train-only sequential-neighbor context to a Rec item embedding table."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from build_neural_rq_sid_v2 import aggregate_rec_neighbors, load_item_rows
from build_neural_rq_sid_v3_aligned_delta import load_behavior_signal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--base-embedding", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--neighbor-window", type=int, default=2)
    parser.add_argument("--max-neighbors-per-item", type=int, default=200)
    parser.add_argument("--context-weight", type=float, default=0.25)
    parser.add_argument("--max-items", type=int, default=0)
    args = parser.parse_args()
    if not 0.0 <= args.context_weight <= 1.0:
        raise ValueError("--context-weight must be in [0, 1]")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    item_ids, _, _ = load_item_rows(data_dir, "", args.max_items)
    base_np, _ = load_behavior_signal(args.base_embedding, item_ids)
    base = torch.from_numpy(base_np).float()
    _, counters = aggregate_rec_neighbors(
        data_dir,
        item_ids,
        args.neighbor_window,
        args.max_neighbors_per_item,
    )
    item_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    context = torch.empty_like(base)
    support = np.zeros(len(item_ids), dtype=np.int64)
    for idx, item_id in enumerate(item_ids):
        rows = [
            (item_to_index[neighbor], count)
            for neighbor, count in counters[item_id].most_common(args.max_neighbors_per_item)
            if neighbor in item_to_index
        ]
        if not rows:
            context[idx] = base[idx]
            continue
        indices = torch.tensor([row[0] for row in rows], dtype=torch.long)
        weights = torch.tensor([math.log1p(row[1]) for row in rows], dtype=torch.float32)
        context[idx] = (base[indices] * weights.unsqueeze(1)).sum(dim=0) / weights.sum()
        support[idx] = sum(row[1] for row in rows)
    context = F.normalize(context, dim=1)
    enriched = F.normalize(
        (1.0 - args.context_weight) * base + args.context_weight * context,
        dim=1,
    )
    torch.save(enriched, output_dir / "UniSAR_neighbor_enhanced_rec.pt")
    cosine = F.cosine_similarity(base, context, dim=1)
    summary = {
        "method": "UniSAR item embedding + log-count sequential-neighbor propagation",
        "future_behavior_used": False,
        "items": len(item_ids),
        "context_covered_items": int((support > 0).sum()),
        "context_coverage": float((support > 0).mean()),
        "context_weight": args.context_weight,
        "neighbor_window": args.neighbor_window,
        "max_neighbors_per_item": args.max_neighbors_per_item,
        "base_context_cosine_mean": float(cosine.mean()),
        "base_context_cosine_p10": float(torch.quantile(cosine, 0.1)),
        "base_enriched_cosine_mean": float(F.cosine_similarity(base, enriched, dim=1).mean()),
        "support_p50": int(np.quantile(support, 0.5)),
        "support_p90": int(np.quantile(support, 0.9)),
        "embedding_shape": list(enriched.shape),
        "params": vars(args),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
