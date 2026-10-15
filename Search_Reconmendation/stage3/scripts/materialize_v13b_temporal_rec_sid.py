#!/usr/bin/env python3
"""Commit a Rec-only SID update while freezing inactive items and shared/Search codes."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from build_neural_rq_sid_v2 import load_item_rows, write_json
from build_neural_rq_sid_v3_aligned_delta import load_behavior_signal, load_precomputed_embedding
from build_neural_rq_sid_v7_anchored_cross_view import mapping_from_codes
from build_neural_rq_sid_v9_direct_private import DirectPrivateTokenizer
from train_multiview_anchor_prefix_v13 import MultiViewAnchorRQVAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--adapter-checkpoint", type=Path, required=True)
    p.add_argument("--base-operational-mapping", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--common-embedding", required=True)
    p.add_argument("--updated-rec-embedding", required=True)
    p.add_argument("--anchor-lookup", type=Path, required=True)
    p.add_argument("--rec-interactions", type=Path, required=True)
    p.add_argument("--update-start", type=int, required=True)
    p.add_argument("--update-end", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", default="cuda:4")
    return p.parse_args()


def full_collision_stats(prefix, suffix):
    counts = Counter(tuple(p) + tuple(s) for p, s in zip(prefix, suffix))
    n = len(prefix)
    return {
        "unique_sid_count": len(counts),
        "duplicate_fraction": (n - len(counts)) / n,
        "collided_item_fraction": sum(v for v in counts.values() if v > 1) / n,
        "collision_groups": sum(v > 1 for v in counts.values()),
        "max_bucket_size": max(counts.values()),
    }


def load_active_items(path, start, end):
    active = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            timestamp = int(row["timestamp"])
            if start <= timestamp < end:
                active.add(str(row["item_id"]))
    return active


@torch.no_grad()
def encode_rec_private(model, z_shared, behavior_x, batch_size, device):
    model.eval()
    values = []
    for start in range(0, len(z_shared), batch_size):
        end = min(start + batch_size, len(z_shared))
        values.append(
            model.rec_adapter(
                torch.from_numpy(z_shared[start:end]).to(device),
                torch.from_numpy(behavior_x[start:end]).to(device),
            ).cpu().numpy()
        )
    return np.concatenate(values).astype(np.float32)


def squared_distances(x, centers):
    return (
        np.sum(x * x, axis=1, keepdims=True)
        + np.sum(centers * centers, axis=1)[None, :]
        - 2 * x @ centers.T
    )


def assign_active_last_codes(prefix, candidate, active_mask, private, codebooks, base_rec):
    """Assign active R2 codes within fixed (prefix, candidate R1) routes."""
    result = base_rec.copy()
    result[active_mask, 0] = candidate[active_mask, 0]
    first_centers = codebooks[0].detach().cpu().numpy().astype(np.float32)
    last_centers = codebooks[1].detach().cpu().numpy().astype(np.float32)
    residual = private - first_centers[result[:, 0]]
    routes = defaultdict(list)
    frozen_occupied = defaultdict(set)
    for idx in np.flatnonzero(~active_mask):
        route = tuple(prefix[idx]) + (int(base_rec[idx, 0]),)
        frozen_occupied[route].add(int(base_rec[idx, 1]))
    for idx in np.flatnonzero(active_mask):
        routes[tuple(prefix[idx]) + (int(result[idx, 0]),)].append(int(idx))

    overflow_items = 0
    for route, active_indices in routes.items():
        occupied = frozen_occupied[route]
        free = np.asarray([code for code in range(len(last_centers)) if code not in occupied])
        indices = np.asarray(active_indices, dtype=np.int64)
        distances = squared_distances(residual[indices], last_centers)
        if len(free):
            rows, columns = linear_sum_assignment(distances[:, free])
            result[indices[rows], 1] = free[columns]
            assigned_rows = set(int(row) for row in rows)
        else:
            assigned_rows = set()
        remaining = [row for row in range(len(indices)) if row not in assigned_rows]
        if remaining:
            overflow_items += len(remaining)
            result[indices[remaining], 1] = distances[remaining].argmin(axis=1)
    return result, {"routes": len(routes), "overflow_items": overflow_items}


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    base_checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    adapter_checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
    params = argparse.Namespace(**base_checkpoint["params"])
    item_ids, _, _ = load_item_rows(args.data_dir, "", params.max_items)
    common_x = load_precomputed_embedding(args.common_embedding, item_ids, "common")
    behavior_x, _ = load_behavior_signal(args.updated_rec_embedding, item_ids)
    lookup = np.load(args.anchor_lookup)
    if [str(x) for x in lookup["item_ids"].tolist()] != item_ids:
        raise ValueError("anchor lookup item order mismatch")
    prefix = lookup["prefix_codes"].astype(np.int64)
    z_shared = lookup["z_anchor"].astype(np.float32)

    anchor_checkpoint = torch.load(params.anchor_checkpoint, map_location="cpu", weights_only=False)
    anchor_args = argparse.Namespace(**anchor_checkpoint["params"])
    shared = MultiViewAnchorRQVAE(
        common_x.shape[1], behavior_x.shape[1], 768,
        base_checkpoint["prefix_sizes"], anchor_args,
    )
    model = DirectPrivateTokenizer(
        shared, common_x.shape[1], 768, behavior_x.shape[1],
        base_checkpoint["search_sizes"], base_checkpoint["rec_sizes"], params,
    ).to(device)
    model.load_state_dict(adapter_checkpoint["model_state_dict"])
    private = encode_rec_private(model, z_shared, behavior_x, args.batch_size, device)

    base_mapping = json.loads(args.base_operational_mapping.read_text(encoding="utf-8"))
    base_search = np.asarray([base_mapping[x]["search_suffix_codes"] for x in item_ids], dtype=np.int64)
    base_rec = np.asarray([base_mapping[x]["rec_suffix_codes"] for x in item_ids], dtype=np.int64)
    candidate = []
    residual = private.copy()
    for codebook in model.rec_quantizer.codebooks:
        centers = codebook.detach().cpu().numpy().astype(np.float32)
        code = squared_distances(residual, centers).argmin(axis=1)
        candidate.append(code)
        residual -= centers[code]
    candidate = np.stack(candidate, axis=1).astype(np.int64)

    active_items = load_active_items(args.rec_interactions, args.update_start, args.update_end)
    active_mask = np.asarray([item_id in active_items for item_id in item_ids])
    refined_rec, assignment = assign_active_last_codes(
        prefix, candidate, active_mask, private, model.rec_quantizer.codebooks, base_rec
    )
    mapping = mapping_from_codes(item_ids, prefix, base_search, refined_rec)
    output_mapping = args.output_dir / "sid_mapping.temporal_rec_update.json"
    write_json(output_mapping, mapping)

    changed = np.any(refined_rec != base_rec, axis=1)
    stats = {
        "method": "v13b Rec adapter update + active-only constrained R2 assignment",
        "update_window": {"start": args.update_start, "end": args.update_end},
        "num_items": len(item_ids),
        "active_items": int(active_mask.sum()),
        "prefix_exactly_preserved": True,
        "search_exactly_preserved": True,
        "inactive_rec_exactly_preserved": bool(np.array_equal(refined_rec[~active_mask], base_rec[~active_mask])),
        "changed_all_fraction": float(changed.mean()),
        "changed_active_fraction": float(changed[active_mask].mean()),
        "base_rec_collision": full_collision_stats(prefix, base_rec),
        "updated_rec_collision": full_collision_stats(prefix, refined_rec),
        "assignment": assignment,
        "mapping": str(output_mapping),
    }
    write_json(args.output_dir / "temporal_sid_stats.json", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
