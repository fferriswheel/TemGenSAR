#!/usr/bin/env python3
"""Materialize drift-selective hierarchical Search SIDs with switching costs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from build_neural_rq_sid_v2 import load_item_rows, write_json
from build_neural_rq_sid_v2_text_encoder_fusion_plc import l2_normalize_np
from build_neural_rq_sid_v3_aligned_delta import load_behavior_signal, load_precomputed_embedding
from build_neural_rq_sid_v7_anchored_cross_view import mapping_from_codes
from build_neural_rq_sid_v9_direct_private import DirectPrivateTokenizer
from materialize_v13b_temporal_rec_sid import full_collision_stats, squared_distances
from materialize_v13b_temporal_rec_sid_sticky import matched_random_items, quantiles
from train_multiview_anchor_prefix_v13 import MultiViewAnchorRQVAE
from update_v13b_rec_adapter import mapping_arrays
from update_v13b_search_adapter_selective import assign_search, search_signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--base-operational-mapping", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--common-embedding", required=True)
    parser.add_argument("--rec-embedding", required=True)
    parser.add_argument("--anchor-lookup", type=Path, required=True)
    parser.add_argument("--drift-csv", type=Path, required=True)
    parser.add_argument("--drift-left", default="C8")
    parser.add_argument("--drift-right", default="W1")
    parser.add_argument("--switch-ratios", default="0,0.01,0.05,0.1,0.2,0.5,1")
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def assign_sticky(
    prefix: np.ndarray,
    relevance: np.ndarray,
    identity: np.ndarray,
    relevance_centers: np.ndarray,
    identity_centers: np.ndarray,
    base_search: np.ndarray,
    selected_mask: np.ndarray,
    switch_ratio: float,
) -> tuple[np.ndarray, dict[str, object]]:
    result = base_search.copy()
    selected = np.flatnonzero(selected_mask)
    rel_dist = squared_distances(relevance[selected], relevance_centers)
    rel_old = rel_dist[np.arange(len(selected)), base_search[selected, 0]]
    rel_scale = float(np.median(rel_old))
    rel_penalty = switch_ratio * max(rel_scale, 1e-12)
    rel_cost = rel_dist + rel_penalty
    rel_cost[np.arange(len(selected)), base_search[selected, 0]] = rel_old
    result[selected, 0] = rel_cost.argmin(axis=1)

    id_dist_all = squared_distances(identity[selected], identity_centers)
    id_old = id_dist_all[np.arange(len(selected)), base_search[selected, 1]]
    id_scale = float(np.median(id_old))
    id_penalty = switch_ratio * max(id_scale, 1e-12)
    routes: dict[tuple[int, ...], list[int]] = defaultdict(list)
    frozen_occupied: dict[tuple[int, ...], set[int]] = defaultdict(set)
    for index in np.flatnonzero(~selected_mask):
        route = tuple(prefix[index]) + (int(base_search[index, 0]),)
        frozen_occupied[route].add(int(base_search[index, 1]))
    for index in selected:
        route = tuple(prefix[index]) + (int(result[index, 0]),)
        routes[route].append(int(index))

    overflow, advantages = 0, []
    for route, indices_list in routes.items():
        indices = np.asarray(indices_list, dtype=np.int64)
        occupied = frozen_occupied[route]
        free = [code for code in range(len(identity_centers)) if code not in occupied]
        distances = squared_distances(identity[indices], identity_centers)
        retained, movers = [], []
        for row, index in enumerate(indices):
            old_code = int(base_search[index, 1])
            same_route = int(result[index, 0]) == int(base_search[index, 0])
            alternatives = [code for code in free if code != old_code]
            best_alt = min((distances[row, code] for code in alternatives), default=float("inf"))
            advantage = float(distances[row, old_code] - best_alt)
            if np.isfinite(advantage):
                advantages.append(advantage)
            if same_route and advantage <= id_penalty:
                retained.append(row)
                result[index, 1] = old_code
            else:
                movers.append(row)
        retained_codes = {
            int(base_search[indices[row], 1])
            for row in retained
            if int(base_search[indices[row], 1]) not in occupied
        }
        available = np.asarray([code for code in free if code not in retained_codes])
        assigned_rows = set()
        if movers and len(available):
            costs = distances[movers][:, available] + id_penalty
            for local, route_row in enumerate(movers):
                old_code = int(base_search[indices[route_row], 1])
                matches = np.flatnonzero(available == old_code)
                if len(matches):
                    costs[local, matches[0]] = distances[route_row, old_code]
            assigned, columns = linear_sum_assignment(costs)
            for local, column in zip(assigned, columns):
                result[indices[movers[int(local)]], 1] = int(available[column])
            assigned_rows = {int(row) for row in assigned}
        for local in range(len(movers)):
            if local not in assigned_rows:
                overflow += 1
                index = indices[movers[local]]
                result[index, 1] = base_search[index, 1]

    changed = np.any(result != base_search, axis=1)
    best_rel = rel_dist.min(axis=1)
    return result, {
        "selected_items": int(selected_mask.sum()),
        "changed_selected_items": int(changed[selected_mask].sum()),
        "s1_changed_items": int((result[:, 0] != base_search[:, 0]).sum()),
        "s2_changed_items": int((result[:, 1] != base_search[:, 1]).sum()),
        "unchanged_outside_selection": bool(
            np.array_equal(result[~selected_mask], base_search[~selected_mask])
        ),
        "routes": len(routes),
        "overflow_items": overflow,
        "switch_ratio": switch_ratio,
        "s1_distance_scale": rel_scale,
        "s2_distance_scale": id_scale,
        "s1_effective_switch_penalty": rel_penalty,
        "s2_effective_switch_penalty": id_penalty,
        "s1_old_minus_best_distance": quantiles(rel_old - best_rel),
        "s2_old_minus_best_alternative_distance": quantiles(np.asarray(advantages)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    adapter = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
    params = argparse.Namespace(**base["params"])
    item_ids, _, _ = load_item_rows(args.data_dir, "", params.max_items)
    common_x = load_precomputed_embedding(args.common_embedding, item_ids, "common")
    behavior_x, _ = load_behavior_signal(args.rec_embedding, item_ids)
    lookup = np.load(args.anchor_lookup)
    prefix = lookup["prefix_codes"].astype(np.int64)
    z_shared = lookup["z_anchor"].astype(np.float32)
    q_shared = lookup["q_anchor"].astype(np.float32)
    semantic_base = l2_normalize_np(lookup["common_base"].astype(np.float32))
    anchor_ckpt = torch.load(params.anchor_checkpoint, map_location="cpu", weights_only=False)
    anchor_args = argparse.Namespace(**anchor_ckpt["params"])
    shared = MultiViewAnchorRQVAE(
        common_x.shape[1], behavior_x.shape[1], 768, base["prefix_sizes"], anchor_args
    )
    model = DirectPrivateTokenizer(
        shared, common_x.shape[1], 768, behavior_x.shape[1],
        base["search_sizes"], base["rec_sizes"], params,
    ).to(device)
    model.load_state_dict(adapter["model_state_dict"])
    search_x = search_signal(
        params, args.data_dir, item_ids, device, args.embedding_cache_dir
    )
    _, relevance, identity, _ = assign_search(
        model, z_shared, q_shared, common_x, semantic_base, search_x,
        args.batch_size, device,
    )
    rel_centers = model.search_relevance_quantizer.codebooks[0].detach().cpu().numpy()
    id_centers = model.search_identity_quantizer.codebooks[0].detach().cpu().numpy()

    base_mapping = json.loads(args.base_operational_mapping.read_text(encoding="utf-8"))
    base_search, base_rec = mapping_arrays(base_mapping, item_ids)
    drift = pd.read_csv(args.drift_csv, dtype={"item_id": str})
    drift = drift[(drift["left"] == args.drift_left) & (drift["right"] == args.drift_right)].copy()
    selected_frame = drift[drift["selected_for_update"]].copy()
    selected_items = set(selected_frame["item_id"].astype(str))
    random_items = matched_random_items(drift, selected_frame, args.seed)
    ratios = [float(value) for value in args.switch_ratios.split(",")]
    sweep = {
        "drift_pair": [args.drift_left, args.drift_right],
        "base_collision": full_collision_stats(prefix, base_search),
        "selection": {
            "drift_selected_items": sorted(selected_items),
            "matched_random_items": sorted(random_items),
            "selected_support_buckets": dict(Counter(selected_frame["support_bucket"])),
        },
        "runs": [],
    }
    for policy, selected_set in {
        "drift_selected": selected_items, "budget_random": random_items
    }.items():
        selected_mask = np.asarray([item in selected_set for item in item_ids])
        for ratio in ratios:
            search_codes, stats = assign_sticky(
                prefix, relevance, identity, rel_centers, id_centers,
                base_search, selected_mask, ratio,
            )
            collision = full_collision_stats(prefix, search_codes)
            tag = f"{policy}_sticky_ratio_{ratio:g}".replace(".", "p")
            output = args.output_dir / tag
            output.mkdir(parents=True, exist_ok=True)
            mapping_path = output / "sid_mapping.json"
            write_json(
                mapping_path,
                mapping_from_codes(item_ids, prefix, search_codes, base_rec),
            )
            row = {
                **stats, "policy": policy, "mapping": str(mapping_path),
                "collision": collision,
            }
            write_json(output / "stats.json", row)
            sweep["runs"].append(row)
            print(json.dumps({key: value for key, value in row.items() if key != "collision"}))
    write_json(args.output_dir / "sweep_summary.json", sweep)


if __name__ == "__main__":
    main()
