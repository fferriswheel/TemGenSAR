#!/usr/bin/env python3
"""Generate drift-selective temporal Rec SIDs with an old-code switching cost."""

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
from build_neural_rq_sid_v3_aligned_delta import load_behavior_signal, load_precomputed_embedding
from build_neural_rq_sid_v7_anchored_cross_view import mapping_from_codes
from build_neural_rq_sid_v9_direct_private import DirectPrivateTokenizer
from materialize_v13b_temporal_rec_sid import encode_rec_private, full_collision_stats, squared_distances
from train_multiview_anchor_prefix_v13 import MultiViewAnchorRQVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--base-operational-mapping", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--common-embedding", required=True)
    parser.add_argument("--updated-rec-embedding", required=True)
    parser.add_argument("--anchor-lookup", type=Path, required=True)
    parser.add_argument("--drift-csv", type=Path, required=True)
    parser.add_argument("--drift-left", default="C8")
    parser.add_argument("--drift-right", default="W1")
    parser.add_argument("--switch-ratios", default="0,0.01,0.05,0.1,0.2,0.5,1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def matched_random_items(drift: pd.DataFrame, selected: pd.DataFrame, seed: int) -> set[str]:
    """Match selected items by support bucket and nearest event support."""
    rng = np.random.default_rng(seed)
    available = drift[drift["eligible"] & ~drift["selected_for_update"]].copy()
    available["tie_break"] = rng.random(len(available))
    chosen: set[str] = set()
    for row in selected.sort_values(["support_bucket", "event_support", "item_id"]).itertuples():
        candidates = available[
            (available["support_bucket"] == row.support_bucket)
            & ~available["item_id"].isin(chosen)
        ].copy()
        if candidates.empty:
            candidates = available[~available["item_id"].isin(chosen)].copy()
        candidates["support_distance"] = (candidates["event_support"] - row.event_support).abs()
        pick = candidates.sort_values(["support_distance", "tie_break", "item_id"]).iloc[0]
        chosen.add(str(pick["item_id"]))
    return chosen


def quantiles(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {}
    return {
        name: float(np.quantile(values, q))
        for name, q in [("min", 0.0), ("p25", 0.25), ("p50", 0.5), ("p75", 0.75), ("p90", 0.9), ("max", 1.0)]
    }


def assign_sticky(
    prefix: np.ndarray,
    private: np.ndarray,
    codebooks: list[torch.Tensor],
    base_rec: np.ndarray,
    selected_mask: np.ndarray,
    switch_ratio: float,
) -> tuple[np.ndarray, dict[str, object]]:
    result = base_rec.copy()
    selected_indices = np.flatnonzero(selected_mask)
    first_centers = codebooks[0].detach().cpu().numpy().astype(np.float32)
    second_centers = codebooks[1].detach().cpu().numpy().astype(np.float32)

    first_distances = squared_distances(private[selected_indices], first_centers)
    first_old_distances = first_distances[
        np.arange(len(selected_indices)), base_rec[selected_indices, 0]
    ]
    first_scale = float(np.median(first_old_distances))
    first_penalty = switch_ratio * max(first_scale, 1e-12)
    first_costs = first_distances + first_penalty
    first_costs[np.arange(len(selected_indices)), base_rec[selected_indices, 0]] = first_distances[
        np.arange(len(selected_indices)), base_rec[selected_indices, 0]
    ]
    result[selected_indices, 0] = first_costs.argmin(axis=1)

    residual = private - first_centers[result[:, 0]]
    all_second_distances = squared_distances(residual[selected_indices], second_centers)
    all_second_old = all_second_distances[
        np.arange(len(selected_indices)), base_rec[selected_indices, 1]
    ]
    second_scale = float(np.median(all_second_old))
    second_penalty = switch_ratio * max(second_scale, 1e-12)
    routes: dict[tuple[int, ...], list[int]] = defaultdict(list)
    frozen_occupied: dict[tuple[int, ...], set[int]] = defaultdict(set)
    for index in np.flatnonzero(~selected_mask):
        route = tuple(prefix[index]) + (int(base_rec[index, 0]),)
        frozen_occupied[route].add(int(base_rec[index, 1]))
    for index in selected_indices:
        route = tuple(prefix[index]) + (int(result[index, 0]),)
        routes[route].append(int(index))

    overflow_items = 0
    second_old_advantages = []
    for route, indices_list in routes.items():
        indices = np.asarray(indices_list, dtype=np.int64)
        occupied = frozen_occupied[route]
        free = [code for code in range(len(second_centers)) if code not in occupied]
        distances = squared_distances(residual[indices], second_centers)
        retained_rows = []
        mover_rows = []
        for row, index in enumerate(indices):
            old_code = int(base_rec[index, 1])
            same_old_route = int(result[index, 0]) == int(base_rec[index, 0])
            alternatives = [code for code in free if code != old_code]
            best_alternative = min(
                (distances[row, code] for code in alternatives), default=float("inf")
            )
            advantage = float(distances[row, old_code] - best_alternative)
            if np.isfinite(advantage):
                second_old_advantages.append(advantage)
            # Existing collisions may be retained. Only newly selected codes must be free.
            if same_old_route and advantage <= second_penalty:
                retained_rows.append(row)
                result[index, 1] = old_code
            else:
                mover_rows.append(row)

        retained_free_codes = {
            int(base_rec[indices[row], 1])
            for row in retained_rows
            if int(base_rec[indices[row], 1]) not in occupied
        }
        available = np.asarray([code for code in free if code not in retained_free_codes])
        if mover_rows and len(available):
            costs = distances[mover_rows][:, available] + second_penalty
            for local_row, route_row in enumerate(mover_rows):
                old_code = int(base_rec[indices[route_row], 1])
                matches = np.flatnonzero(available == old_code)
                if len(matches):
                    costs[local_row, matches[0]] = distances[route_row, old_code]
            assigned, columns = linear_sum_assignment(costs)
            for local_row, column in zip(assigned, columns):
                result[indices[mover_rows[int(local_row)]], 1] = available[column]
            assigned_movers = {int(row) for row in assigned}
        else:
            assigned_movers = set()
        remaining = [row for row in range(len(mover_rows)) if row not in assigned_movers]
        if remaining:
            overflow_items += len(remaining)
            # Preserve the previous code instead of introducing an uncontrolled collision.
            for local_row in remaining:
                index = indices[mover_rows[local_row]]
                result[index, 1] = base_rec[index, 1]

    old_first = first_distances[np.arange(len(selected_indices)), base_rec[selected_indices, 0]]
    best_first = first_distances.min(axis=1)
    changed = np.any(result != base_rec, axis=1)
    return result, {
        "selected_items": int(selected_mask.sum()),
        "changed_selected_items": int(changed[selected_mask].sum()),
        "r1_changed_items": int((result[:, 0] != base_rec[:, 0]).sum()),
        "r2_changed_items": int((result[:, 1] != base_rec[:, 1]).sum()),
        "unchanged_outside_selection": bool(np.array_equal(result[~selected_mask], base_rec[~selected_mask])),
        "routes": len(routes),
        "overflow_items": overflow_items,
        "switch_ratio": switch_ratio,
        "r1_distance_scale": first_scale,
        "r2_distance_scale": second_scale,
        "r1_effective_switch_penalty": first_penalty,
        "r2_effective_switch_penalty": second_penalty,
        "r1_old_minus_best_distance": quantiles(old_first - best_first),
        "r2_old_minus_best_alternative_distance": quantiles(np.asarray(second_old_advantages)),
    }


def main() -> None:
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
    search_dim = int(getattr(params, "text_embedding_dim", 768))
    shared = MultiViewAnchorRQVAE(
        common_x.shape[1], behavior_x.shape[1], search_dim,
        base_checkpoint["prefix_sizes"], anchor_args,
    )
    model = DirectPrivateTokenizer(
        shared, common_x.shape[1], search_dim, behavior_x.shape[1],
        base_checkpoint["search_sizes"], base_checkpoint["rec_sizes"], params,
    ).to(device)
    model.load_state_dict(adapter_checkpoint["model_state_dict"])
    private = encode_rec_private(model, z_shared, behavior_x, args.batch_size, device)

    base_mapping = json.loads(args.base_operational_mapping.read_text(encoding="utf-8"))
    base_search = np.asarray([base_mapping[item]["search_suffix_codes"] for item in item_ids], dtype=np.int64)
    base_rec = np.asarray([base_mapping[item]["rec_suffix_codes"] for item in item_ids], dtype=np.int64)

    drift = pd.read_csv(args.drift_csv, dtype={"item_id": str})
    drift = drift[(drift["left"] == args.drift_left) & (drift["right"] == args.drift_right)].copy()
    drift["item_id"] = drift["item_id"].astype(str)
    selected_frame = drift[drift["selected_for_update"]].copy()
    selected_items = set(selected_frame["item_id"])
    random_items = matched_random_items(drift, selected_frame, args.seed)
    if len(random_items) != len(selected_items):
        raise ValueError("Matched-random selection size mismatch")
    policies = {"drift_selected": selected_items, "budget_random": random_items}
    ratios = [float(value) for value in args.switch_ratios.split(",")]

    sweep = {
        "drift_pair": [args.drift_left, args.drift_right],
        "base_collision": full_collision_stats(prefix, base_rec),
        "selection": {
            "drift_selected_items": sorted(selected_items),
            "matched_random_items": sorted(random_items),
            "selected_support_buckets": Counter(selected_frame["support_bucket"]),
            "random_support_buckets": Counter(
                drift.loc[drift["item_id"].isin(random_items), "support_bucket"]
            ),
        },
        "runs": [],
    }
    for policy, selected in policies.items():
        selected_mask = np.asarray([item in selected for item in item_ids])
        for ratio in ratios:
            rec_codes, stats = assign_sticky(
                prefix, private, model.rec_quantizer.codebooks, base_rec,
                selected_mask, ratio,
            )
            tag = f"{policy}_sticky_ratio_{ratio:g}".replace(".", "p")
            mapping = mapping_from_codes(item_ids, prefix, base_search, rec_codes)
            mapping_path = args.output_dir / tag / "sid_mapping.json"
            write_json(mapping_path, mapping)
            stats.update({
                "policy": policy,
                "switch_ratio": ratio,
                "mapping": str(mapping_path),
                "collision": full_collision_stats(prefix, rec_codes),
            })
            write_json(args.output_dir / tag / "stats.json", stats)
            sweep["runs"].append(stats)
    write_json(args.output_dir / "sweep_summary.json", sweep)
    print(json.dumps(sweep, ensure_ascii=False, indent=2, default=lambda value: dict(value)))


if __name__ == "__main__":
    main()
