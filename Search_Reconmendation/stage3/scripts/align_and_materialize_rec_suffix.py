#!/usr/bin/env python3
"""Align a refreshed Rec suffix tokenizer to t0 tokens and selectively publish SIDs."""

from __future__ import annotations

import argparse
import copy
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
    parser.add_argument("--refreshed-checkpoint", type=Path, required=True)
    parser.add_argument("--base-operational-mapping", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--common-embedding", required=True)
    parser.add_argument("--updated-rec-embedding", required=True)
    parser.add_argument("--anchor-lookup", type=Path, required=True)
    parser.add_argument("--drift-csv", type=Path, required=True)
    parser.add_argument("--drift-left", default="C8")
    parser.add_argument("--drift-right", default="W1")
    parser.add_argument("--identity-prior", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def align_token_levels(
    old_codes: np.ndarray,
    new_codes: np.ndarray,
    sizes: list[int],
    identity_prior: float,
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, float | int]]]:
    aligned = np.empty_like(new_codes)
    permutations = []
    stats = []
    for level, size in enumerate(sizes):
        cooccurrence = np.zeros((size, size), dtype=np.float64)
        np.add.at(cooccurrence, (new_codes[:, level], old_codes[:, level]), 1.0)
        score = cooccurrence + identity_prior * np.eye(size, dtype=np.float64)
        new_rows, old_columns = linear_sum_assignment(-score)
        new_to_old = np.empty(size, dtype=np.int64)
        new_to_old[new_rows] = old_columns
        aligned[:, level] = new_to_old[new_codes[:, level]]
        permutations.append(new_to_old)
        stats.append({
            "level": level,
            "size": size,
            "supported_new_tokens": int((cooccurrence.sum(axis=1) > 0).sum()),
            "raw_item_agreement": float((new_codes[:, level] == old_codes[:, level]).mean()),
            "aligned_item_agreement": float((aligned[:, level] == old_codes[:, level]).mean()),
            "matched_cooccurrence": int(cooccurrence[new_rows, old_columns].sum()),
        })
    return aligned, permutations, stats


def selective_unique_assignment(
    prefix: np.ndarray,
    old_codes: np.ndarray,
    proposals: np.ndarray,
    selected_mask: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, dict[str, int | float | bool]]:
    base_keys = [tuple(p) + tuple(s) for p, s in zip(prefix, old_codes)]
    if len(set(base_keys)) != len(base_keys):
        raise ValueError("Base operational Rec SIDs must be unique for collision-safe refresh")

    result = old_codes.copy()
    selected_by_prefix: dict[tuple[int, ...], list[int]] = defaultdict(list)
    frozen_by_prefix: dict[tuple[int, ...], set[tuple[int, ...]]] = defaultdict(set)
    for index, prefix_code in enumerate(prefix):
        key = tuple(int(value) for value in prefix_code)
        if selected_mask[index]:
            selected_by_prefix[key].append(index)
        else:
            frozen_by_prefix[key].add(tuple(int(value) for value in old_codes[index]))

    rejected_frozen_collision = 0
    for prefix_key, indices_list in selected_by_prefix.items():
        indices = np.asarray(indices_list, dtype=np.int64)
        frozen = frozen_by_prefix[prefix_key]
        candidates = set(tuple(int(value) for value in old_codes[index]) for index in indices)
        for index in indices:
            proposal = tuple(int(value) for value in proposals[index])
            if proposal not in frozen:
                candidates.add(proposal)
            elif proposal != tuple(int(value) for value in old_codes[index]):
                rejected_frozen_collision += 1
        columns = sorted(candidates)
        column_index = {code: position for position, code in enumerate(columns)}
        cost = np.full((len(indices), len(columns)), 1e6, dtype=np.float64)
        for row, index in enumerate(indices):
            old = tuple(int(value) for value in old_codes[index])
            proposal = tuple(int(value) for value in proposals[index])
            cost[row, column_index[old]] = 1.0
            if proposal in column_index:
                cost[row, column_index[proposal]] = min(
                    cost[row, column_index[proposal]], 1.0 - float(confidence[index])
                )
        rows, assigned_columns = linear_sum_assignment(cost)
        if len(rows) != len(indices) or np.any(cost[rows, assigned_columns] >= 1e5):
            raise RuntimeError(f"No feasible collision-safe assignment for prefix {prefix_key}")
        for row, column in zip(rows, assigned_columns):
            result[indices[row]] = np.asarray(columns[column], dtype=np.int64)

    changed = np.any(result != old_codes, axis=1)
    return result, {
        "selected_items": int(selected_mask.sum()),
        "changed_selected_items": int(changed[selected_mask].sum()),
        "changed_all_fraction": float(changed.mean()),
        "proposal_changes_selected": int(
            np.any(proposals[selected_mask] != old_codes[selected_mask], axis=1).sum()
        ),
        "rejected_frozen_collision_proposals": rejected_frozen_collision,
        "unchanged_outside_selection": bool(
            np.array_equal(result[~selected_mask], old_codes[~selected_mask])
        ),
    }


def normalized_confidence(frame: pd.DataFrame, item_ids: list[str]) -> np.ndarray:
    scores = frame.set_index("item_id")["drift_rank_score"].to_dict()
    values = np.asarray([float(scores.get(item, 0.0)) for item in item_ids], dtype=np.float64)
    selected_values = values[values > 0]
    if not len(selected_values):
        return np.zeros(len(item_ids), dtype=np.float64)
    low, high = np.quantile(selected_values, [0.05, 0.95])
    return np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    base_checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    refreshed = torch.load(args.refreshed_checkpoint, map_location="cpu", weights_only=False)
    params = argparse.Namespace(**base_checkpoint["params"])
    item_ids, _, _ = load_item_rows(args.data_dir, "", params.max_items)
    common_x = load_precomputed_embedding(args.common_embedding, item_ids, "common")
    behavior_x, _ = load_behavior_signal(args.updated_rec_embedding, item_ids)
    lookup = np.load(args.anchor_lookup)
    if [str(value) for value in lookup["item_ids"].tolist()] != item_ids:
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
    model.load_state_dict(refreshed["model_state_dict"])
    private = encode_rec_private(model, z_shared, behavior_x, args.batch_size, device)
    residual = private.copy()
    raw_codes = []
    for codebook in model.rec_quantizer.codebooks:
        centers = codebook.detach().cpu().numpy().astype(np.float32)
        code = squared_distances(residual, centers).argmin(axis=1)
        raw_codes.append(code)
        residual -= centers[code]
    raw_codes = np.stack(raw_codes, axis=1).astype(np.int64)

    base_mapping = json.loads(args.base_operational_mapping.read_text(encoding="utf-8"))
    base_search = np.asarray(
        [base_mapping[item]["search_suffix_codes"] for item in item_ids], dtype=np.int64
    )
    base_rec = np.asarray(
        [base_mapping[item]["rec_suffix_codes"] for item in item_ids], dtype=np.int64
    )
    aligned, permutations, alignment_stats = align_token_levels(
        base_rec, raw_codes, list(base_checkpoint["rec_sizes"]), args.identity_prior
    )

    drift = pd.read_csv(args.drift_csv, dtype={"item_id": str})
    drift = drift[(drift["left"] == args.drift_left) & (drift["right"] == args.drift_right)].copy()
    drift["item_id"] = drift["item_id"].astype(str)
    selected_ids = set(drift.loc[drift["selected_for_update"], "item_id"])
    selected_mask = np.asarray([item in selected_ids for item in item_ids])
    confidence = normalized_confidence(drift, item_ids)
    operational, materialization_stats = selective_unique_assignment(
        prefix, base_rec, aligned, selected_mask, confidence
    )
    mapping = mapping_from_codes(item_ids, prefix, base_search, operational)
    mapping_path = args.output_dir / "sid_mapping.aligned_selective.json"
    write_json(mapping_path, mapping)

    aligned_state = copy.deepcopy(refreshed["model_state_dict"])
    for level, new_to_old in enumerate(permutations):
        key = f"rec_quantizer.codebooks.{level}"
        original = aligned_state[key]
        permuted = torch.empty_like(original)
        for new_code, old_code in enumerate(new_to_old):
            permuted[int(old_code)] = original[int(new_code)]
        aligned_state[key] = permuted
    aligned_checkpoint = args.output_dir / "rec_tokenizer_aligned.pt"
    torch.save(
        {
            "model_state_dict": aligned_state,
            "base_checkpoint": str(args.base_checkpoint),
            "refreshed_checkpoint": str(args.refreshed_checkpoint),
            "new_to_old_token_permutations": [value.tolist() for value in permutations],
        },
        aligned_checkpoint,
    )

    summary = {
        "method": "task-local refreshed Rec suffix + Hungarian token alignment + selective materialization",
        "drift_pair": [args.drift_left, args.drift_right],
        "identity_prior": args.identity_prior,
        "prefix_exactly_preserved": True,
        "search_suffix_exactly_preserved": True,
        "alignment": alignment_stats,
        "materialization": materialization_stats,
        "base_collision": full_collision_stats(prefix, base_rec),
        "raw_refreshed_collision": full_collision_stats(prefix, raw_codes),
        "aligned_refreshed_collision": full_collision_stats(prefix, aligned),
        "operational_collision": full_collision_stats(prefix, operational),
        "mapping": str(mapping_path),
        "aligned_checkpoint": str(aligned_checkpoint),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
