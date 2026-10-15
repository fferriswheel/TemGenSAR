#!/usr/bin/env python3
"""Diagnose whether a temporal Rec SID update follows the updated behavior space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--item-ids", type=Path, required=True)
    parser.add_argument("--base-mapping", type=Path, required=True)
    parser.add_argument("--updated-mapping", type=Path, required=True)
    parser.add_argument("--t0-embedding", type=Path, required=True)
    parser.add_argument("--updated-embedding", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:4")
    return parser.parse_args()


def load_item_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [str(json.loads(line)["item_id"]) for line in handle]


def load_embedding(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=False).float()
    norm = value.norm(dim=1)
    return torch.nn.functional.normalize(value, dim=1), norm


@torch.no_grad()
def topk_neighbors(embedding: torch.Tensor, k: int, batch_size: int, device: torch.device) -> np.ndarray:
    database = embedding.to(device)
    output = []
    for start in range(0, len(database), batch_size):
        end = min(start + batch_size, len(database))
        scores = database[start:end] @ database.T
        rows = torch.arange(end - start, device=device)
        scores[rows, torch.arange(start, end, device=device)] = -float("inf")
        output.append(scores.topk(k, dim=1).indices.cpu().numpy())
    return np.concatenate(output)


def rec_sid(mapping: dict, item_ids: list[str]) -> np.ndarray:
    return np.asarray([
        [*mapping[item]["prefix_codes"], *mapping[item]["rec_suffix_codes"]]
        for item in item_ids
    ], dtype=np.int64)


def neighbor_sid_metrics(sids: np.ndarray, neighbors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbor_sids = sids[neighbors]
    same = neighbor_sids == sids[:, None, :]
    # Prefix route plus R1 is the route within which R2 is assigned.
    route_agreement = same[:, :, :3].all(axis=2).mean(axis=1)
    suffix_agreement = same[:, :, 2:].mean(axis=(1, 2))
    lcp = np.cumprod(same.astype(np.int8), axis=2).sum(axis=2).mean(axis=1) / sids.shape[1]
    return route_agreement, suffix_agreement, lcp


def neighbor_overlap(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray([
        len(set(a).intersection(b)) / len(a) for a, b in zip(left, right)
    ], dtype=np.float64)


def safe_spearman(left: pd.Series, right: pd.Series) -> dict[str, float]:
    result = spearmanr(left.to_numpy(), right.to_numpy(), nan_policy="omit")
    return {"rho": float(result.statistic), "p_value": float(result.pvalue)}


def summarize_group(frame: pd.DataFrame) -> dict[str, float | int]:
    t0_available = frame[frame["t0_embedding_available"]]
    return {
        "items": len(frame),
        "embedding_cosine_shift_mean": float(frame["embedding_cosine_shift"].mean()),
        "embedding_neighbor_overlap_mean": float(frame["embedding_neighbor_overlap"].mean()),
        "old_route_agreement_w1_mean": float(frame["old_route_agreement_w1"].mean()),
        "new_route_agreement_w1_mean": float(frame["new_route_agreement_w1"].mean()),
        "route_agreement_delta_mean": float(frame["route_agreement_delta"].mean()),
        "old_suffix_agreement_w1_mean": float(frame["old_suffix_agreement_w1"].mean()),
        "new_suffix_agreement_w1_mean": float(frame["new_suffix_agreement_w1"].mean()),
        "suffix_agreement_delta_mean": float(frame["suffix_agreement_delta"].mean()),
        "old_lcp_w1_mean": float(frame["old_lcp_w1"].mean()),
        "new_lcp_w1_mean": float(frame["new_lcp_w1"].mean()),
        "lcp_delta_mean": float(frame["lcp_delta"].mean()),
        "route_improved_fraction": float((frame["route_agreement_delta"] > 0).mean()),
        "route_worsened_fraction": float((frame["route_agreement_delta"] < 0).mean()),
        "t0_embedding_available_fraction": float(frame["t0_embedding_available"].mean()),
        "embedding_cosine_shift_mean_t0_available": (
            float(t0_available["embedding_cosine_shift"].mean()) if len(t0_available) else float("nan")
        ),
        "embedding_neighbor_overlap_mean_t0_available": (
            float(t0_available["embedding_neighbor_overlap"].mean()) if len(t0_available) else float("nan")
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations = pd.read_csv(args.annotations).sort_values("item_int").reset_index(drop=True)
    item_ids = load_item_ids(args.item_ids)
    if annotations["item_id"].astype(str).tolist() != item_ids:
        raise ValueError("Annotation order does not match embedding item order")

    t0, t0_norm = load_embedding(args.t0_embedding)
    updated, updated_norm = load_embedding(args.updated_embedding)
    if len(t0) != len(item_ids) or updated.shape != t0.shape:
        raise ValueError("Embedding shape does not match item universe")
    device = torch.device(args.device)
    t0_neighbors = topk_neighbors(t0, args.top_k, args.batch_size, device)
    updated_neighbors = topk_neighbors(updated, args.top_k, args.batch_size, device)

    base_mapping = json.loads(args.base_mapping.read_text(encoding="utf-8"))
    updated_mapping = json.loads(args.updated_mapping.read_text(encoding="utf-8"))
    old_sid = rec_sid(base_mapping, item_ids)
    new_sid = rec_sid(updated_mapping, item_ids)
    old_route, old_suffix, old_lcp = neighbor_sid_metrics(old_sid, updated_neighbors)
    new_route, new_suffix, new_lcp = neighbor_sid_metrics(new_sid, updated_neighbors)

    quality = annotations.copy()
    quality["t0_embedding_available"] = t0_norm.numpy() > 1e-8
    quality["updated_embedding_available"] = updated_norm.numpy() > 1e-8
    quality["embedding_cosine_shift"] = 1.0 - (t0 * updated).sum(dim=1).numpy()
    quality["embedding_neighbor_overlap"] = neighbor_overlap(t0_neighbors, updated_neighbors)
    quality["old_route_agreement_w1"] = old_route
    quality["new_route_agreement_w1"] = new_route
    quality["route_agreement_delta"] = new_route - old_route
    quality["old_suffix_agreement_w1"] = old_suffix
    quality["new_suffix_agreement_w1"] = new_suffix
    quality["suffix_agreement_delta"] = new_suffix - old_suffix
    quality["old_lcp_w1"] = old_lcp
    quality["new_lcp_w1"] = new_lcp
    quality["lcp_delta"] = new_lcp - old_lcp
    quality.to_csv(args.output_dir / "sid_quality_by_item.csv", index=False)

    summary = {
        "top_k": args.top_k,
        "groups": {
            group: summarize_group(rows)
            for group, rows in quality.groupby("update_group", sort=True)
        },
        "changed_by_popularity": {
            group: summarize_group(rows)
            for group, rows in quality[quality["changed"]].groupby("popularity_group", sort=True)
        },
        "correlations_active_items": {
            "drift_score_vs_embedding_shift": safe_spearman(
                quality.loc[quality["active"], "drift_rank_score"],
                quality.loc[quality["active"], "embedding_cosine_shift"],
            ),
            "drift_score_vs_changed": safe_spearman(
                quality.loc[quality["active"], "drift_rank_score"],
                quality.loc[quality["active"], "changed"].astype(float),
            ),
            "embedding_shift_vs_changed": safe_spearman(
                quality.loc[quality["active"], "embedding_cosine_shift"],
                quality.loc[quality["active"], "changed"].astype(float),
            ),
        },
    }
    (args.output_dir / "sid_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
