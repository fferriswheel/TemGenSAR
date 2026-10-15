#!/usr/bin/env python3
"""Update the v13b Search adapter on drift-selected items plus bounded replay."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from build_neural_rq_sid_v2 import (
    aggregate_search_queries,
    code_usage,
    load_item_rows,
    write_json,
)
from build_neural_rq_sid_v2_text_encoder_fusion_plc import (
    build_search_signal_embeddings,
    l2_normalize_np,
)
from build_neural_rq_sid_v3_aligned_delta import (
    PrefixBatchSampler,
    conditional_collision_stats,
    inprefix_topology_loss,
    load_behavior_signal,
    load_precomputed_embedding,
)
from build_neural_rq_sid_v9_direct_private import (
    DirectPrivateTokenizer,
    cosine_loss,
    hierarchical_search_codes_np,
)
from train_multiview_anchor_prefix_v13 import MultiViewAnchorRQVAE
from update_v13b_rec_adapter import mapping_arrays
from update_v13b_rec_adapter_selective import select_training_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--t0-mapping", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--t0-data-dir", type=Path, required=True)
    parser.add_argument("--common-embedding", required=True)
    parser.add_argument("--rec-embedding", required=True)
    parser.add_argument("--anchor-lookup", type=Path, required=True)
    parser.add_argument("--drift-csv", type=Path, required=True)
    parser.add_argument("--drift-left", default="C8")
    parser.add_argument("--drift-right", default="W1")
    parser.add_argument("--replay-ratio", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--relevance-weight", type=float, default=0.2)
    parser.add_argument("--identity-weight", type=float, default=0.5)
    parser.add_argument("--topology-weight", type=float, default=0.5)
    parser.add_argument("--commit-weight", type=float, default=0.05)
    parser.add_argument("--stability-weight", type=float, default=0.1)
    parser.add_argument("--outside-stability-weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


@torch.no_grad()
def assign_search(
    model: DirectPrivateTokenizer,
    z_shared: np.ndarray,
    q_shared: np.ndarray,
    common_x: np.ndarray,
    semantic_base: np.ndarray,
    search_x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    private_rows, relevance_rows, identity_rows = [], [], []
    for start in range(0, len(search_x), batch_size):
        stop = start + batch_size
        private, relevance, identity = model.search_adapter(
            torch.from_numpy(z_shared[start:stop]).to(device),
            torch.from_numpy(q_shared[start:stop]).to(device),
            torch.from_numpy(common_x[start:stop]).to(device),
            torch.from_numpy(semantic_base[start:stop]).to(device),
            torch.from_numpy(search_x[start:stop]).to(device),
        )
        private_rows.append(private.cpu().numpy())
        relevance_rows.append(relevance.cpu().numpy())
        identity_rows.append(identity.cpu().numpy())
    private = np.concatenate(private_rows).astype(np.float32)
    relevance = np.concatenate(relevance_rows).astype(np.float32)
    identity = np.concatenate(identity_rows).astype(np.float32)
    codes = hierarchical_search_codes_np(model, relevance, identity)
    return private, relevance, identity, codes


def search_signal(
    params: argparse.Namespace,
    data_dir: Path,
    item_ids: list[str],
    device: torch.device,
    cache_dir: Path,
) -> np.ndarray:
    _, counter = aggregate_search_queries(data_dir, item_ids, params.max_queries_per_item)
    params.text_embedding_dim = 768
    value, _ = build_search_signal_embeddings(params, item_ids, counter, device, cache_dir)
    return value


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    params = argparse.Namespace(**checkpoint["params"])
    if not params.split_search_private or not params.hierarchical_search_vq:
        raise ValueError("Selective Search update requires split hierarchical Search suffixes")
    item_ids, _, _ = load_item_rows(args.data_dir, "", params.max_items)
    common_x = load_precomputed_embedding(args.common_embedding, item_ids, "common")
    behavior_x, _ = load_behavior_signal(args.rec_embedding, item_ids)
    lookup = np.load(args.anchor_lookup)
    if [str(value) for value in lookup["item_ids"].tolist()] != item_ids:
        raise ValueError("anchor lookup item order mismatch")
    prefix_codes = lookup["prefix_codes"].astype(np.int64)
    z_shared = lookup["z_anchor"].astype(np.float32)
    q_shared = lookup["q_anchor"].astype(np.float32)
    semantic_base = l2_normalize_np(lookup["common_base"].astype(np.float32))

    anchor_ckpt = torch.load(params.anchor_checkpoint, map_location="cpu", weights_only=False)
    anchor_args = argparse.Namespace(**anchor_ckpt["params"])
    shared = MultiViewAnchorRQVAE(
        common_x.shape[1], behavior_x.shape[1], 768,
        checkpoint["prefix_sizes"], anchor_args,
    )
    model = DirectPrivateTokenizer(
        shared, common_x.shape[1], 768, behavior_x.shape[1],
        checkpoint["search_sizes"], checkpoint["rec_sizes"], params,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    initial_state = copy.deepcopy(model.state_dict())
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.search_adapter.parameters():
        parameter.requires_grad = True
    model.shared.eval()
    model.rec_adapter.eval()
    model.search_relevance_quantizer.eval()
    model.search_identity_quantizer.eval()
    model.search_decoder.eval()
    model.relevance_decoder.eval()
    model.identity_decoder.eval()

    t0_search_x = search_signal(
        params, args.t0_data_dir, item_ids, device, args.embedding_cache_dir / "t0"
    )
    updated_search_x = search_signal(
        params, args.data_dir, item_ids, device, args.embedding_cache_dir / "updated"
    )
    t0_mapping = json.loads(args.t0_mapping.read_text(encoding="utf-8"))
    t0_search, _ = mapping_arrays(t0_mapping, item_ids)
    with torch.no_grad():
        old_private, _, _, _ = assign_search(
            model, z_shared, q_shared, common_x, semantic_base, t0_search_x,
            args.batch_size, device,
        )

    selected, train_indices, selection = select_training_items(
        args.drift_csv, item_ids, args.drift_left, args.drift_right,
        args.replay_ratio, args.seed,
    )
    outside_mask = np.ones(len(item_ids), dtype=bool)
    outside_mask[selected] = False
    dataset = TensorDataset(
        torch.from_numpy(common_x[train_indices]),
        torch.from_numpy(updated_search_x[train_indices]),
        torch.from_numpy(semantic_base[train_indices]),
        torch.from_numpy(prefix_codes[train_indices]),
        torch.from_numpy(z_shared[train_indices]),
        torch.from_numpy(q_shared[train_indices]),
        torch.from_numpy(old_private[train_indices]),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=PrefixBatchSampler(
            prefix_codes[train_indices], args.batch_size, 16, args.seed + 17
        ),
        num_workers=0,
    )
    trainable = list(model.search_adapter.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    best_state, best_score, best_epoch, stale = None, float("inf"), None, 0
    logs, optimizer_steps = [], 0
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.shared.eval()
        totals, steps = Counter(), 0
        for common_b, search_b, base_b, prefix_b, z_b, q_b, old_private_b in loader:
            common_b, search_b, base_b = common_b.to(device), search_b.to(device), base_b.to(device)
            prefix_b, z_b, q_b = prefix_b.to(device), z_b.to(device), q_b.to(device)
            old_private_b = old_private_b.to(device)
            private, relevance, identity = model.search_adapter(
                z_b, q_b, common_b, base_b, search_b
            )
            q_rel, rel_idx, rel_commit, _ = model.search_relevance_quantizer(relevance)
            q_id, _, id_commit, _ = model.search_identity_quantizer(identity)
            q_search = torch.cat([q_rel, q_id], dim=1)
            target_full = F.normalize(
                common_b + params.search_target_scale * search_b, dim=1
            )
            residual_target = target_full - base_b
            residual_recon = model.search_decoder(q_search, q_b)
            full_recon = F.normalize(base_b + residual_recon, dim=1)
            reconstruction = F.mse_loss(residual_recon, residual_target) + cosine_loss(
                full_recon, target_full
            )
            relevance_loss = cosine_loss(model.relevance_decoder(q_rel), search_b)
            identity_recon = model.identity_decoder(q_id)
            identity_loss = F.mse_loss(identity_recon, common_b) + 0.1 * cosine_loss(
                identity_recon, common_b
            )
            routes = torch.cat([prefix_b, rel_idx[0][:, None]], dim=1)
            topology = inprefix_topology_loss(routes, q_id, common_b)
            stability = F.mse_loss(private, old_private_b)
            commit = rel_commit + id_commit
            loss = reconstruction + args.relevance_weight * relevance_loss
            loss = loss + args.identity_weight * identity_loss
            loss = loss + args.topology_weight * topology
            loss = loss + args.commit_weight * commit + args.stability_weight * stability
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer_steps += 1
            for key, value in {
                "loss": loss, "reconstruction": reconstruction,
                "relevance": relevance_loss, "identity": identity_loss,
                "topology": topology, "commit": commit, "stability": stability,
            }.items():
                totals[key] += float(value.detach().cpu())
            steps += 1

        private_np, relevance_np, identity_np, search_codes = assign_search(
            model, z_shared, q_shared, common_x, semantic_base, updated_search_x,
            args.batch_size, device,
        )
        collision = conditional_collision_stats(prefix_codes, search_codes)
        outside_shift = float(np.mean((private_np[outside_mask] - old_private[outside_mask]) ** 2))
        selected_change = float(np.any(search_codes[selected] != t0_search[selected], axis=1).mean())
        row = {
            "epoch": epoch,
            **{key: value / max(steps, 1) for key, value in totals.items()},
            "outside_private_shift": outside_shift,
            "selected_raw_change": selected_change,
            "search_duplicate": collision["duplicate_fraction"],
        }
        score = row["reconstruction"]
        score += args.outside_stability_weight * outside_shift
        score += 0.5 * max(row["search_duplicate"] - 0.09, 0.0)
        row["selection_score"] = score
        logs.append(row)
        print(json.dumps(row), flush=True)
        if score < best_score - 1e-5:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Search adapter training produced no checkpoint")
    model.load_state_dict(best_state)
    private_np, relevance_np, identity_np, search_codes = assign_search(
        model, z_shared, q_shared, common_x, semantic_base, updated_search_x,
        args.batch_size, device,
    )
    final_state = model.state_dict()
    changed_tensors = [
        key for key in final_state
        if not torch.equal(final_state[key].cpu(), initial_state[key].cpu())
    ]
    illegal = [key for key in changed_tensors if not key.startswith("search_adapter.")]
    if illegal:
        raise RuntimeError(f"non-Search-adapter parameters changed: {illegal}")

    summary = {
        "method": "v13b drift-selective Search adapter update",
        **selection,
        "item_universe": len(item_ids),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "optimizer_steps": optimizer_steps,
        "wall_seconds": time.perf_counter() - start_time,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "changed_parameter_tensors": changed_tensors,
        "prefix_exactly_preserved": True,
        "rec_suffix_exactly_preserved": True,
        "raw_selected_change": float(
            np.any(search_codes[selected] != t0_search[selected], axis=1).mean()
        ),
        "raw_all_change": float(np.any(search_codes != t0_search, axis=1).mean()),
        "outside_private_shift": float(
            np.mean((private_np[outside_mask] - old_private[outside_mask]) ** 2)
        ),
        "search_collision": conditional_collision_stats(prefix_codes, search_codes),
        "search_usage": code_usage(search_codes, checkpoint["search_sizes"]),
        "history": logs,
    }
    write_json(args.output_dir / "update_stats.json", summary)
    torch.save(
        {
            "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "base_checkpoint": str(args.checkpoint),
            "summary": summary,
        },
        args.output_dir / "search_adapter_best.pt",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
