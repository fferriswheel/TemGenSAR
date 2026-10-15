#!/usr/bin/env python3
"""Update the v13b Rec adapter on drift-selected items plus bounded replay."""

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

from build_neural_rq_sid_v2 import code_usage, load_item_rows, write_json
from build_neural_rq_sid_v3_aligned_delta import (
    PrefixBatchSampler,
    conditional_collision_stats,
    inprefix_topology_loss,
    load_behavior_signal,
    load_precomputed_embedding,
)
from build_neural_rq_sid_v9_direct_private import DirectPrivateTokenizer, cosine_loss
from build_neural_rq_sid_v2_text_encoder_fusion_plc import l2_normalize_np
from train_multiview_anchor_prefix_v13 import MultiViewAnchorRQVAE
from update_v13b_rec_adapter import assign_rec, mapping_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--t0-mapping", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--common-embedding", required=True)
    parser.add_argument("--updated-rec-embedding", required=True)
    parser.add_argument("--t0-rec-embedding", required=True)
    parser.add_argument("--anchor-lookup", type=Path, required=True)
    parser.add_argument("--drift-csv", type=Path, required=True)
    parser.add_argument("--drift-left", default="C8")
    parser.add_argument("--drift-right", default="W1")
    parser.add_argument("--replay-ratio", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--topology-weight", type=float, default=0.5)
    parser.add_argument("--commit-weight", type=float, default=0.05)
    parser.add_argument("--stability-weight", type=float, default=0.1)
    parser.add_argument("--outside-stability-weight", type=float, default=0.1)
    parser.add_argument("--transition-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument(
        "--tri-state-objective",
        action="store_true",
        help=(
            "Train high-drift items toward the new behavior view, use low-drift replay "
            "only as an old-view stability anchor, and omit the uncertain middle group."
        ),
    )
    parser.add_argument(
        "--train-rec-codebook",
        action="store_true",
        help="Allow the Rec suffix codebooks to refresh together with the adapter.",
    )
    parser.add_argument(
        "--train-rec-decoder",
        action="store_true",
        help="Allow the Rec reconstruction decoder to refresh together with the adapter.",
    )
    parser.add_argument("--deterministic-frozen-modules", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def select_training_items(
    drift_path: Path,
    item_ids: list[str],
    left: str,
    right: str,
    replay_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    drift = pd.read_csv(drift_path, dtype={"item_id": str})
    drift = drift[(drift["left"] == left) & (drift["right"] == right)].copy()
    drift["item_id"] = drift["item_id"].astype(str)
    selected_ids = set(drift.loc[drift["selected_for_update"], "item_id"])
    selected = np.asarray([idx for idx, item in enumerate(item_ids) if item in selected_ids])
    if not len(selected):
        raise ValueError("Drift selector produced no training items")

    replay_count = int(round(len(selected) * replay_ratio))
    candidates = drift[
        drift["eligible"] & ~drift["item_id"].isin(selected_ids)
    ].copy()
    rng = np.random.default_rng(seed)
    candidates["tie_break"] = rng.random(len(candidates))
    candidates = candidates.sort_values(
        ["drift_rank_score", "event_support", "tie_break", "item_id"],
        ascending=[True, False, True, True],
    )
    replay_ids = set(candidates.head(replay_count)["item_id"])
    if len(replay_ids) < replay_count:
        raise ValueError(
            f"Only {len(replay_ids)} eligible replay items for requested {replay_count}"
        )
    replay = np.asarray([idx for idx, item in enumerate(item_ids) if item in replay_ids])
    train = np.concatenate([selected, replay])
    middle_count = int(
        (drift["eligible"] & ~drift["item_id"].isin(selected_ids | replay_ids)).sum()
    )
    return selected, train, {
        "selected_items": len(selected),
        "replay_items": len(replay),
        "train_items": len(train),
        "replay_ratio": replay_ratio,
        "selected_item_ids": sorted(selected_ids),
        "replay_item_ids": sorted(replay_ids),
        "uncertain_middle_items": middle_count,
    }


def build_transition_targets(
    path: Path,
    item_ids: list[str],
    behavior: np.ndarray,
) -> tuple[np.ndarray, dict[str, int | float]]:
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    sums = np.zeros_like(behavior, dtype=np.float32)
    counts = np.zeros(len(item_ids), dtype=np.int64)
    last_by_user: dict[str, int] = {}
    transitions = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            current = item_to_index.get(str(row["item_id"]))
            if current is None:
                continue
            user = str(row["user_id"])
            previous = last_by_user.get(user)
            if previous is not None and previous != current:
                sums[current] += behavior[previous]
                sums[previous] += behavior[current]
                counts[current] += 1
                counts[previous] += 1
                transitions += 1
            last_by_user[user] = current
    target = behavior.copy()
    covered = counts > 0
    target[covered] = sums[covered] / counts[covered, None]
    target = l2_normalize_np(target).astype(np.float32)
    return target, {
        "transition_edges": transitions,
        "covered_items": int(covered.sum()),
        "coverage_fraction": float(covered.mean()),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    params = argparse.Namespace(**checkpoint["params"])
    item_ids, _, _ = load_item_rows(args.data_dir, "", params.max_items)
    common_x = load_precomputed_embedding(args.common_embedding, item_ids, "common")
    behavior_x, behavior_stats = load_behavior_signal(args.updated_rec_embedding, item_ids)
    t0_behavior_x, _ = load_behavior_signal(args.t0_rec_embedding, item_ids)
    lookup = np.load(args.anchor_lookup)
    if [str(value) for value in lookup["item_ids"].tolist()] != item_ids:
        raise ValueError("anchor lookup item order mismatch")
    prefix_codes = lookup["prefix_codes"].astype(np.int64)
    z_shared = lookup["z_anchor"].astype(np.float32)
    q_shared = lookup["q_anchor"].astype(np.float32)
    behavior_base = l2_normalize_np(lookup["behavior_base"].astype(np.float32)).astype(np.float32)

    anchor_ckpt = torch.load(params.anchor_checkpoint, map_location="cpu", weights_only=False)
    anchor_args = argparse.Namespace(**anchor_ckpt["params"])
    search_dim = int(getattr(params, "text_embedding_dim", 768))
    shared = MultiViewAnchorRQVAE(
        common_x.shape[1], behavior_x.shape[1], search_dim,
        checkpoint["prefix_sizes"], anchor_args,
    )
    model = DirectPrivateTokenizer(
        shared, common_x.shape[1], search_dim, behavior_x.shape[1],
        checkpoint["search_sizes"], checkpoint["rec_sizes"], params,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    initial_state = copy.deepcopy(model.state_dict())
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.rec_adapter.parameters():
        parameter.requires_grad = True
    model.shared.eval()
    model.search_adapter.eval()
    model.rec_quantizer.eval()
    model.rec_decoder.eval()

    t0_mapping = json.loads(args.t0_mapping.read_text(encoding="utf-8"))
    t0_search, t0_rec = mapping_arrays(t0_mapping, item_ids)
    with torch.no_grad():
        old_private, _ = assign_rec(model, z_shared, t0_behavior_x, args.batch_size, device)

    selected, train_indices, selection = select_training_items(
        args.drift_csv, item_ids, args.drift_left, args.drift_right,
        args.replay_ratio, args.seed,
    )
    outside_mask = np.ones(len(item_ids), dtype=bool)
    outside_mask[selected] = False
    transition_target, transition_stats = build_transition_targets(
        args.data_dir / "rec_interactions_chrono.jsonl", item_ids, behavior_x
    )
    high_drift = np.zeros(len(train_indices), dtype=np.float32)
    high_drift[: len(selected)] = 1.0
    reconstruction_target = behavior_x[train_indices].copy()
    if args.tri_state_objective:
        reconstruction_target[len(selected) :] = t0_behavior_x[
            train_indices[len(selected) :]
        ]
    dataset = TensorDataset(
        torch.from_numpy(behavior_x[train_indices]),
        torch.from_numpy(reconstruction_target),
        torch.from_numpy(behavior_base[train_indices]),
        torch.from_numpy(prefix_codes[train_indices]),
        torch.from_numpy(z_shared[train_indices]),
        torch.from_numpy(q_shared[train_indices]),
        torch.from_numpy(old_private[train_indices]),
        torch.from_numpy(transition_target[train_indices]),
        torch.from_numpy(high_drift),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=PrefixBatchSampler(
            prefix_codes[train_indices], args.batch_size, 16, args.seed + 17
        ),
        num_workers=0,
    )
    trainable = list(model.rec_adapter.parameters())
    if args.train_rec_codebook:
        for parameter in model.rec_quantizer.parameters():
            parameter.requires_grad = True
        trainable.extend(model.rec_quantizer.parameters())
    if args.train_rec_decoder:
        for parameter in model.rec_decoder.parameters():
            parameter.requires_grad = True
        trainable.extend(model.rec_decoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    best_state, best_score, best_epoch, stale = None, float("inf"), None, 0
    logs = []
    optimizer_steps = 0
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.shared.eval()
        if args.deterministic_frozen_modules and not args.train_rec_codebook:
            model.rec_quantizer.eval()
        if args.deterministic_frozen_modules and not args.train_rec_decoder:
            model.rec_decoder.eval()
        totals, steps = Counter(), 0
        for (
            behavior_b,
            target_b,
            base_b,
            prefix_b,
            z_b,
            q_b,
            old_private_b,
            transition_b,
            high_drift_b,
        ) in loader:
            behavior_b, target_b, base_b = (
                behavior_b.to(device), target_b.to(device), base_b.to(device)
            )
            prefix_b, z_b, q_b = prefix_b.to(device), z_b.to(device), q_b.to(device)
            old_private_b = old_private_b.to(device)
            transition_b = transition_b.to(device)
            high_drift_b = high_drift_b.to(device)
            private = model.rec_adapter(z_b, behavior_b)
            q_rec, _, commit, _ = model.rec_quantizer(private)
            full_recon = F.normalize(base_b + model.rec_decoder(q_rec, q_b), dim=1)
            reconstruction = (1.0 - F.cosine_similarity(full_recon, target_b, dim=1)).mean()
            topology = inprefix_topology_loss(prefix_b, q_rec, target_b)
            stability_per_item = (private - old_private_b).pow(2).mean(dim=1)
            if args.tri_state_objective:
                low_drift_b = 1.0 - high_drift_b
                stability = (
                    stability_per_item * low_drift_b
                ).sum() / low_drift_b.sum().clamp_min(1.0)
            else:
                stability = stability_per_item.mean()
            transition_contrastive = full_recon.new_tensor(0.0)
            if args.transition_contrastive_weight > 0 and len(full_recon) > 1:
                active = high_drift_b.bool() if args.tri_state_objective else torch.ones_like(
                    high_drift_b, dtype=torch.bool
                )
                if int(active.sum()) > 1:
                    prediction = F.normalize(full_recon[active], dim=1)
                    target = F.normalize(transition_b[active], dim=1)
                    logits = prediction @ target.t() / args.contrastive_temperature
                    labels = torch.arange(len(logits), device=logits.device)
                    transition_contrastive = 0.5 * (
                        F.cross_entropy(logits, labels)
                        + F.cross_entropy(logits.t(), labels)
                    )
            loss = reconstruction + args.topology_weight * topology
            loss = loss + args.commit_weight * commit + args.stability_weight * stability
            loss = loss + args.transition_contrastive_weight * transition_contrastive
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer_steps += 1
            for key, value in {
                "loss": loss, "reconstruction": reconstruction,
                "topology": topology, "commit": commit, "stability": stability,
                "transition_contrastive": transition_contrastive,
            }.items():
                totals[key] += float(value.detach().cpu())
            steps += 1

        private_np, rec_codes = assign_rec(model, z_shared, behavior_x, args.batch_size, device)
        collision = conditional_collision_stats(prefix_codes, rec_codes)
        outside_shift = float(np.mean((private_np[outside_mask] - old_private[outside_mask]) ** 2))
        selected_change = float(np.any(rec_codes[selected] != t0_rec[selected], axis=1).mean())
        row = {
            "epoch": epoch,
            **{key: value / max(steps, 1) for key, value in totals.items()},
            "outside_private_shift": outside_shift,
            "selected_raw_change": selected_change,
            "rec_duplicate": collision["duplicate_fraction"],
        }
        score = row["reconstruction"]
        score += args.transition_contrastive_weight * row["transition_contrastive"]
        score += args.outside_stability_weight * outside_shift
        score += 0.5 * max(row["rec_duplicate"] - 0.09, 0.0)
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

    model.load_state_dict(best_state)
    private_np, rec_codes = assign_rec(model, z_shared, behavior_x, args.batch_size, device)
    final_state = model.state_dict()
    changed_tensors = [
        key for key in final_state
        if not torch.equal(final_state[key].cpu(), initial_state[key].cpu())
    ]
    allowed_prefixes = ["rec_adapter."]
    if args.train_rec_codebook:
        allowed_prefixes.append("rec_quantizer.")
    if args.train_rec_decoder:
        allowed_prefixes.append("rec_decoder.")
    illegal = [
        key for key in changed_tensors if not any(key.startswith(prefix) for prefix in allowed_prefixes)
    ]
    if illegal:
        raise RuntimeError(f"non-Rec-adapter parameters changed: {illegal}")
    if not np.array_equal(checkpoint["fixed_prefix_codes"], prefix_codes):
        raise RuntimeError("prefix codes changed")

    summary = {
        "method": (
            "v13b tri-state task-local Rec tokenizer refresh"
            if args.tri_state_objective
            else "v13b drift-selective Rec adapter update"
        ),
        **selection,
        "item_universe": len(item_ids),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "optimizer_steps": optimizer_steps,
        "wall_seconds": time.perf_counter() - start_time,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "tri_state_objective": args.tri_state_objective,
        "train_rec_codebook": args.train_rec_codebook,
        "train_rec_decoder": args.train_rec_decoder,
        "changed_parameter_tensors": changed_tensors,
        "prefix_exactly_preserved": True,
        "search_suffix_exactly_preserved": True,
        "raw_selected_change": float(
            np.any(rec_codes[selected] != t0_rec[selected], axis=1).mean()
        ),
        "raw_all_change": float(np.any(rec_codes != t0_rec, axis=1).mean()),
        "outside_private_shift": float(
            np.mean((private_np[outside_mask] - old_private[outside_mask]) ** 2)
        ),
        "rec_collision": conditional_collision_stats(prefix_codes, rec_codes),
        "rec_usage": code_usage(rec_codes, checkpoint["rec_sizes"]),
        "updated_behavior": behavior_stats,
        "transition_target": transition_stats,
        "transition_contrastive_weight": args.transition_contrastive_weight,
        "contrastive_temperature": args.contrastive_temperature,
        "deterministic_frozen_modules": args.deterministic_frozen_modules,
        "history": logs,
    }
    write_json(args.output_dir / "update_stats.json", summary)
    torch.save(
        {
            "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "base_checkpoint": str(args.checkpoint),
            "summary": summary,
        },
        args.output_dir / "rec_adapter_best.pt",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
