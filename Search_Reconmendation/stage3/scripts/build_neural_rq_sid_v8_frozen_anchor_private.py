#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.data import DataLoader, TensorDataset

from build_neural_rq_sid_v2 import (
    MLP,
    aggregate_rec_neighbors,
    aggregate_search_queries,
    code_usage,
    count_summary,
    field_coverage,
    load_item_rows,
    make_docs,
    parse_sizes,
    rq_assign_codes_np,
    write_json,
)
from build_neural_rq_sid_v2_text_encoder_fusion_plc import (
    SharedRQVAE,
    build_search_signal_embeddings,
    train_shared,
)
from build_neural_rq_sid_v3_aligned_delta import (
    conditional_collision_stats,
    encode_task_deltas,
    fixed_prefix,
    load_behavior_signal,
    load_precomputed_embedding,
    train_task_model,
    vector_summary,
)
from build_neural_rq_sid_v7_anchored_cross_view import mapping_from_codes
from refine_m7b_collision_group_sinkhorn import refine_last_suffix_layer


class FrozenPrefixBehaviorBridge(nn.Module):
    """Decode only behavior that is predictable from a frozen semantic prefix."""

    def __init__(self, latent_dim, hidden_dim, behavior_dim, dropout):
        super().__init__()
        self.decoder = MLP(latent_dim, hidden_dim, behavior_dim, dropout)

    def forward(self, q_shared):
        return self.decoder(q_shared)


def cosine_loss(prediction, target):
    return (1.0 - F.cosine_similarity(prediction, target, dim=1)).mean()


def restore_semantic_shared(args, common_dim, prefix_sizes, device):
    if not args.shared_checkpoint:
        return None
    checkpoint = torch.load(args.shared_checkpoint, map_location="cpu", weights_only=False)
    model = SharedRQVAE(
        input_dim=common_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        prefix_sizes=prefix_sizes,
        beta=args.vq_beta,
        dropout=args.dropout,
        balance_temperature=args.balance_temperature,
    ).to(device)
    source = checkpoint["model_state_dict"]
    shared_state = {
        key.removeprefix("shared."): value
        for key, value in source.items()
        if key.startswith("shared.")
    }
    missing, unexpected = model.load_state_dict(shared_state, strict=False)
    if missing or unexpected:
        raise ValueError(f"shared checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model


def train_behavior_bridge(args, q_shared, behavior_x, device):
    bridge = FrozenPrefixBehaviorBridge(
        q_shared.shape[1], args.bridge_hidden_dim, behavior_x.shape[1], args.dropout
    ).to(device)
    dataset = TensorDataset(torch.from_numpy(q_shared), torch.from_numpy(behavior_x))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 31),
    )
    optimizer = torch.optim.AdamW(
        bridge.parameters(), lr=args.bridge_lr, weight_decay=args.weight_decay
    )
    logs = []
    for epoch in range(1, args.bridge_epochs + 1):
        bridge.train()
        totals = Counter()
        steps = 0
        for q_b, behavior_b in loader:
            q_b = q_b.to(device)
            behavior_b = behavior_b.to(device)
            prediction = bridge(q_b)
            mse = F.mse_loss(prediction, behavior_b)
            cosine = cosine_loss(prediction, behavior_b)
            norm = prediction.pow(2).mean()
            loss = (
                args.bridge_mse_weight * mse
                + args.bridge_cos_weight * cosine
                + args.bridge_norm_weight * norm
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), args.grad_clip)
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["mse"] += float(mse.detach().cpu())
            totals["cosine"] += float(cosine.detach().cpu())
            totals["norm"] += float(norm.detach().cpu())
            steps += 1
        row = {"epoch": epoch, **{key: value / max(steps, 1) for key, value in totals.items()}}
        logs.append(row)
        if epoch == 1 or epoch == args.bridge_epochs or epoch % args.log_every == 0:
            print(json.dumps({"stage": "frozen_behavior_bridge", **row}), flush=True)
    return bridge, logs


@torch.no_grad()
def decode_bridge(bridge, q_shared, device, batch_size):
    values = []
    bridge.eval()
    for (q_b,) in DataLoader(
        TensorDataset(torch.from_numpy(q_shared)), batch_size=batch_size, shuffle=False
    ):
        values.append(bridge(q_b.to(device)).cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def bridge_quality(base, target):
    residual = target - base
    target_var = float(np.square(target - target.mean(axis=0, keepdims=True)).mean())
    residual_var = float(np.square(residual - residual.mean(axis=0, keepdims=True)).mean())
    cosine = F.cosine_similarity(torch.from_numpy(base), torch.from_numpy(target), dim=1)
    return {
        "mse": float(np.square(residual).mean()),
        "cosine_mean": float(cosine.mean()),
        "variance_explained": 1.0 - residual_var / max(target_var, 1e-12),
    }


def build(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_sizes = parse_sizes(args.prefix_codebook_sizes)
    search_sizes = parse_sizes(args.search_codebook_sizes)
    rec_sizes = parse_sizes(args.rec_codebook_sizes)

    item_ids, item_rows, has_metadata = load_item_rows(data_dir, args.item_ids, args.max_items)
    query_docs, query_counter = aggregate_search_queries(
        data_dir, item_ids, args.max_queries_per_item
    )
    rec_docs, _ = aggregate_rec_neighbors(
        data_dir, item_ids, args.neighbor_window, args.max_neighbors_per_item
    )
    make_docs(item_ids, item_rows, query_docs, rec_docs)
    common_x = load_precomputed_embedding(args.common_embedding, item_ids, "common")
    behavior_x, behavior_stats = load_behavior_signal(args.rec_behavior_embedding, item_ids)
    args.text_embedding_dim = int(common_x.shape[1])
    search_signal, search_signal_stats = build_search_signal_embeddings(
        args, item_ids, query_counter, device, Path(args.embedding_cache_dir)
    )

    shared = restore_semantic_shared(args, common_x.shape[1], prefix_sizes, device)
    if shared is None:
        shared, shared_logs = train_shared(args, common_x, prefix_sizes, device)
    else:
        shared_logs = [{"source": args.shared_checkpoint}]
    shared.eval()
    prefix_codes, q_shared, z_shared = fixed_prefix(shared, common_x, args.prefix_assignment)
    with torch.no_grad():
        semantic_base = (
            shared.decoder(torch.from_numpy(q_shared).to(device)).cpu().numpy().astype(np.float32)
        )

    bridge, bridge_logs = train_behavior_bridge(args, q_shared, behavior_x, device)
    behavior_base = decode_bridge(bridge, q_shared, device, args.batch_size)
    search_target = (common_x - semantic_base).astype(np.float32)
    rec_target = (behavior_x - behavior_base).astype(np.float32)

    model, task_logs = train_task_model(
        args,
        shared,
        common_x,
        search_signal,
        behavior_x,
        search_target,
        rec_target,
        semantic_base,
        behavior_base,
        common_x,
        behavior_x,
        True,
        common_x,
        behavior_x,
        prefix_codes,
        search_sizes,
        rec_sizes,
        device,
    )
    search_delta, rec_delta, search_gate, rec_gate = encode_task_deltas(
        model, common_x, search_signal, behavior_x, device, args.batch_size
    )
    raw_search = np.asarray(
        rq_assign_codes_np(search_delta, model.search_quantizer.codebooks, "argmin"),
        dtype=np.int64,
    )
    raw_rec = np.asarray(
        rq_assign_codes_np(rec_delta, model.rec_quantizer.codebooks, "argmin"),
        dtype=np.int64,
    )
    final_search, search_refine = refine_last_suffix_layer(
        prefix_codes,
        raw_search,
        search_delta,
        model.search_quantizer.codebooks,
        args.export_sinkhorn_epsilon,
        args.export_sinkhorn_iters,
        args.export_sinkhorn_rounds,
    )
    final_rec, rec_refine = refine_last_suffix_layer(
        prefix_codes,
        raw_rec,
        rec_delta,
        model.rec_quantizer.codebooks,
        args.export_sinkhorn_epsilon,
        args.export_sinkhorn_iters,
        args.export_sinkhorn_rounds,
    )
    final_search = np.asarray(final_search, dtype=np.int64)
    final_rec = np.asarray(final_rec, dtype=np.int64)
    write_json(
        output_dir / "sid_mapping.raw.json",
        mapping_from_codes(item_ids, prefix_codes, raw_search, raw_rec),
    )
    write_json(
        output_dir / "sid_mapping.v8_frozen_anchor_private.json",
        mapping_from_codes(item_ids, prefix_codes, final_search, final_rec),
    )
    write_json(output_dir / "shared_train_log.json", shared_logs)
    write_json(output_dir / "bridge_train_log.json", bridge_logs)
    write_json(output_dir / "task_train_log.json", task_logs)

    prefix_labels = ["_".join(map(str, row)) for row in prefix_codes]
    search_labels = ["_".join(map(str, row)) for row in final_search]
    rec_labels = ["_".join(map(str, row)) for row in final_rec]
    prefix_counter = Counter(prefix_labels)
    stats = {
        "method": "M7b-v8 Frozen Semantic Anchor + Shared/Private Behavior Bridge",
        "num_items": len(item_ids),
        "has_metadata": has_metadata,
        "field_coverage": field_coverage(item_rows),
        "codebook_sizes": {"prefix": prefix_sizes, "search": search_sizes, "rec": rec_sizes},
        "prefix": {
            "assignment": args.prefix_assignment,
            "unique_count": len(prefix_counter),
            "bucket_size": count_summary(list(prefix_counter.values())),
            "usage": code_usage(prefix_codes, prefix_sizes),
        },
        "bridge": bridge_quality(behavior_base, behavior_x),
        "views": {
            "common": vector_summary(common_x),
            "behavior": {**behavior_stats, **vector_summary(behavior_x)},
            "search_signal": {**search_signal_stats, **vector_summary(search_signal)},
            "semantic_base": vector_summary(semantic_base),
            "behavior_shared": vector_summary(behavior_base),
            "semantic_private": vector_summary(search_target),
            "behavior_private": vector_summary(rec_target),
        },
        "latent": {
            "z_shared": vector_summary(z_shared),
            "search_delta": vector_summary(search_delta),
            "rec_delta": vector_summary(rec_delta),
            "search_gate_mean": float(search_gate.mean()),
            "rec_gate_mean": float(rec_gate.mean()),
        },
        "raw_assignment": {
            "search_collision": conditional_collision_stats(prefix_codes, raw_search),
            "rec_collision": conditional_collision_stats(prefix_codes, raw_rec),
            "search_usage": code_usage(raw_search, search_sizes),
            "rec_usage": code_usage(raw_rec, rec_sizes),
        },
        "final_assignment": {
            "search_collision": conditional_collision_stats(prefix_codes, final_search),
            "rec_collision": conditional_collision_stats(prefix_codes, final_rec),
            "search_usage": code_usage(final_search, search_sizes),
            "rec_usage": code_usage(final_rec, rec_sizes),
            "search_changed_from_raw": float(np.any(final_search != raw_search, axis=1).mean()),
            "rec_changed_from_raw": float(np.any(final_rec != raw_rec, axis=1).mean()),
            "prefix_search_suffix_nmi": normalized_mutual_info_score(prefix_labels, search_labels),
            "prefix_rec_suffix_nmi": normalized_mutual_info_score(prefix_labels, rec_labels),
        },
        "refinement": {"search": search_refine, "rec": rec_refine},
        "shared_train_log_final": shared_logs[-1],
        "bridge_train_log_final": bridge_logs[-1],
        "task_train_log_final": task_logs[-1],
        "params": vars(args),
    }
    write_json(output_dir / "sid_stats.v8_frozen_anchor_private.json", stats)
    torch.save(
        {
            "task_model_state_dict": model.cpu().state_dict(),
            "bridge_state_dict": bridge.cpu().state_dict(),
            "params": vars(args),
            "prefix_sizes": prefix_sizes,
            "search_sizes": search_sizes,
            "rec_sizes": rec_sizes,
            "fixed_prefix_codes": prefix_codes,
        },
        output_dir / "frozen_anchor_private_sid_model.pt",
    )
    return stats


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--common-embedding", required=True)
    p.add_argument("--rec-behavior-embedding", required=True)
    p.add_argument("--shared-checkpoint", default="")
    p.add_argument("--item-ids", default="")
    p.add_argument("--max-items", type=int, default=20000)
    p.add_argument("--prefix-codebook-sizes", default="256,256")
    p.add_argument("--search-codebook-sizes", default="256,256")
    p.add_argument("--rec-codebook-sizes", default="256,256")
    p.add_argument("--prefix-assignment", choices=["argmin", "balanced"], default="argmin")
    p.add_argument("--max-queries-per-item", type=int, default=20)
    p.add_argument("--neighbor-window", type=int, default=2)
    p.add_argument("--max-neighbors-per-item", type=int, default=200)
    p.add_argument("--text-encoder-model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--text-encoder-prefix", default="")
    p.add_argument("--query-encoder-prefix", default="")
    p.add_argument("--text-encoder-pooling", choices=["mean", "cls"], default="cls")
    p.add_argument("--text-encoder-max-length", type=int, default=256)
    p.add_argument("--text-encoder-batch-size", type=int, default=256)
    p.add_argument("--embedding-cache-dir", required=True)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--augment-scale", type=float, default=0.25)
    p.add_argument("--gate-bias", type=float, default=-2.0)
    p.add_argument("--delta-max-ratio", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--shared-epochs", type=int, default=50)
    p.add_argument("--bridge-epochs", type=int, default=30)
    p.add_argument("--task-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bridge-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--vq-beta", type=float, default=0.25)
    p.add_argument("--balance-temperature", type=float, default=0.05)
    p.add_argument("--prefix-sk-epsilons", default="")
    p.add_argument("--sk-iters", type=int, default=100)
    p.add_argument("--init-codebooks", choices=["kmeans", "random"], default="kmeans")
    p.add_argument("--bridge-hidden-dim", type=int, default=128)
    p.add_argument("--bridge-mse-weight", type=float, default=5.0)
    p.add_argument("--bridge-cos-weight", type=float, default=0.5)
    p.add_argument("--bridge-norm-weight", type=float, default=0.001)
    p.add_argument("--task-mse-weight", type=float, default=10.0)
    p.add_argument("--task-cos-weight", type=float, default=0.1)
    p.add_argument("--full-mse-weight", type=float, default=1.0)
    p.add_argument("--commit-weight", type=float, default=0.05)
    p.add_argument("--balance-weight", type=float, default=0.02)
    p.add_argument("--delta-norm-weight", type=float, default=0.001)
    p.add_argument("--conditional-sinkhorn-weight", type=float, default=0.0)
    p.add_argument("--conditional-sinkhorn-epsilon", type=float, default=0.03)
    p.add_argument("--conditional-sinkhorn-iters", type=int, default=10)
    p.add_argument("--conditional-sinkhorn-temperature", type=float, default=0.1)
    p.add_argument("--conditional-every", type=int, default=1)
    p.add_argument("--prefix-group-size", type=int, default=16)
    p.add_argument("--inprefix-contrast-weight", type=float, default=0.1)
    p.add_argument("--inprefix-contrast-temperature", type=float, default=0.1)
    p.add_argument("--full-sid-collision-weight", type=float, default=0.2)
    p.add_argument("--full-sid-temperature", type=float, default=0.05)
    p.add_argument("--full-sid-topk", type=int, default=4)
    p.add_argument("--assignment-confidence-weight", type=float, default=0.05)
    p.add_argument("--variance-weight", type=float, default=2.0)
    p.add_argument("--variance-std-target", type=float, default=0.08)
    p.add_argument("--covariance-weight", type=float, default=2.0)
    p.add_argument("--task-gate-min", type=float, default=0.2)
    p.add_argument("--task-gate-floor-weight", type=float, default=0.5)
    p.add_argument("--topology-weight", type=float, default=2.0)
    p.add_argument("--adaptive-collision", action="store_true", default=True)
    p.add_argument("--collision-warmup-epochs", type=int, default=10)
    p.add_argument("--collision-target-duplicate", type=float, default=0.08)
    p.add_argument("--collision-controller-gain", type=float, default=0.15)
    p.add_argument("--export-sinkhorn-rounds", type=int, default=5)
    p.add_argument("--export-sinkhorn-epsilon", type=float, default=0.003)
    p.add_argument("--export-sinkhorn-iters", type=int, default=50)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--seed", type=int, default=2026)
    return p


def main():
    args = parser().parse_args()
    stats = build(args)
    print(
        json.dumps(
            {
                "method": stats["method"],
                "prefix": stats["prefix"],
                "bridge": stats["bridge"],
                "raw_assignment": stats["raw_assignment"],
                "final_assignment": stats["final_assignment"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
