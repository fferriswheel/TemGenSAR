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
    ResidualVectorQuantizer,
    aggregate_rec_neighbors,
    aggregate_search_queries,
    code_usage,
    count_summary,
    field_coverage,
    init_rq_codebooks,
    load_item_rows,
    make_docs,
    parse_sizes,
    rq_assign_codes_np,
    write_json,
)
from build_neural_rq_sid_v2_text_encoder_fusion_plc import (
    build_search_signal_embeddings,
    token_list,
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
from refine_m7b_collision_group_sinkhorn import refine_last_suffix_layer


class AnchoredCrossViewRQVAE(nn.Module):
    """A stable semantic anchor whose codes are supervised by both initial views."""

    def __init__(self, semantic_dim, behavior_dim, hidden_dim, latent_dim, prefix_sizes, args):
        super().__init__()
        self.encoder = MLP(semantic_dim, hidden_dim, latent_dim, args.dropout)
        self.quantizer = ResidualVectorQuantizer(
            prefix_sizes, latent_dim, args.vq_beta, args.balance_temperature
        )
        self.semantic_decoder = MLP(latent_dim, hidden_dim, semantic_dim, args.dropout)
        self.behavior_decoder = MLP(latent_dim, hidden_dim, behavior_dim, args.dropout)
        self.behavior_projector = nn.Linear(behavior_dim, latent_dim)

    def forward(self, semantic_x, behavior_x):
        z = self.encoder(semantic_x)
        q, indices, commit, balance = self.quantizer(z)
        return {
            "z": z,
            "q": q,
            "indices": indices,
            "semantic_recon": self.semantic_decoder(q),
            "behavior_recon": self.behavior_decoder(q),
            "behavior_latent": self.behavior_projector(behavior_x),
            "commit": commit,
            "balance": balance,
        }


def cosine_loss(prediction, target):
    return (1.0 - F.cosine_similarity(prediction, target, dim=1)).mean()


def alignment_loss(anchor, behavior, temperature):
    anchor = F.normalize(anchor, dim=1)
    behavior = F.normalize(behavior, dim=1)
    logits = anchor @ behavior.t() / temperature
    labels = torch.arange(len(anchor), device=anchor.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def train_shared(args, common_x, behavior_x, prefix_sizes, device):
    model = AnchoredCrossViewRQVAE(
        common_x.shape[1], behavior_x.shape[1], args.hidden_dim, args.latent_dim,
        prefix_sizes, args,
    ).to(device)
    if args.init_codebooks == "kmeans":
        with torch.no_grad():
            initial_z = model.encoder(torch.from_numpy(common_x).to(device)).cpu().numpy()
        init_rq_codebooks(model.quantizer, initial_z, args.seed)
    dataset = TensorDataset(torch.from_numpy(common_x), torch.from_numpy(behavior_x))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logs = []
    for epoch in range(1, args.shared_epochs + 1):
        model.train()
        totals = Counter()
        steps = 0
        for common_b, behavior_b in loader:
            common_b = common_b.to(device)
            behavior_b = behavior_b.to(device)
            out = model(common_b, behavior_b)
            semantic_mse = F.mse_loss(out["semantic_recon"], common_b)
            semantic_cos = cosine_loss(out["semantic_recon"], common_b)
            behavior_mse = F.mse_loss(out["behavior_recon"], behavior_b)
            behavior_cos = cosine_loss(out["behavior_recon"], behavior_b)
            align = alignment_loss(out["z"], out["behavior_latent"], args.alignment_temperature)
            loss = (
                args.shared_semantic_mse_weight * semantic_mse
                + args.shared_semantic_cos_weight * semantic_cos
                + args.shared_behavior_mse_weight * behavior_mse
                + args.shared_behavior_cos_weight * behavior_cos
                + args.shared_alignment_weight * align
                + args.commit_weight * out["commit"]
                + args.balance_weight * out["balance"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            for key, value in {
                "loss": loss, "semantic_mse": semantic_mse, "semantic_cos": semantic_cos,
                "behavior_mse": behavior_mse, "behavior_cos": behavior_cos,
                "alignment": align, "commit": out["commit"], "balance": out["balance"],
            }.items():
                totals[key] += float(value.detach().cpu())
            steps += 1
        row = {"epoch": epoch, **{key: value / max(steps, 1) for key, value in totals.items()}}
        logs.append(row)
        if epoch == 1 or epoch == args.shared_epochs or epoch % args.log_every == 0:
            print(json.dumps({"stage": "anchored_shared", **row}), flush=True)
    return model, logs


@torch.no_grad()
def shared_reconstructions(shared, q_shared, device, batch_size):
    semantic, behavior = [], []
    loader = DataLoader(torch.from_numpy(q_shared), batch_size=batch_size, shuffle=False)
    shared.eval()
    for q_b in loader:
        q_b = q_b.to(device)
        semantic.append(shared.semantic_decoder(q_b).cpu().numpy())
        behavior.append(shared.behavior_decoder(q_b).cpu().numpy())
    return np.concatenate(semantic).astype(np.float32), np.concatenate(behavior).astype(np.float32)


def mapping_from_codes(item_ids, prefix_codes, search_codes, rec_codes):
    mapping = {}
    for item, prefix, search, rec in zip(item_ids, prefix_codes, search_codes, rec_codes):
        p, s, r = token_list(prefix, "P"), token_list(search, "S"), token_list(rec, "R")
        mapping[item] = {
            "prefix": p, "search_suffix": s, "rec_suffix": r,
            "search_sid": p + s, "rec_sid": p + r,
            "prefix_codes": [int(x) for x in prefix],
            "search_suffix_codes": [int(x) for x in search],
            "rec_suffix_codes": [int(x) for x in rec],
        }
    return mapping


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
    query_docs, query_counter = aggregate_search_queries(data_dir, item_ids, args.max_queries_per_item)
    rec_docs, _ = aggregate_rec_neighbors(
        data_dir, item_ids, args.neighbor_window, args.max_neighbors_per_item
    )
    text_docs, _, _ = make_docs(item_ids, item_rows, query_docs, rec_docs)
    common_x = load_precomputed_embedding(args.common_embedding, item_ids, "common")
    behavior_x, behavior_stats = load_behavior_signal(args.rec_behavior_embedding, item_ids)
    args.text_embedding_dim = int(common_x.shape[1])
    cache_dir = Path(args.embedding_cache_dir)
    search_signal, search_signal_stats = build_search_signal_embeddings(
        args, item_ids, query_counter, device, cache_dir
    )

    shared, shared_logs = train_shared(args, common_x, behavior_x, prefix_sizes, device)
    shared.eval()
    prefix_codes, q_shared, z_shared = fixed_prefix(shared, common_x, args.prefix_assignment)
    semantic_base, behavior_base = shared_reconstructions(
        shared, q_shared, device, args.batch_size
    )
    search_target = (common_x - semantic_base).astype(np.float32)
    rec_target = (behavior_x - behavior_base).astype(np.float32)

    model, task_logs = train_task_model(
        args, shared, common_x, search_signal, behavior_x,
        search_target, rec_target, semantic_base, behavior_base,
        common_x, behavior_x, True, common_x, behavior_x,
        prefix_codes, search_sizes, rec_sizes, device,
    )
    search_delta, rec_delta, search_gate, rec_gate = encode_task_deltas(
        model, common_x, search_signal, behavior_x, device, args.batch_size
    )
    raw_search = np.asarray(
        rq_assign_codes_np(search_delta, model.search_quantizer.codebooks, "argmin"), dtype=np.int64
    )
    raw_rec = np.asarray(
        rq_assign_codes_np(rec_delta, model.rec_quantizer.codebooks, "argmin"), dtype=np.int64
    )
    final_search, search_refine = refine_last_suffix_layer(
        prefix_codes, raw_search, search_delta, model.search_quantizer.codebooks,
        args.export_sinkhorn_epsilon, args.export_sinkhorn_iters, args.export_sinkhorn_rounds,
    )
    final_rec, rec_refine = refine_last_suffix_layer(
        prefix_codes, raw_rec, rec_delta, model.rec_quantizer.codebooks,
        args.export_sinkhorn_epsilon, args.export_sinkhorn_iters, args.export_sinkhorn_rounds,
    )
    final_search = np.asarray(final_search, dtype=np.int64)
    final_rec = np.asarray(final_rec, dtype=np.int64)
    raw_mapping = mapping_from_codes(item_ids, prefix_codes, raw_search, raw_rec)
    final_mapping = mapping_from_codes(item_ids, prefix_codes, final_search, final_rec)
    write_json(output_dir / "sid_mapping.raw.json", raw_mapping)
    write_json(output_dir / "sid_mapping.v7_anchored_cross_view.json", final_mapping)
    write_json(output_dir / "shared_train_log.json", shared_logs)
    write_json(output_dir / "task_train_log.json", task_logs)

    prefix_labels = ["_".join(map(str, row)) for row in prefix_codes]
    search_labels = ["_".join(map(str, row)) for row in final_search]
    rec_labels = ["_".join(map(str, row)) for row in final_rec]
    prefix_counter = Counter(prefix_labels)
    stats = {
        "method": "M7b-v7 Anchored Cross-View Shared Prefix + Task-Local LoRA Suffix",
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
        "views": {
            "common": vector_summary(common_x),
            "behavior": {**behavior_stats, **vector_summary(behavior_x)},
            "search_signal": {**search_signal_stats, **vector_summary(search_signal)},
            "semantic_shared_recon": vector_summary(semantic_base),
            "behavior_shared_recon": vector_summary(behavior_base),
            "semantic_residual": vector_summary(search_target),
            "behavior_residual": vector_summary(rec_target),
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
        },
        "final_assignment": {
            "search_collision": conditional_collision_stats(prefix_codes, final_search),
            "rec_collision": conditional_collision_stats(prefix_codes, final_rec),
            "search_changed_from_raw": float(np.any(final_search != raw_search, axis=1).mean()),
            "rec_changed_from_raw": float(np.any(final_rec != raw_rec, axis=1).mean()),
            "prefix_search_suffix_nmi": normalized_mutual_info_score(prefix_labels, search_labels),
            "prefix_rec_suffix_nmi": normalized_mutual_info_score(prefix_labels, rec_labels),
        },
        "refinement": {"search": search_refine, "rec": rec_refine},
        "shared_train_log_final": shared_logs[-1],
        "task_train_log_final": task_logs[-1],
        "params": vars(args),
    }
    write_json(output_dir / "sid_stats.v7_anchored_cross_view.json", stats)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(), "params": vars(args),
            "prefix_sizes": prefix_sizes, "search_sizes": search_sizes, "rec_sizes": rec_sizes,
            "fixed_prefix_codes": prefix_codes,
        },
        output_dir / "anchored_cross_view_sid_model.pt",
    )
    return stats


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--common-embedding", required=True)
    p.add_argument("--rec-behavior-embedding", required=True)
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
    p.add_argument("--task-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--vq-beta", type=float, default=0.25)
    p.add_argument("--balance-temperature", type=float, default=0.05)
    p.add_argument("--init-codebooks", choices=["kmeans", "random"], default="kmeans")
    p.add_argument("--shared-semantic-mse-weight", type=float, default=10.0)
    p.add_argument("--shared-semantic-cos-weight", type=float, default=1.0)
    p.add_argument("--shared-behavior-mse-weight", type=float, default=5.0)
    p.add_argument("--shared-behavior-cos-weight", type=float, default=0.5)
    p.add_argument("--shared-alignment-weight", type=float, default=0.1)
    p.add_argument("--alignment-temperature", type=float, default=0.1)
    p.add_argument("--task-mse-weight", type=float, default=10.0)
    p.add_argument("--task-cos-weight", type=float, default=0.1)
    p.add_argument("--full-mse-weight", type=float, default=1.0)
    p.add_argument("--commit-weight", type=float, default=0.05)
    p.add_argument("--balance-weight", type=float, default=0.02)
    p.add_argument("--delta-norm-weight", type=float, default=0.001)
    p.add_argument("--conditional-sinkhorn-weight", type=float, default=0.02)
    p.add_argument("--conditional-sinkhorn-epsilon", type=float, default=0.03)
    p.add_argument("--conditional-sinkhorn-iters", type=int, default=10)
    p.add_argument("--conditional-sinkhorn-temperature", type=float, default=0.1)
    p.add_argument("--conditional-every", type=int, default=1)
    p.add_argument("--prefix-group-size", type=int, default=16)
    p.add_argument("--inprefix-contrast-weight", type=float, default=0.5)
    p.add_argument("--inprefix-contrast-temperature", type=float, default=0.1)
    p.add_argument("--full-sid-collision-weight", type=float, default=1.0)
    p.add_argument("--full-sid-temperature", type=float, default=0.05)
    p.add_argument("--full-sid-topk", type=int, default=4)
    p.add_argument("--assignment-confidence-weight", type=float, default=0.2)
    p.add_argument("--variance-weight", type=float, default=10.0)
    p.add_argument("--variance-std-target", type=float, default=0.08)
    p.add_argument("--covariance-weight", type=float, default=20.0)
    p.add_argument("--task-gate-min", type=float, default=0.2)
    p.add_argument("--task-gate-floor-weight", type=float, default=0.5)
    p.add_argument("--topology-weight", type=float, default=2.0)
    p.add_argument("--adaptive-collision", action="store_true", default=True)
    p.add_argument("--collision-warmup-epochs", type=int, default=10)
    p.add_argument("--collision-target-duplicate", type=float, default=0.1)
    p.add_argument("--collision-controller-gain", type=float, default=2.0)
    p.add_argument("--export-sinkhorn-rounds", type=int, default=20)
    p.add_argument("--export-sinkhorn-epsilon", type=float, default=0.003)
    p.add_argument("--export-sinkhorn-iters", type=int, default=50)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    return p


def main():
    args = parser().parse_args()
    stats = build(args)
    print(json.dumps({
        "method": stats["method"], "prefix": stats["prefix"],
        "raw_assignment": stats["raw_assignment"],
        "final_assignment": stats["final_assignment"],
    }, indent=2))


if __name__ == "__main__":
    main()
