#!/usr/bin/env python3
import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from build_neural_rq_sid_v2 import (
    MLP,
    ResidualVectorQuantizer,
    aggregate_search_queries,
    code_usage,
    count_summary,
    init_rq_codebooks,
    load_item_rows,
    parse_sizes,
    rq_assign_codes_np,
    write_json,
)
from build_neural_rq_sid_v2_text_encoder_fusion_plc import (
    build_search_signal_embeddings,
)
from build_neural_rq_sid_v3_aligned_delta import (
    load_behavior_signal,
    load_precomputed_embedding,
    vector_summary,
)


def cosine_loss(prediction, target):
    return (1.0 - F.cosine_similarity(prediction, target, dim=1)).mean()


class MultiViewAnchorRQVAE(nn.Module):
    def __init__(self, common_dim, behavior_dim, search_dim, prefix_sizes, args):
        super().__init__()
        self.mode = args.anchor_mode
        self.consensus_alpha_max = float(
            getattr(args, "consensus_alpha_max", 0.5)
        )
        self.common_encoder = MLP(
            common_dim, args.hidden_dim, args.latent_dim, args.dropout
        )
        self.behavior_encoder = MLP(
            behavior_dim, args.hidden_dim, args.latent_dim, args.dropout
        )
        self.search_encoder = (
            MLP(search_dim, args.hidden_dim, args.latent_dim, args.dropout)
            if self.mode in {"full", "consensus"}
            else None
        )
        self.num_views = 3 if self.mode in {"full", "consensus"} else 2
        if self.mode == "consensus":
            self.consensus_gate = nn.Sequential(
                nn.Linear(args.latent_dim * 4, args.hidden_dim),
                nn.GELU(),
                nn.Linear(args.hidden_dim, 1),
            )
            self.gate = None
        else:
            self.consensus_gate = None
            self.gate = nn.Sequential(
                nn.Linear(args.latent_dim * self.num_views, args.hidden_dim),
                nn.GELU(),
                nn.Linear(args.hidden_dim, self.num_views),
            )
        self.anchor_norm = nn.LayerNorm(args.latent_dim)
        self.quantizer = ResidualVectorQuantizer(
            prefix_sizes,
            args.latent_dim,
            args.vq_beta,
            args.balance_temperature,
        )
        self.common_decoder = MLP(
            args.latent_dim, args.hidden_dim, common_dim, args.dropout
        )
        self.behavior_decoder = MLP(
            args.latent_dim, args.hidden_dim, behavior_dim, args.dropout
        )
        self.search_decoder = (
            MLP(args.latent_dim, args.hidden_dim, search_dim, args.dropout)
            if self.mode in {"full", "consensus"}
            else None
        )

    def encode(
        self,
        common_x,
        behavior_x,
        search_x,
        behavior_mask=None,
        search_mask=None,
        return_aux=False,
    ):
        common = self.common_encoder(common_x)
        behavior = self.behavior_encoder(behavior_x)
        search = self.search_encoder(search_x) if self.search_encoder is not None else None

        if self.mode == "consensus":
            batch_size = common.size(0)
            if behavior_mask is None:
                behavior_mask = common.new_ones(batch_size, 1)
            if search_mask is None:
                search_mask = common.new_ones(batch_size, 1)
            behavior_visible = behavior * behavior_mask
            search_visible = search * search_mask
            behavior_unit = F.normalize(behavior_visible, dim=1)
            search_unit = F.normalize(search_visible, dim=1)
            consensus = 0.5 * (behavior_unit + search_unit)
            gate_features = torch.cat(
                [common, behavior_visible, search_visible, torch.abs(behavior_visible - search_visible)],
                dim=1,
            )
            both_visible = behavior_mask * search_mask
            alpha = self.consensus_alpha_max * torch.sigmoid(
                self.consensus_gate(gate_features)
            ) * both_visible
            anchor = self.anchor_norm(common + alpha * consensus)
            denominator = 1.0 + alpha
            gate = torch.cat(
                [1.0 / denominator, 0.5 * alpha / denominator, 0.5 * alpha / denominator],
                dim=1,
            )
            aux = {
                "common_shared": common,
                "behavior_shared": behavior,
                "search_shared": search,
                "consensus": consensus,
                "consensus_alpha": alpha,
            }
        else:
            views = [common, behavior]
            if search is not None:
                views.append(search)
            gate = F.softmax(self.gate(torch.cat(views, dim=1)), dim=1)
            stacked = torch.stack(views, dim=1)
            anchor = self.anchor_norm((gate.unsqueeze(-1) * stacked).sum(dim=1))
            aux = {}
        if return_aux:
            return anchor, gate, aux
        return anchor, gate

    def forward(
        self,
        common_x,
        behavior_x,
        search_x,
        quantize=True,
        behavior_mask=None,
        search_mask=None,
    ):
        z_anchor, gate, aux = self.encode(
            common_x,
            behavior_x,
            search_x,
            behavior_mask=behavior_mask,
            search_mask=search_mask,
            return_aux=True,
        )
        if quantize:
            q_anchor, indices, commit, balance = self.quantizer(z_anchor)
        else:
            q_anchor = z_anchor
            indices = []
            commit = balance = z_anchor.new_tensor(0.0)
        output = {
            "z_anchor": z_anchor,
            "q_anchor": q_anchor,
            "indices": indices,
            "commit": commit,
            "balance": balance,
            "gate": gate,
            "common_recon": self.common_decoder(q_anchor),
            "behavior_recon": self.behavior_decoder(q_anchor),
            **aux,
        }
        if self.search_decoder is not None:
            output["search_recon"] = self.search_decoder(q_anchor)
        return output


@torch.no_grad()
def encode_all(model, common_x, behavior_x, search_x, device, batch_size):
    z_values, q_values, gate_values = [], [], []
    common_recon, behavior_recon, search_recon, consensus_alpha = [], [], [], []
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(common_x),
            torch.from_numpy(behavior_x),
            torch.from_numpy(search_x),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    for common_b, behavior_b, search_b in loader:
        output = model(
            common_b.to(device), behavior_b.to(device), search_b.to(device), quantize=True
        )
        z_values.append(output["z_anchor"].cpu().numpy())
        q_values.append(output["q_anchor"].cpu().numpy())
        gate_values.append(output["gate"].cpu().numpy())
        common_recon.append(output["common_recon"].cpu().numpy())
        behavior_recon.append(output["behavior_recon"].cpu().numpy())
        if "search_recon" in output:
            search_recon.append(output["search_recon"].cpu().numpy())
        if "consensus_alpha" in output:
            consensus_alpha.append(output["consensus_alpha"].cpu().numpy())
    return {
        "z_anchor": np.concatenate(z_values).astype(np.float32),
        "q_anchor": np.concatenate(q_values).astype(np.float32),
        "gate": np.concatenate(gate_values).astype(np.float32),
        "common_recon": np.concatenate(common_recon).astype(np.float32),
        "behavior_recon": np.concatenate(behavior_recon).astype(np.float32),
        "search_recon": np.concatenate(search_recon).astype(np.float32)
        if search_recon
        else None,
        "consensus_alpha": np.concatenate(consensus_alpha).astype(np.float32)
        if consensus_alpha
        else None,
    }


def prefix_stats(codes, sizes):
    labels = [tuple(int(value) for value in row) for row in codes]
    counts = Counter(labels)
    return {
        "unique_count": len(counts),
        "duplicate_fraction": (len(labels) - len(counts)) / max(len(labels), 1),
        "bucket_size": count_summary(list(counts.values())),
        "usage": code_usage(codes, sizes),
    }


def train_anchor(args, common_x, behavior_x, search_x, prefix_sizes, device):
    model = MultiViewAnchorRQVAE(
        common_x.shape[1], behavior_x.shape[1], search_x.shape[1], prefix_sizes, args
    ).to(device)
    dataset = TensorDataset(
        torch.from_numpy(common_x),
        torch.from_numpy(behavior_x),
        torch.from_numpy(search_x),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.anchor_lr, weight_decay=args.weight_decay
    )
    logs = []
    best_score = float("inf")
    best_epoch = None
    best_state = None
    for epoch in range(1, args.anchor_epochs + 1):
        if epoch == args.anchor_warmup_epochs + 1:
            with torch.no_grad():
                latent = []
                model.eval()
                for common_b, behavior_b, search_b in DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=False
                ):
                    z_anchor, _ = model.encode(
                        common_b.to(device), behavior_b.to(device), search_b.to(device)
                    )
                    latent.append(z_anchor.cpu().numpy())
            init_rq_codebooks(
                model.quantizer, np.concatenate(latent).astype(np.float32), args.seed
            )
        quantize = epoch > args.anchor_warmup_epochs
        model.train()
        totals = Counter()
        steps = 0
        for common_b, behavior_b, search_b in loader:
            common_b = common_b.to(device)
            behavior_b = behavior_b.to(device)
            search_b = search_b.to(device)
            output = model(common_b, behavior_b, search_b, quantize=quantize)
            common_recon = cosine_loss(output["common_recon"], common_b)
            behavior_recon = cosine_loss(output["behavior_recon"], behavior_b)
            search_recon = (
                cosine_loss(output["search_recon"], search_b)
                if "search_recon" in output
                else common_recon.new_tensor(0.0)
            )
            agreement = common_recon.new_tensor(0.0)
            consensus_common = common_recon.new_tensor(0.0)
            anchor_consistency = common_recon.new_tensor(0.0)
            dropout_reconstruction = common_recon.new_tensor(0.0)
            if args.anchor_mode == "consensus":
                aux_weight = args.consensus_aux_recon_weight
                reconstruction = (
                    common_recon + aux_weight * (behavior_recon + search_recon)
                ) / (1.0 + 2.0 * aux_weight)
                agreement = cosine_loss(
                    output["behavior_shared"], output["search_shared"]
                )
                consensus_common = cosine_loss(
                    output["consensus"], output["common_shared"]
                )
                if args.task_view_dropout > 0:
                    behavior_mask = (
                        torch.rand(len(common_b), 1, device=device)
                        >= args.task_view_dropout
                    ).to(common_b.dtype)
                    search_mask = (
                        torch.rand(len(common_b), 1, device=device)
                        >= args.task_view_dropout
                    ).to(common_b.dtype)
                    dropped = model(
                        common_b,
                        behavior_b,
                        search_b,
                        quantize=quantize,
                        behavior_mask=behavior_mask,
                        search_mask=search_mask,
                    )
                    dropped_common = cosine_loss(dropped["common_recon"], common_b)
                    dropped_behavior = cosine_loss(
                        dropped["behavior_recon"], behavior_b
                    )
                    dropped_search = cosine_loss(dropped["search_recon"], search_b)
                    dropout_reconstruction = (
                        dropped_common
                        + aux_weight * (dropped_behavior + dropped_search)
                    ) / (1.0 + 2.0 * aux_weight)
                    anchor_consistency = cosine_loss(
                        dropped["z_anchor"], output["z_anchor"].detach()
                    )
                    reconstruction = 0.5 * (
                        reconstruction + dropout_reconstruction
                    )
                gate_diversity = common_recon.new_tensor(0.0)
            else:
                view_count = 3 if args.anchor_mode == "full" else 2
                reconstruction = (common_recon + behavior_recon + search_recon) / view_count
                gate_entropy = -(
                    output["gate"] * (output["gate"] + 1e-8).log()
                ).sum(dim=1).mean()
                gate_diversity = common_recon.new_tensor(np.log(view_count)) - gate_entropy
            loss = (
                reconstruction
                + args.commit_weight * output["commit"]
                + args.balance_weight * output["balance"]
                + args.gate_diversity_weight * gate_diversity
                + args.consensus_agreement_weight * agreement
                + args.consensus_common_weight * consensus_common
                + args.consensus_consistency_weight * anchor_consistency
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            for key, value in {
                "loss": loss,
                "reconstruction": reconstruction,
                "common_recon": common_recon,
                "behavior_recon": behavior_recon,
                "search_recon": search_recon,
                "commit": output["commit"],
                "balance": output["balance"],
                "gate_diversity": gate_diversity,
                "agreement": agreement,
                "consensus_common": consensus_common,
                "anchor_consistency": anchor_consistency,
                "dropout_reconstruction": dropout_reconstruction,
            }.items():
                totals[key] += float(value.detach().cpu())
            steps += 1
        row = {"epoch": epoch, **{key: value / steps for key, value in totals.items()}}
        row["phase"] = "quantized" if quantize else "continuous_warmup"
        if quantize:
            encoded = encode_all(
                model, common_x, behavior_x, search_x, device, args.batch_size
            )
            codes = np.asarray(
                rq_assign_codes_np(
                    encoded["z_anchor"], model.quantizer.codebooks, "argmin"
                ),
                dtype=np.int64,
            )
            stats = prefix_stats(codes, prefix_sizes)
            underuse = sum(
                max(args.minimum_code_usage - level["used_codes"], 0)
                / max(args.minimum_code_usage, 1)
                for level in stats["usage"]
            )
            score = row["reconstruction"] + args.selection_underuse_weight * underuse
            if args.anchor_mode == "consensus":
                score += args.consensus_agreement_weight * row["agreement"]
                score += args.consensus_common_weight * row["consensus_common"]
                score += args.consensus_consistency_weight * row["anchor_consistency"]
            row.update(
                {
                    "prefix_unique": stats["unique_count"],
                    "prefix_duplicate": stats["duplicate_fraction"],
                    "prefix_max_bucket": stats["bucket_size"]["max"],
                    "prefix_usage": [level["used_codes"] for level in stats["usage"]],
                    "selection_score": score,
                }
            )
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )
        logs.append(row)
        if epoch == 1 or epoch == args.anchor_epochs or epoch % args.log_every == 0:
            print(json.dumps({"stage": "anchor_prefix", **row}), flush=True)
    if best_state is None:
        raise RuntimeError("anchor prefix training produced no quantized checkpoint")
    model.load_state_dict(best_state)
    return model, logs, {"selected_best_epoch": best_epoch, "selected_best_score": best_score}


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
    item_ids, _, _ = load_item_rows(data_dir, args.item_ids, args.max_items)
    _, query_counter = aggregate_search_queries(
        data_dir, item_ids, args.max_queries_per_item
    )
    common_x = load_precomputed_embedding(args.common_embedding, item_ids, "common")
    behavior_x, behavior_meta = load_behavior_signal(
        args.rec_behavior_embedding, item_ids
    )
    args.text_embedding_dim = int(common_x.shape[1])
    if args.search_embedding:
        search_x = load_precomputed_embedding(args.search_embedding, item_ids, "search")
        search_meta = {
            "mode": "precomputed_search",
            "path": str(args.search_embedding),
            "dim": int(search_x.shape[1]),
        }
    else:
        search_x, search_meta = build_search_signal_embeddings(
            args, item_ids, query_counter, device, Path(args.embedding_cache_dir)
        )
    model, logs, selection = train_anchor(
        args, common_x, behavior_x, search_x, prefix_sizes, device
    )
    encoded = encode_all(
        model, common_x, behavior_x, search_x, device, args.batch_size
    )
    codes = np.asarray(
        rq_assign_codes_np(encoded["z_anchor"], model.quantizer.codebooks, "argmin"),
        dtype=np.int64,
    )
    stats = {
        "method": (
            "P2 Common-Centered Bounded-Consensus Prefix"
            if args.anchor_mode == "consensus"
            else "M7b-v13 Snapshot-Anchored Multi-View Prefix"
        ),
        "anchor_mode": args.anchor_mode,
        "num_items": len(item_ids),
        "codebook_sizes": prefix_sizes,
        "prefix": prefix_stats(codes, prefix_sizes),
        "selection": selection,
        "views": {
            "common": vector_summary(common_x),
            "behavior": {**behavior_meta, **vector_summary(behavior_x)},
            "search": {**search_meta, **vector_summary(search_x)},
        },
        "anchor": {
            "z": vector_summary(encoded["z_anchor"]),
            "q": vector_summary(encoded["q_anchor"]),
            "gate_mean": encoded["gate"].mean(axis=0).tolist(),
            "gate_p10": np.quantile(encoded["gate"], 0.1, axis=0).tolist(),
            "gate_p90": np.quantile(encoded["gate"], 0.9, axis=0).tolist(),
            "consensus_alpha_mean": float(encoded["consensus_alpha"].mean())
            if encoded["consensus_alpha"] is not None
            else None,
            "consensus_alpha_p10": float(np.quantile(encoded["consensus_alpha"], 0.1))
            if encoded["consensus_alpha"] is not None
            else None,
            "consensus_alpha_p90": float(np.quantile(encoded["consensus_alpha"], 0.9))
            if encoded["consensus_alpha"] is not None
            else None,
            "common_cosine": float(
                F.cosine_similarity(
                    torch.from_numpy(encoded["common_recon"]),
                    torch.from_numpy(common_x),
                    dim=1,
                ).mean()
            ),
            "behavior_cosine": float(
                F.cosine_similarity(
                    torch.from_numpy(encoded["behavior_recon"]),
                    torch.from_numpy(behavior_x),
                    dim=1,
                ).mean()
            ),
            "search_cosine": float(
                F.cosine_similarity(
                    torch.from_numpy(encoded["search_recon"]),
                    torch.from_numpy(search_x),
                    dim=1,
                ).mean()
            )
            if encoded["search_recon"] is not None
            else None,
        },
        "params": vars(args),
    }
    write_json(output_dir / f"anchor_stats.{args.anchor_mode}.json", stats)
    write_json(output_dir / "anchor_train_log.json", logs)
    np.savez_compressed(
        output_dir / "anchor_lookup.npz",
        item_ids=np.asarray(item_ids),
        prefix_codes=codes,
        z_anchor=encoded["z_anchor"],
        q_anchor=encoded["q_anchor"],
        common_base=encoded["common_recon"],
        behavior_base=encoded["behavior_recon"],
    )
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "params": vars(args),
            "prefix_sizes": prefix_sizes,
            "selection": selection,
        },
        output_dir / "anchor_prefix_best.pt",
    )
    return stats


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--common-embedding", required=True)
    p.add_argument("--rec-behavior-embedding", required=True)
    p.add_argument("--search-embedding", default="")
    p.add_argument("--embedding-cache-dir", required=True)
    p.add_argument("--item-ids", default="")
    p.add_argument("--max-items", type=int, default=20000)
    p.add_argument("--anchor-mode", choices=["match", "full", "consensus"], required=True)
    p.add_argument("--prefix-codebook-sizes", default="256,256")
    p.add_argument("--max-queries-per-item", type=int, default=20)
    p.add_argument("--text-encoder-model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--text-encoder-prefix", default="")
    p.add_argument("--query-encoder-prefix", default="")
    p.add_argument("--text-encoder-pooling", choices=["mean", "cls"], default="cls")
    p.add_argument("--text-encoder-max-length", type=int, default=256)
    p.add_argument("--text-encoder-batch-size", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--anchor-epochs", type=int, default=100)
    p.add_argument("--anchor-warmup-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--anchor-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--vq-beta", type=float, default=0.25)
    p.add_argument("--balance-temperature", type=float, default=0.05)
    p.add_argument("--commit-weight", type=float, default=0.05)
    p.add_argument("--balance-weight", type=float, default=0.02)
    p.add_argument("--gate-diversity-weight", type=float, default=0.1)
    p.add_argument("--consensus-alpha-max", type=float, default=0.5)
    p.add_argument("--consensus-aux-recon-weight", type=float, default=0.25)
    p.add_argument("--consensus-agreement-weight", type=float, default=0.2)
    p.add_argument("--consensus-common-weight", type=float, default=0.05)
    p.add_argument("--consensus-consistency-weight", type=float, default=0.2)
    p.add_argument("--task-view-dropout", type=float, default=0.3)
    p.add_argument("--minimum-code-usage", type=int, default=128)
    p.add_argument("--selection-underuse-weight", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--device", default="cuda:7")
    p.add_argument("--seed", type=int, default=2026)
    return p


def main():
    args = parser().parse_args()
    stats = build(args)
    print(
        json.dumps(
            {
                "method": stats["method"],
                "anchor_mode": stats["anchor_mode"],
                "prefix": stats["prefix"],
                "anchor": stats["anchor"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
