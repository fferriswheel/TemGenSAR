#!/usr/bin/env python3
import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.data import DataLoader, TensorDataset

from build_neural_rq_sid_v2 import (
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
    l2_normalize_np,
)
from build_neural_rq_sid_v3_aligned_delta import (
    PrefixBatchSampler,
    collision_group_sinkhorn_loss,
    conditional_collision_stats,
    fixed_prefix,
    full_sid_collision_loss,
    inprefix_topology_loss,
    load_behavior_signal,
    load_precomputed_embedding,
    vector_summary,
)
from build_neural_rq_sid_v7_anchored_cross_view import mapping_from_codes
from build_neural_rq_sid_v8_frozen_anchor_private import (
    FrozenPrefixBehaviorBridge,
    bridge_quality,
    decode_bridge,
    restore_semantic_shared,
)
from refine_m7b_collision_group_sinkhorn import refine_last_suffix_layer
from refine_m7b_collision_group_sinkhorn import sinkhorn_assign_np


class DirectPrivateAdapter(nn.Module):
    """A task-private low-rank map conditioned on frozen shared and task views."""

    def __init__(self, task_dim, latent_dim, rank, dropout):
        super().__init__()
        self.task_projection = nn.Linear(task_dim, latent_dim)
        self.down = nn.Linear(latent_dim * 4, rank, bias=False)
        self.up = nn.Linear(rank, latent_dim, bias=False)
        self.norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.task_projection.weight)
        nn.init.zeros_(self.task_projection.bias)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)

    def forward(self, z_shared, task_x):
        task = self.norm(self.task_projection(task_x))
        features = torch.cat(
            [z_shared, task, z_shared * task, torch.abs(z_shared - task)], dim=1
        )
        private = self.up(self.dropout(F.gelu(self.down(features))))
        return self.norm(private)


class SplitSearchPrivateAdapter(nn.Module):
    """Separate query relevance from within-query item identity."""

    def __init__(
        self,
        common_dim,
        task_dim,
        latent_dim,
        relevance_dim,
        rank,
        dropout,
        use_full_common=False,
    ):
        super().__init__()
        if relevance_dim <= 0 or relevance_dim >= latent_dim:
            raise ValueError("search relevance dimension must be between 0 and latent_dim")
        self.relevance_dim = relevance_dim
        self.identity_dim = latent_dim - relevance_dim
        self.use_full_common = use_full_common
        self.query_projection = nn.Linear(task_dim, relevance_dim)
        self.relevance_down = nn.Linear(relevance_dim, min(rank, relevance_dim), bias=False)
        self.relevance_up = nn.Linear(min(rank, relevance_dim), relevance_dim, bias=False)
        self.common_residual_projection = nn.Linear(common_dim, self.identity_dim)
        self.latent_residual_projection = nn.Linear(latent_dim, self.identity_dim)
        self.identity_condition = nn.Linear(task_dim, self.identity_dim * 2)
        self.identity_down = nn.Linear(self.identity_dim, min(rank, self.identity_dim), bias=False)
        self.identity_up = nn.Linear(min(rank, self.identity_dim), self.identity_dim, bias=False)
        self.relevance_norm = nn.LayerNorm(relevance_dim)
        self.identity_norm = nn.LayerNorm(self.identity_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_shared, q_shared, common_x, semantic_base, search_x):
        relevance = self.query_projection(search_x)
        relevance = self.relevance_up(
            self.dropout(F.gelu(self.relevance_down(relevance)))
        )
        relevance = self.relevance_norm(relevance)

        common_identity = common_x if self.use_full_common else common_x - semantic_base
        identity = self.common_residual_projection(common_identity)
        identity = identity + self.latent_residual_projection(z_shared - q_shared)
        gamma, beta = self.identity_condition(search_x).chunk(2, dim=1)
        identity = identity * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
        identity = self.identity_up(self.dropout(F.gelu(self.identity_down(identity))))
        identity = self.identity_norm(identity)
        return torch.cat([relevance, identity], dim=1), relevance, identity


class PrefixConditionedResidualDecoder(nn.Module):
    """Decode a task residual while using the frozen prefix only as FiLM context."""

    def __init__(self, latent_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.suffix_projection = nn.Linear(latent_dim, hidden_dim)
        self.prefix_condition = nn.Linear(latent_dim, hidden_dim * 2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, suffix, q_shared):
        hidden = self.norm(self.suffix_projection(suffix))
        gamma, beta = self.prefix_condition(q_shared.detach()).chunk(2, dim=1)
        hidden = hidden * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
        return self.output(self.dropout(F.gelu(hidden)))


class DirectPrivateTokenizer(nn.Module):
    def __init__(self, shared, common_dim, search_dim, rec_dim, search_sizes, rec_sizes, args):
        super().__init__()
        self.shared = shared
        self.hierarchical_search_vq = args.hierarchical_search_vq
        self.split_search_private = args.split_search_private
        self.search_relevance_dim = args.search_relevance_dim
        if self.split_search_private:
            self.search_adapter = SplitSearchPrivateAdapter(
                common_dim,
                search_dim,
                args.latent_dim,
                args.search_relevance_dim,
                args.private_rank,
                args.dropout,
                use_full_common=bool(args.anchor_lookup),
            )
        else:
            self.search_adapter = DirectPrivateAdapter(
                search_dim, args.latent_dim, args.private_rank, args.dropout
            )
        self.rec_adapter = DirectPrivateAdapter(
            rec_dim, args.latent_dim, args.private_rank, args.dropout
        )
        if self.hierarchical_search_vq:
            if len(search_sizes) != 2:
                raise ValueError("hierarchical Search VQ requires exactly two suffix codebooks")
            identity_dim = args.latent_dim - args.search_relevance_dim
            self.search_relevance_quantizer = ResidualVectorQuantizer(
                [search_sizes[0]],
                args.search_relevance_dim,
                args.vq_beta,
                args.balance_temperature,
            )
            self.search_identity_quantizer = ResidualVectorQuantizer(
                [search_sizes[1]],
                identity_dim,
                args.vq_beta,
                args.balance_temperature,
                sk_epsilons=[args.direct_sinkhorn_epsilon]
                if args.direct_sinkhorn_vq
                else None,
                sk_iters=args.direct_sinkhorn_iters,
            )
        else:
            self.search_quantizer = ResidualVectorQuantizer(
                search_sizes, args.latent_dim, args.vq_beta, args.balance_temperature
            )
        self.rec_quantizer = ResidualVectorQuantizer(
            rec_sizes,
            args.latent_dim,
            args.vq_beta,
            args.balance_temperature,
            sk_epsilons=([0.0] * (len(rec_sizes) - 1) + [args.direct_sinkhorn_epsilon])
            if args.direct_sinkhorn_vq
            else None,
            sk_iters=args.direct_sinkhorn_iters,
        )
        self.conditioned_residual_decoder = args.conditioned_residual_decoder
        if self.conditioned_residual_decoder:
            self.search_decoder = PrefixConditionedResidualDecoder(
                args.latent_dim, args.hidden_dim, common_dim, args.dropout
            )
            self.rec_decoder = PrefixConditionedResidualDecoder(
                args.latent_dim, args.hidden_dim, rec_dim, args.dropout
            )
        else:
            self.search_decoder = nn.Sequential(
                nn.Linear(args.latent_dim, args.hidden_dim),
                nn.LayerNorm(args.hidden_dim),
                nn.GELU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_dim, common_dim),
            )
            self.rec_decoder = nn.Sequential(
                nn.Linear(args.latent_dim, args.hidden_dim),
                nn.LayerNorm(args.hidden_dim),
                nn.GELU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_dim, rec_dim),
            )
        if self.split_search_private:
            identity_dim = args.latent_dim - args.search_relevance_dim
            self.relevance_decoder = nn.Linear(args.search_relevance_dim, search_dim)
            self.identity_decoder = nn.Linear(identity_dim, common_dim)

    def private_latents(
        self,
        common_x,
        search_x,
        rec_x,
        fixed_z_shared=None,
        fixed_q_shared=None,
        fixed_semantic_base=None,
    ):
        if fixed_z_shared is None:
            with torch.no_grad():
                z_shared = self.shared.encoder(common_x)
                q_shared = self.shared.quantizer(z_shared)[0]
                semantic_base = self.shared.decoder(q_shared)
        else:
            z_shared = fixed_z_shared
            q_shared = fixed_q_shared
            semantic_base = fixed_semantic_base
        if self.split_search_private:
            search_private, search_relevance, search_identity = self.search_adapter(
                z_shared, q_shared, common_x, semantic_base, search_x
            )
        else:
            search_private = self.search_adapter(z_shared, search_x)
            search_relevance = search_private
            search_identity = search_private
        return (
            z_shared,
            q_shared,
            search_private,
            search_relevance,
            search_identity,
            self.rec_adapter(z_shared, rec_x),
        )

    def forward(
        self,
        common_x,
        search_x,
        rec_x,
        quantize=True,
        fixed_z_shared=None,
        fixed_q_shared=None,
        fixed_semantic_base=None,
    ):
        z_shared, q_shared, search_private, search_relevance, search_identity, rec_private = self.private_latents(
            common_x,
            search_x,
            rec_x,
            fixed_z_shared,
            fixed_q_shared,
            fixed_semantic_base,
        )
        if quantize and self.hierarchical_search_vq:
            q_relevance, relevance_idx, relevance_commit, relevance_balance = (
                self.search_relevance_quantizer(search_relevance)
            )
            q_identity, identity_idx, identity_commit, identity_balance = (
                self.search_identity_quantizer(search_identity)
            )
            q_search = torch.cat([q_relevance, q_identity], dim=1)
            search_idx = [relevance_idx[0], identity_idx[0]]
            search_commit = relevance_commit + identity_commit
            search_balance = relevance_balance + identity_balance
        elif quantize:
            q_search, search_idx, search_commit, search_balance = self.search_quantizer(
                search_private
            )
        else:
            q_search = search_private
            search_idx = []
            search_commit = search_private.new_tensor(0.0)
            search_balance = search_private.new_tensor(0.0)
        if quantize:
            q_rec, rec_idx, rec_commit, rec_balance = self.rec_quantizer(rec_private)
        else:
            q_rec = rec_private
            rec_idx = []
            rec_commit = rec_private.new_tensor(0.0)
            rec_balance = rec_private.new_tensor(0.0)
        if self.conditioned_residual_decoder:
            search_recon = self.search_decoder(q_search, q_shared)
            rec_recon = self.rec_decoder(q_rec, q_shared)
        else:
            search_recon = self.search_decoder(q_search)
            rec_recon = self.rec_decoder(q_rec)
        output = {
            "z_shared": z_shared,
            "q_shared": q_shared,
            "search_private": search_private,
            "search_relevance": search_relevance,
            "search_identity": search_identity,
            "rec_private": rec_private,
            "q_search": q_search,
            "q_rec": q_rec,
            "search_idx": search_idx,
            "rec_idx": rec_idx,
            "search_recon": search_recon,
            "rec_recon": rec_recon,
            "commit": search_commit + rec_commit,
            "balance": search_balance + rec_balance,
        }
        if self.split_search_private:
            relevance_dim = self.search_relevance_dim
            output["search_relevance_view"] = self.relevance_decoder(
                q_search[:, :relevance_dim]
            )
            output["search_identity_recon"] = self.identity_decoder(
                q_search[:, relevance_dim:]
            )
        return output


def cosine_loss(prediction, target):
    return (1.0 - F.cosine_similarity(prediction, target, dim=1)).mean()


def symmetric_contrastive_loss(left, right, temperature):
    left = F.normalize(left, dim=1)
    right = F.normalize(right, dim=1)
    logits = left @ right.t() / temperature
    labels = torch.arange(len(left), device=left.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def multi_positive_contrastive_loss(left, right, query_membership, temperature):
    """Treat items sharing any observed query as positives instead of false negatives."""
    valid = query_membership.any(dim=1)
    if int(valid.sum()) < 2:
        return left.new_tensor(0.0)
    left = F.normalize(left[valid], dim=1)
    right = F.normalize(right[valid], dim=1)
    membership = query_membership[valid].float()
    positive = membership @ membership.t() > 0
    logits = left @ right.t() / temperature

    def direction_loss(scores, mask):
        numerator = torch.logsumexp(scores.masked_fill(~mask, float("-inf")), dim=1)
        denominator = torch.logsumexp(scores, dim=1)
        return (denominator - numerator).mean()

    return 0.5 * (
        direction_loss(logits, positive)
        + direction_loss(logits.t(), positive.t())
    )


def cross_covariance_loss(shared, private):
    shared = (shared - shared.mean(dim=0)) / (shared.std(dim=0) + 1e-4)
    private = (private - private.mean(dim=0)) / (private.std(dim=0) + 1e-4)
    covariance = shared.t() @ private / max(len(shared) - 1, 1)
    return covariance.pow(2).mean()


@torch.no_grad()
def initialize_codebooks(
    model,
    common_x,
    search_x,
    rec_x,
    device,
    seed,
    batch_size,
    fixed_z_shared=None,
    fixed_q_shared=None,
    fixed_semantic_base=None,
):
    search_values, rec_values, relevance_values, identity_values = encode_private(
        model,
        common_x,
        search_x,
        rec_x,
        device,
        batch_size,
        return_parts=True,
        fixed_z_shared=fixed_z_shared,
        fixed_q_shared=fixed_q_shared,
        fixed_semantic_base=fixed_semantic_base,
    )
    if model.hierarchical_search_vq:
        init_rq_codebooks(
            model.search_relevance_quantizer, relevance_values, seed + 1000
        )
        init_rq_codebooks(
            model.search_identity_quantizer, identity_values, seed + 1500
        )
    else:
        init_rq_codebooks(model.search_quantizer, search_values, seed + 1000)
    init_rq_codebooks(model.rec_quantizer, rec_values, seed + 2000)


def hierarchical_search_codes_np(model, relevance_values, identity_values):
    relevance = rq_assign_codes_np(
        relevance_values, model.search_relevance_quantizer.codebooks, "argmin"
    )
    identity = rq_assign_codes_np(
        identity_values, model.search_identity_quantizer.codebooks, "argmin"
    )
    return np.column_stack(
        [np.asarray(relevance, dtype=np.int64)[:, 0], np.asarray(identity, dtype=np.int64)[:, 0]]
    )


def conditional_identity_sinkhorn_loss(
    prefix_codes,
    relevance_indices,
    identity_values,
    identity_quantizer,
    epsilon,
    iters,
    temperature,
):
    """Reassign only collided identity codes within a fixed prefix/relevance route."""
    if epsilon <= 0:
        return identity_values.new_tensor(0.0), 0
    codebook = identity_quantizer.codebooks[0]
    distances = (
        identity_values.pow(2).sum(dim=1, keepdim=True)
        - 2 * identity_values @ codebook.t()
        + codebook.pow(2).sum(dim=1).unsqueeze(0)
    )
    identity_indices = distances.argmin(dim=1)
    groups = defaultdict(list)
    for idx, (prefix, relevance, identity) in enumerate(
        zip(
            prefix_codes.detach().cpu().numpy(),
            relevance_indices.detach().cpu().numpy(),
            identity_indices.detach().cpu().numpy(),
        )
    ):
        groups[tuple(prefix.tolist()) + (int(relevance), int(identity))].append(idx)
    losses = []
    for group in groups.values():
        if len(group) < 2:
            continue
        group_distances = distances[group]
        targets, _ = sinkhorn_assign_np(
            group_distances.detach().cpu().numpy(), epsilon, iters
        )
        target_t = torch.from_numpy(targets).to(identity_values.device)
        losses.append(F.cross_entropy(-group_distances / temperature, target_t))
    if not losses:
        return identity_values.new_tensor(0.0), 0
    return torch.stack(losses).mean(), len(losses)


def conditional_identity_topology_loss(
    prefix_codes, relevance_indices, student_identity, teacher_identity
):
    group_codes = torch.cat([prefix_codes, relevance_indices[:, None]], dim=1)
    return inprefix_topology_loss(group_codes, student_identity, teacher_identity)


@torch.no_grad()
def encode_private(
    model,
    common_x,
    search_x,
    rec_x,
    device,
    batch_size,
    return_parts=False,
    fixed_z_shared=None,
    fixed_q_shared=None,
    fixed_semantic_base=None,
):
    search_values, rec_values = [], []
    relevance_values, identity_values = [], []
    if fixed_z_shared is None:
        fixed_z_shared = np.zeros((len(common_x), 0), dtype=np.float32)
        fixed_q_shared = np.zeros((len(common_x), 0), dtype=np.float32)
        fixed_semantic_base = np.zeros((len(common_x), 0), dtype=np.float32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(common_x),
            torch.from_numpy(search_x),
            torch.from_numpy(rec_x),
            torch.from_numpy(fixed_z_shared),
            torch.from_numpy(fixed_q_shared),
            torch.from_numpy(fixed_semantic_base),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    for common_b, search_b, rec_b, z_b, q_b, base_b in loader:
        use_fixed = z_b.shape[1] > 0
        _, _, search_private, search_relevance, search_identity, rec_private = model.private_latents(
            common_b.to(device),
            search_b.to(device),
            rec_b.to(device),
            z_b.to(device) if use_fixed else None,
            q_b.to(device) if use_fixed else None,
            base_b.to(device) if use_fixed else None,
        )
        search_values.append(search_private.cpu().numpy())
        rec_values.append(rec_private.cpu().numpy())
        relevance_values.append(search_relevance.cpu().numpy())
        identity_values.append(search_identity.cpu().numpy())
    values = (
        np.concatenate(search_values).astype(np.float32),
        np.concatenate(rec_values).astype(np.float32),
    )
    if not return_parts:
        return values
    return values + (
        np.concatenate(relevance_values).astype(np.float32),
        np.concatenate(identity_values).astype(np.float32),
    )


def build_query_membership(item_ids, query_counter, max_queries_per_item):
    queries = sorted(
        {
            query
            for item_id in item_ids
            for query, _ in query_counter[item_id].most_common(max_queries_per_item)
        }
    )
    query_to_idx = {query: idx for idx, query in enumerate(queries)}
    membership = np.zeros((len(item_ids), len(queries)), dtype=np.bool_)
    signatures = []
    for row_idx, item_id in enumerate(item_ids):
        rows = query_counter[item_id].most_common(max_queries_per_item)
        signature = []
        for query, _ in rows:
            membership[row_idx, query_to_idx[query]] = True
            signature.append(query_to_idx[query])
        signatures.append(tuple(signature))
    return membership, {
        "query_count": len(queries),
        "nonempty_item_count": int(membership.any(axis=1).sum()),
        "unique_query_set_count": len(set(signatures)),
    }


def collision_query_overlap_stats(prefix_codes, suffix_codes, query_membership):
    buckets = {}
    for idx, (prefix, suffix) in enumerate(zip(prefix_codes, suffix_codes)):
        buckets.setdefault(tuple(prefix) + tuple(suffix), []).append(idx)
    pair_count = 0
    overlap_count = 0
    for members in buckets.values():
        if len(members) < 2:
            continue
        for left_pos, left in enumerate(members):
            for right in members[left_pos + 1 :]:
                pair_count += 1
                overlap_count += bool(
                    np.logical_and(query_membership[left], query_membership[right]).any()
                )
    return {
        "collision_pair_count": pair_count,
        "shared_query_collision_pair_count": overlap_count,
        "shared_query_fraction": overlap_count / max(pair_count, 1),
    }


def restore_bridge(args, latent_dim, behavior_dim, device):
    bridge = FrozenPrefixBehaviorBridge(
        latent_dim, args.bridge_hidden_dim, behavior_dim, args.dropout
    ).to(device)
    checkpoint = torch.load(args.bridge_checkpoint, map_location="cpu", weights_only=False)
    bridge.load_state_dict(checkpoint["bridge_state_dict"])
    bridge.eval()
    for parameter in bridge.parameters():
        parameter.requires_grad = False
    return bridge


def train_task(
    args,
    shared,
    common_x,
    search_signal,
    behavior_x,
    semantic_base,
    behavior_base,
    search_target,
    rec_target,
    search_full_target,
    query_membership,
    prefix_codes,
    fixed_z_shared,
    fixed_q_shared,
    search_sizes,
    rec_sizes,
    device,
):
    for parameter in shared.parameters():
        parameter.requires_grad = False
    shared.eval()
    model = DirectPrivateTokenizer(
        shared,
        common_x.shape[1],
        search_signal.shape[1],
        behavior_x.shape[1],
        search_sizes,
        rec_sizes,
        args,
    ).to(device)
    if not args.hierarchical_search_vq:
        initialize_codebooks(
            model,
            common_x,
            search_signal,
            behavior_x,
            device,
            args.seed,
            args.batch_size,
            fixed_z_shared,
            fixed_q_shared,
            semantic_base,
        )
    dataset = TensorDataset(
        torch.from_numpy(common_x),
        torch.from_numpy(search_signal),
        torch.from_numpy(behavior_x),
        torch.from_numpy(semantic_base),
        torch.from_numpy(behavior_base),
        torch.from_numpy(search_target),
        torch.from_numpy(rec_target),
        torch.from_numpy(search_full_target),
        torch.from_numpy(query_membership),
        torch.from_numpy(prefix_codes),
        torch.from_numpy(fixed_z_shared),
        torch.from_numpy(fixed_q_shared),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=PrefixBatchSampler(
            prefix_codes, args.batch_size, args.prefix_group_size, args.seed + 17
        ),
        num_workers=0,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    logs = []
    search_adaptive_scale = 0.0
    rec_adaptive_scale = 0.0
    best_state = None
    best_score = float("inf")
    best_epoch = None
    for epoch in range(1, args.task_epochs + 1):
        if args.hierarchical_search_vq and epoch == args.continuous_warmup_epochs + 1:
            initialize_codebooks(
                model,
                common_x,
                search_signal,
                behavior_x,
                device,
                args.seed,
                args.batch_size,
                fixed_z_shared,
                fixed_q_shared,
                semantic_base,
            )
        quantize = not args.hierarchical_search_vq or epoch > args.continuous_warmup_epochs
        model.train()
        model.shared.eval()
        totals = Counter()
        steps = 0
        for (
            common_b,
            search_b,
            behavior_b,
            semantic_base_b,
            behavior_base_b,
            search_target_b,
            rec_target_b,
            search_full_target_b,
            query_membership_b,
            prefix_b,
            z_shared_b,
            q_shared_b,
        ) in loader:
            common_b = common_b.to(device)
            search_b = search_b.to(device)
            behavior_b = behavior_b.to(device)
            semantic_base_b = semantic_base_b.to(device)
            behavior_base_b = behavior_base_b.to(device)
            search_target_b = search_target_b.to(device)
            rec_target_b = rec_target_b.to(device)
            search_full_target_b = search_full_target_b.to(device)
            query_membership_b = query_membership_b.to(device)
            prefix_b = prefix_b.to(device)
            z_shared_b = z_shared_b.to(device)
            q_shared_b = q_shared_b.to(device)
            out = model(
                common_b,
                search_b,
                behavior_b,
                quantize=quantize,
                fixed_z_shared=z_shared_b,
                fixed_q_shared=q_shared_b,
                fixed_semantic_base=semantic_base_b,
            )
            residual_mse = F.mse_loss(out["search_recon"], search_target_b) + F.mse_loss(
                out["rec_recon"], rec_target_b
            )
            residual_cos = cosine_loss(out["search_recon"], search_target_b) + cosine_loss(
                out["rec_recon"], rec_target_b
            )
            search_full_recon = F.normalize(
                semantic_base_b + out["search_recon"], dim=1
            )
            rec_full_recon = F.normalize(behavior_base_b + out["rec_recon"], dim=1)
            full_mse = F.mse_loss(search_full_recon, search_full_target_b) + F.mse_loss(
                rec_full_recon, behavior_b
            )
            if args.split_search_private:
                query_item = multi_positive_contrastive_loss(
                    out["search_relevance_view"],
                    search_b,
                    query_membership_b,
                    args.query_contrast_temperature,
                )
                identity_target = (
                    common_b if args.anchor_lookup else common_b - semantic_base_b
                )
                identity_recon = F.mse_loss(
                    out["search_identity_recon"], identity_target
                ) + args.identity_cos_scale * cosine_loss(
                    out["search_identity_recon"], identity_target
                )
                identity_topology = inprefix_topology_loss(
                    prefix_b, out["q_search"][:, args.search_relevance_dim :], identity_target
                )
                search_redundancy = cross_covariance_loss(
                    out["z_shared"], out["search_relevance"]
                )
            else:
                query_item = symmetric_contrastive_loss(
                    search_full_recon, search_b, args.query_contrast_temperature
                )
                identity_recon = search_full_recon.new_tensor(0.0)
                identity_topology = search_full_recon.new_tensor(0.0)
                search_redundancy = cross_covariance_loss(
                    out["z_shared"], out["search_private"]
                )
            if args.hierarchical_search_vq:
                if quantize:
                    search_topology = conditional_identity_topology_loss(
                        prefix_b,
                        out["search_idx"][0],
                        out["q_search"][:, args.search_relevance_dim :],
                        identity_target,
                    )
                else:
                    search_topology = inprefix_topology_loss(
                        prefix_b, out["search_identity"], identity_target
                    )
            else:
                search_topology = inprefix_topology_loss(
                    prefix_b, out["q_search"], search_full_target_b
                )
            rec_topology = inprefix_topology_loss(prefix_b, out["q_rec"], behavior_b)
            if args.hierarchical_search_vq:
                if quantize and not args.direct_sinkhorn_vq:
                    search_conditional, search_groups = conditional_identity_sinkhorn_loss(
                        prefix_b,
                        out["search_idx"][0],
                        out["search_identity"],
                        model.search_identity_quantizer,
                        args.conditional_sinkhorn_epsilon,
                        args.conditional_sinkhorn_iters,
                        args.conditional_sinkhorn_temperature,
                    )
                    rec_conditional, rec_groups = collision_group_sinkhorn_loss(
                        prefix_b,
                        out["rec_private"],
                        model.rec_quantizer,
                        args.conditional_sinkhorn_epsilon,
                        args.conditional_sinkhorn_iters,
                        args.conditional_sinkhorn_temperature,
                    )
                else:
                    search_conditional = search_full_recon.new_tensor(0.0)
                    rec_conditional = search_full_recon.new_tensor(0.0)
                    search_groups = rec_groups = 0
                reconstruction = cosine_loss(
                    search_full_recon, search_full_target_b
                ) + cosine_loss(rec_full_recon, behavior_b)
                reconstruction = reconstruction + args.identity_anchor_weight * F.mse_loss(
                    out["search_identity_recon"], identity_target
                )
                topology = search_topology + rec_topology
                loss = (
                    reconstruction
                    + args.hierarchical_residual_mse_weight * residual_mse
                    + args.hierarchical_residual_cos_weight * residual_cos
                    + args.hierarchical_full_mse_weight * full_mse
                    + args.query_contrast_weight * query_item
                    + args.commit_weight * out["commit"]
                    + args.topology_weight * topology
                    + (0.0 if args.direct_sinkhorn_vq else args.conditional_sinkhorn_weight)
                    * (
                        search_adaptive_scale * search_conditional
                        + rec_adaptive_scale * rec_conditional
                    )
                )
                redundancy = search_full_recon.new_tensor(0.0)
                uniqueness = search_full_recon.new_tensor(0.0)
                confidence = search_full_recon.new_tensor(0.0)
            else:
                search_conditional = search_full_recon.new_tensor(0.0)
                rec_conditional = search_full_recon.new_tensor(0.0)
                search_groups = rec_groups = 0
                reconstruction = residual_mse
                topology = search_topology + rec_topology
                redundancy = search_redundancy + cross_covariance_loss(
                    out["z_shared"], out["rec_private"]
                )
                search_collision, _, search_confidence, _ = full_sid_collision_loss(
                    prefix_b,
                    out["search_private"],
                    model.search_quantizer,
                    args.full_sid_temperature,
                    args.full_sid_topk,
                )
                rec_collision, _, rec_confidence, _ = full_sid_collision_loss(
                    prefix_b,
                    out["rec_private"],
                    model.rec_quantizer,
                    args.full_sid_temperature,
                    args.full_sid_topk,
                )
                uniqueness = search_collision + rec_collision
                confidence = search_confidence + rec_confidence
                loss = (
                    args.task_mse_weight * residual_mse
                    + args.task_cos_weight * residual_cos
                    + args.full_mse_weight * full_mse
                    + args.query_contrast_weight * query_item
                    + args.identity_recon_weight * identity_recon
                    + args.identity_topology_weight * identity_topology
                    + args.topology_weight * topology
                    + args.redundancy_weight * redundancy
                    + args.full_sid_collision_weight * uniqueness
                    + args.assignment_confidence_weight * confidence
                    + args.commit_weight * out["commit"]
                    + args.balance_weight * out["balance"]
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            for key, value in {
                "loss": loss,
                "residual_mse": residual_mse,
                "residual_cos": residual_cos,
                "full_mse": full_mse,
                "reconstruction": reconstruction,
                "query_item": query_item,
                "identity_recon": identity_recon,
                "identity_topology": identity_topology,
                "search_topology": search_topology,
                "rec_topology": rec_topology,
                "search_conditional": search_conditional,
                "rec_conditional": rec_conditional,
                "search_adaptive_scale": search_full_recon.new_tensor(search_adaptive_scale),
                "rec_adaptive_scale": search_full_recon.new_tensor(rec_adaptive_scale),
                "redundancy": redundancy,
                "uniqueness": uniqueness,
                "confidence": confidence,
                "commit": out["commit"],
                "balance": out["balance"],
            }.items():
                totals[key] += float(value.detach().cpu())
            totals["search_collision_groups"] += search_groups
            totals["rec_collision_groups"] += rec_groups
            steps += 1
        row = {"epoch": epoch, **{key: value / max(steps, 1) for key, value in totals.items()}}
        should_evaluate = (
            args.hierarchical_search_vq and quantize
        ) or epoch == 1 or epoch == args.task_epochs or epoch % args.log_every == 0
        if should_evaluate:
            search_private, rec_private, relevance_values, identity_values = encode_private(
                model,
                common_x,
                search_signal,
                behavior_x,
                device,
                args.batch_size,
                return_parts=True,
                fixed_z_shared=fixed_z_shared,
                fixed_q_shared=fixed_q_shared,
                fixed_semantic_base=semantic_base,
            )
            if args.hierarchical_search_vq:
                if not quantize:
                    row["phase"] = "continuous_warmup"
                    logs.append(row)
                    if epoch == 1 or epoch % args.log_every == 0:
                        print(json.dumps({"stage": "hierarchical", **row}), flush=True)
                    continue
                search_codes = hierarchical_search_codes_np(
                    model, relevance_values, identity_values
                )
            else:
                search_codes = np.asarray(
                    rq_assign_codes_np(
                        search_private, model.search_quantizer.codebooks, "argmin"
                    ),
                    dtype=np.int64,
                )
            rec_codes = np.asarray(
                rq_assign_codes_np(rec_private, model.rec_quantizer.codebooks, "argmin"),
                dtype=np.int64,
            )
            search_stats = conditional_collision_stats(
                prefix_codes, search_codes
            )
            rec_stats = conditional_collision_stats(prefix_codes, rec_codes)
            row["search_duplicate"] = search_stats["duplicate_fraction"]
            row["rec_duplicate"] = rec_stats["duplicate_fraction"]
            if args.hierarchical_search_vq:
                if epoch >= args.collision_warmup_epochs:
                    search_adaptive_scale = float(
                        np.clip(
                            search_adaptive_scale
                            + args.collision_controller_gain
                            * (row["search_duplicate"] - args.search_collision_target),
                            0.0,
                            1.0,
                        )
                    )
                    rec_adaptive_scale = float(
                        np.clip(
                            rec_adaptive_scale
                            + args.collision_controller_gain
                            * (row["rec_duplicate"] - args.rec_collision_target),
                            0.0,
                            1.0,
                        )
                    )
                search_usage = code_usage(search_codes, search_sizes)
                underuse = sum(
                    max(args.minimum_code_usage - level["used_codes"], 0)
                    / max(args.minimum_code_usage, 1)
                    for level in search_usage
                )
                excess_collision = max(
                    row["search_duplicate"] - args.search_collision_target, 0.0
                ) + max(row["rec_duplicate"] - args.rec_collision_target, 0.0)
                selection_score = (
                    row["reconstruction"]
                    + args.query_contrast_weight * row["query_item"]
                    + args.topology_weight
                    * (row["search_topology"] + row["rec_topology"])
                    + args.selection_collision_weight * excess_collision
                    + args.selection_underuse_weight * underuse
                )
                row.update(
                    {
                        "phase": "quantized",
                        "next_search_adaptive_scale": search_adaptive_scale,
                        "next_rec_adaptive_scale": rec_adaptive_scale,
                        "search_code_usage": [level["used_codes"] for level in search_usage],
                        "selection_score": selection_score,
                    }
                )
                if selection_score < best_score:
                    best_score = selection_score
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            if epoch == 1 or epoch == args.task_epochs or epoch % args.log_every == 0:
                print(
                    json.dumps(
                        {"stage": "hierarchical" if args.hierarchical_search_vq else "direct_private", **row}
                    ),
                    flush=True,
                )
        logs.append(row)
    if args.hierarchical_search_vq:
        if best_state is None:
            raise RuntimeError("hierarchical training produced no quantized checkpoint")
        model.load_state_dict(best_state)
        logs.append(
            {"selected_best_epoch": best_epoch, "selected_best_score": best_score}
        )
    return model, logs


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
    if args.direct_sinkhorn_vq and not args.hierarchical_search_vq:
        raise ValueError("--direct-sinkhorn-vq requires --hierarchical-search-vq")
    if args.hierarchical_search_vq:
        args.split_search_private = True
        args.conditioned_residual_decoder = True
        if args.export_sinkhorn_rounds:
            raise ValueError("v11 reports raw learned SID; set --export-sinkhorn-rounds 0")

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
    if args.search_embedding:
        search_signal = load_precomputed_embedding(
            args.search_embedding, item_ids, "search"
        )
        search_signal_stats = {
            "mode": "precomputed_search",
            "path": str(args.search_embedding),
            "dim": int(search_signal.shape[1]),
        }
    else:
        search_signal, search_signal_stats = build_search_signal_embeddings(
            args, item_ids, query_counter, device, Path(args.embedding_cache_dir)
        )
    query_membership, query_membership_stats = build_query_membership(
        item_ids, query_counter, args.max_queries_per_item
    )
    anchor_mode = None
    if args.anchor_lookup:
        from train_multiview_anchor_prefix_v13 import MultiViewAnchorRQVAE

        if not args.anchor_checkpoint:
            raise ValueError("--anchor-checkpoint is required with --anchor-lookup")
        anchor_checkpoint = torch.load(
            args.anchor_checkpoint, map_location="cpu", weights_only=False
        )
        anchor_args = argparse.Namespace(**anchor_checkpoint["params"])
        anchor_mode = anchor_args.anchor_mode
        shared = MultiViewAnchorRQVAE(
            common_x.shape[1],
            behavior_x.shape[1],
            search_signal.shape[1],
            prefix_sizes,
            anchor_args,
        ).to(device)
        shared.load_state_dict(anchor_checkpoint["model_state_dict"])
        shared.eval()
        lookup = np.load(args.anchor_lookup)
        lookup_ids = [str(value) for value in lookup["item_ids"].tolist()]
        if lookup_ids != item_ids:
            raise ValueError("anchor lookup item order does not match the current dataset")
        prefix_codes = lookup["prefix_codes"].astype(np.int64)
        z_shared = lookup["z_anchor"].astype(np.float32)
        q_shared = lookup["q_anchor"].astype(np.float32)
        semantic_base = l2_normalize_np(
            lookup["common_base"].astype(np.float32)
        ).astype(np.float32)
        behavior_base = l2_normalize_np(
            lookup["behavior_base"].astype(np.float32)
        ).astype(np.float32)
        bridge = None
    else:
        shared = restore_semantic_shared(args, common_x.shape[1], prefix_sizes, device)
        if shared is None:
            raise ValueError("--shared-checkpoint is required for the controlled v9 experiment")
        shared.eval()
        prefix_codes, q_shared, z_shared = fixed_prefix(shared, common_x, args.prefix_assignment)
        with torch.no_grad():
            semantic_base = (
                shared.decoder(torch.from_numpy(q_shared).to(device)).cpu().numpy().astype(np.float32)
            )
        bridge = restore_bridge(args, args.latent_dim, behavior_x.shape[1], device)
        behavior_base = decode_bridge(bridge, q_shared, device, args.batch_size)
    search_full_target = l2_normalize_np(
        common_x + args.search_target_scale * search_signal
    ).astype(np.float32)
    search_target = (search_full_target - semantic_base).astype(np.float32)
    rec_target = (behavior_x - behavior_base).astype(np.float32)

    model, task_logs = train_task(
        args,
        shared,
        common_x,
        search_signal,
        behavior_x,
        semantic_base,
        behavior_base,
        search_target,
        rec_target,
        search_full_target,
        query_membership,
        prefix_codes,
        z_shared,
        q_shared,
        search_sizes,
        rec_sizes,
        device,
    )
    search_private, rec_private, search_relevance, search_identity = encode_private(
        model,
        common_x,
        search_signal,
        behavior_x,
        device,
        args.batch_size,
        return_parts=True,
        fixed_z_shared=z_shared,
        fixed_q_shared=q_shared,
        fixed_semantic_base=semantic_base,
    )
    if args.hierarchical_search_vq:
        raw_search = hierarchical_search_codes_np(
            model, search_relevance, search_identity
        )
    else:
        raw_search = np.asarray(
            rq_assign_codes_np(search_private, model.search_quantizer.codebooks, "argmin"),
            dtype=np.int64,
        )
    raw_rec = np.asarray(
        rq_assign_codes_np(rec_private, model.rec_quantizer.codebooks, "argmin"),
        dtype=np.int64,
    )
    if args.hierarchical_search_vq:
        final_search = raw_search.copy()
        search_refine = {
            "method": "none",
            "reason": "v11 evaluates raw learned hierarchical suffix codes",
            "changed_fraction": 0.0,
        }
    else:
        final_search, search_refine = refine_last_suffix_layer(
            prefix_codes,
            raw_search,
            search_private,
            model.search_quantizer.codebooks,
            args.export_sinkhorn_epsilon,
            args.export_sinkhorn_iters,
            args.export_sinkhorn_rounds,
        )
    final_rec, rec_refine = refine_last_suffix_layer(
        prefix_codes,
        raw_rec,
        rec_private,
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
    explicit_conditional_residual = any(
        weight > 0
        for weight in (
            args.hierarchical_residual_mse_weight,
            args.hierarchical_residual_cos_weight,
            args.hierarchical_full_mse_weight,
        )
    )
    variant = (
        f"v13_{anchor_mode}_anchor_conditional_residual_sinkhorn"
        if anchor_mode and explicit_conditional_residual
        else f"v13_{anchor_mode}_anchor_direct_sinkhorn"
        if anchor_mode
        else "v12_direct_sinkhorn_vq"
        if args.hierarchical_search_vq and args.direct_sinkhorn_vq
        else "v11_hierarchical_relid"
        if args.hierarchical_search_vq
        else "v10c_conditioned_split"
        if args.conditioned_residual_decoder and args.split_search_private
        else "v10a_conditioned"
        if args.conditioned_residual_decoder
        else "v10b_split"
        if args.split_search_private
        else "v9_direct_private"
    )
    write_json(
        output_dir / f"sid_mapping.{variant}.json",
        mapping_from_codes(item_ids, prefix_codes, final_search, final_rec),
    )
    write_json(output_dir / "task_train_log.json", task_logs)
    # Persist the learned suffix before deriving optional report-only statistics.
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "params": vars(args),
            "prefix_sizes": prefix_sizes,
            "search_sizes": search_sizes,
            "rec_sizes": rec_sizes,
            "fixed_prefix_codes": prefix_codes,
        },
        output_dir / "direct_private_sid_model.pt",
    )

    prefix_labels = ["_".join(map(str, row)) for row in prefix_codes]
    search_labels = ["_".join(map(str, row)) for row in final_search]
    rec_labels = ["_".join(map(str, row)) for row in final_rec]
    prefix_counter = Counter(prefix_labels)
    method = {
        "v9_direct_private": "M7b-v9 Direct Private Low-Rank Suffix + Query-Item Contrast",
        "v10a_conditioned": "M7b-v10a Prefix-Conditioned Residual Decoder",
        "v10b_split": "M7b-v10b Split Search Relevance-Identity",
        "v10c_conditioned_split": "M7b-v10c Conditioned Decoder + Split Search",
        "v11_hierarchical_relid": (
            "M7b-v11 Hierarchical Relevance-Identity VQ + Adaptive Conditional Sinkhorn"
        ),
        "v12_direct_sinkhorn_vq": (
            "M7b-v12 Hierarchical Relevance-Identity + GenSAR Direct Sinkhorn VQ"
        ),
        "v13_match_anchor_direct_sinkhorn": (
            "M7b-v13a Matched Multi-View Anchor + Direct Sinkhorn Suffix"
        ),
        "v13_full_anchor_direct_sinkhorn": (
            "M7b-v13b Explicit-Search Multi-View Anchor + Direct Sinkhorn Suffix"
        ),
        "v13_consensus_anchor_direct_sinkhorn": (
            "P2 Common-Centered Bounded-Consensus Prefix + Direct Sinkhorn Suffix"
        ),
        "v13_full_anchor_conditional_residual_sinkhorn": (
            "P1-R Snapshot Anchor + Explicit Conditional-Residual Suffix"
        ),
        "v13_consensus_anchor_conditional_residual_sinkhorn": (
            "P2-R Bounded-Consensus Anchor + Explicit Conditional-Residual Suffix"
        ),
    }[variant]
    final_train_row = next(row for row in reversed(task_logs) if "epoch" in row)
    selection = task_logs[-1] if "selected_best_epoch" in task_logs[-1] else None
    stats = {
        "method": method,
        "variant": variant,
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
        "anchor_source": {
            "mode": anchor_mode,
            "lookup": args.anchor_lookup or None,
            "checkpoint": args.anchor_checkpoint or None,
        },
        "views": {
            "common": vector_summary(common_x),
            "search_signal": {
                **search_signal_stats,
                **query_membership_stats,
                **vector_summary(search_signal),
            },
            "behavior": {**behavior_stats, **vector_summary(behavior_x)},
            "search_full_target": vector_summary(search_full_target),
            "search_private_target": vector_summary(search_target),
            "rec_private_target": vector_summary(rec_target),
        },
        "latent": {
            "z_shared": vector_summary(z_shared),
            "search_private": vector_summary(search_private),
            "search_relevance": vector_summary(search_relevance),
            "search_identity": vector_summary(search_identity),
            "rec_private": vector_summary(rec_private),
        },
        "raw_assignment": {
            "search_collision": conditional_collision_stats(prefix_codes, raw_search),
            "rec_collision": conditional_collision_stats(prefix_codes, raw_rec),
            "search_usage": code_usage(raw_search, search_sizes),
            "rec_usage": code_usage(raw_rec, rec_sizes),
            "search_collision_query_overlap": collision_query_overlap_stats(
                prefix_codes, raw_search, query_membership
            ),
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
        "task_train_log_final": final_train_row,
        "checkpoint_selection": selection,
        "params": vars(args),
    }
    write_json(output_dir / f"sid_stats.{variant}.json", stats)
    return stats


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--common-embedding", required=True)
    p.add_argument("--rec-behavior-embedding", required=True)
    p.add_argument("--search-embedding", default="")
    p.add_argument("--shared-checkpoint", default="")
    p.add_argument("--bridge-checkpoint", default="")
    p.add_argument("--anchor-lookup", default="")
    p.add_argument("--anchor-checkpoint", default="")
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
    p.add_argument("--private-rank", type=int, default=32)
    p.add_argument("--conditioned-residual-decoder", action="store_true")
    p.add_argument("--split-search-private", action="store_true")
    p.add_argument("--hierarchical-search-vq", action="store_true")
    p.add_argument("--direct-sinkhorn-vq", action="store_true")
    p.add_argument("--direct-sinkhorn-epsilon", type=float, default=0.003)
    p.add_argument("--direct-sinkhorn-iters", type=int, default=50)
    p.add_argument("--search-relevance-dim", type=int, default=24)
    p.add_argument("--bridge-hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--task-epochs", type=int, default=30)
    p.add_argument("--continuous-warmup-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--vq-beta", type=float, default=0.25)
    p.add_argument("--balance-temperature", type=float, default=0.05)
    p.add_argument("--search-target-scale", type=float, default=0.5)
    p.add_argument("--task-mse-weight", type=float, default=10.0)
    p.add_argument("--task-cos-weight", type=float, default=0.1)
    p.add_argument("--full-mse-weight", type=float, default=1.0)
    p.add_argument("--hierarchical-residual-mse-weight", type=float, default=0.0)
    p.add_argument("--hierarchical-residual-cos-weight", type=float, default=0.0)
    p.add_argument("--hierarchical-full-mse-weight", type=float, default=0.0)
    p.add_argument("--query-contrast-weight", type=float, default=0.2)
    p.add_argument("--query-contrast-temperature", type=float, default=0.1)
    p.add_argument("--identity-recon-weight", type=float, default=2.0)
    p.add_argument("--identity-cos-scale", type=float, default=0.1)
    p.add_argument("--identity-anchor-weight", type=float, default=0.5)
    p.add_argument("--identity-topology-weight", type=float, default=0.5)
    p.add_argument("--topology-weight", type=float, default=1.0)
    p.add_argument("--redundancy-weight", type=float, default=0.05)
    p.add_argument("--full-sid-collision-weight", type=float, default=0.02)
    p.add_argument("--full-sid-temperature", type=float, default=0.05)
    p.add_argument("--full-sid-topk", type=int, default=4)
    p.add_argument("--assignment-confidence-weight", type=float, default=0.01)
    p.add_argument("--commit-weight", type=float, default=0.05)
    p.add_argument("--balance-weight", type=float, default=0.02)
    p.add_argument("--conditional-sinkhorn-weight", type=float, default=0.02)
    p.add_argument("--conditional-sinkhorn-epsilon", type=float, default=0.03)
    p.add_argument("--conditional-sinkhorn-iters", type=int, default=10)
    p.add_argument("--conditional-sinkhorn-temperature", type=float, default=0.1)
    p.add_argument("--collision-warmup-epochs", type=int, default=10)
    p.add_argument("--collision-controller-gain", type=float, default=2.0)
    p.add_argument("--search-collision-target", type=float, default=0.08)
    p.add_argument("--rec-collision-target", type=float, default=0.08)
    p.add_argument("--minimum-code-usage", type=int, default=128)
    p.add_argument("--selection-collision-weight", type=float, default=2.0)
    p.add_argument("--selection-underuse-weight", type=float, default=0.2)
    p.add_argument("--prefix-group-size", type=int, default=16)
    p.add_argument("--export-sinkhorn-rounds", type=int, default=0)
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
                "latent": stats["latent"],
                "raw_assignment": stats["raw_assignment"],
                "final_assignment": stats["final_assignment"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
