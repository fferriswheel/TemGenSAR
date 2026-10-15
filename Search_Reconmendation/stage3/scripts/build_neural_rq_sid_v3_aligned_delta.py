#!/usr/bin/env python3
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.data import DataLoader, Sampler, TensorDataset

from build_neural_rq_sid_v2 import (
    ResidualVectorQuantizer,
    aggregate_rec_neighbors,
    aggregate_search_queries,
    code_usage,
    collision_rate,
    count_summary,
    field_coverage,
    init_rq_codebooks,
    load_item_rows,
    make_docs,
    parse_sizes,
    read_jsonl,
    rq_assign_codes_np,
    write_json,
)
from build_neural_rq_sid_v2_text_encoder_fusion_plc import (
    LoRAEncoder,
    SharedRQVAE,
    build_rec_signal_embeddings,
    build_search_signal_embeddings,
    encode_text_docs,
    l2_normalize_np,
    make_task_targets,
    reconstruct_from_rq_codes,
    token_list,
    train_shared,
)
from refine_m7b_collision_group_sinkhorn import refine_last_suffix_layer, sinkhorn_assign_np


class BoundedTaskAugment(nn.Module):
    """Identity-preserving task augmentation in the frozen text-embedding space."""

    def __init__(self, common_dim, task_dim, hidden_dim, max_scale, gate_bias):
        super().__init__()
        self.max_scale = float(max_scale)
        self.task_proj = nn.Linear(task_dim, common_dim)
        self.gate = nn.Sequential(
            nn.Linear(common_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.xavier_uniform_(self.task_proj.weight)
        nn.init.zeros_(self.task_proj.bias)
        nn.init.zeros_(self.gate[2].weight)
        nn.init.constant_(self.gate[2].bias, gate_bias)

    def forward(self, common_x, task_x):
        direction = F.normalize(self.task_proj(task_x), p=2, dim=1)
        gate = self.gate(torch.cat([common_x, direction], dim=1))
        augmented = F.normalize(common_x + self.max_scale * gate * direction, p=2, dim=1)
        return augmented, gate


class AlignedDeltaTokenizer(nn.Module):
    def __init__(
        self,
        shared,
        common_dim,
        search_signal_dim,
        rec_signal_dim,
        hidden_dim,
        latent_dim,
        search_sizes,
        rec_sizes,
        beta,
        balance_temperature,
        lora_rank,
        lora_alpha,
        augment_scale,
        gate_bias,
        delta_max_ratio,
        search_target_dim,
        rec_target_dim,
        dropout,
    ):
        super().__init__()
        self.shared = shared
        self.delta_max_ratio = float(delta_max_ratio)
        self.search_augment = BoundedTaskAugment(
            common_dim, search_signal_dim, hidden_dim, augment_scale, gate_bias
        )
        self.rec_augment = BoundedTaskAugment(common_dim, rec_signal_dim, hidden_dim, augment_scale, gate_bias)
        self.search_encoder = LoRAEncoder(shared.encoder, lora_rank, lora_alpha)
        self.rec_encoder = LoRAEncoder(shared.encoder, lora_rank, lora_alpha)
        self.search_quantizer = ResidualVectorQuantizer(
            search_sizes, latent_dim, beta, balance_temperature
        )
        self.rec_quantizer = ResidualVectorQuantizer(rec_sizes, latent_dim, beta, balance_temperature)
        self.search_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, search_target_dim),
        )
        self.rec_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, rec_target_dim),
        )

    def bounded_delta(self, raw_delta, z_shared):
        raw_norm = raw_delta.norm(dim=1, keepdim=True).clamp(min=1e-8)
        max_norm = self.delta_max_ratio * z_shared.norm(dim=1, keepdim=True).detach().clamp(min=1e-8)
        scale = torch.minimum(torch.ones_like(raw_norm), max_norm / raw_norm)
        return raw_delta * scale

    def task_deltas(self, z_shared, common_x, search_signal, rec_signal):
        search_aug, search_gate = self.search_augment(common_x, search_signal)
        rec_aug, rec_gate = self.rec_augment(common_x, rec_signal)
        search_raw = self.search_encoder(search_aug) - z_shared
        rec_raw = self.rec_encoder(rec_aug) - z_shared
        search_delta = self.bounded_delta(search_raw, z_shared)
        rec_delta = self.bounded_delta(rec_raw, z_shared)
        return search_delta, rec_delta, search_gate, rec_gate

    def forward(self, common_x, search_signal, rec_signal):
        with torch.no_grad():
            z_shared = self.shared.encoder(common_x)
        search_delta, rec_delta, search_gate, rec_gate = self.task_deltas(
            z_shared, common_x, search_signal, rec_signal
        )
        q_search, search_idx, search_commit, search_balance = self.search_quantizer(search_delta)
        q_rec, rec_idx, rec_commit, rec_balance = self.rec_quantizer(rec_delta)
        return {
            "z_shared": z_shared,
            "search_delta": search_delta,
            "rec_delta": rec_delta,
            "search_gate": search_gate,
            "rec_gate": rec_gate,
            "q_search": q_search,
            "q_rec": q_rec,
            "search_idx": search_idx,
            "rec_idx": rec_idx,
            "search_recon": self.search_decoder(q_search),
            "rec_recon": self.rec_decoder(q_rec),
            "commit": search_commit + rec_commit,
            "balance": search_balance + rec_balance,
        }


class PrefixBatchSampler(Sampler):
    def __init__(self, prefix_codes, batch_size, group_size, seed):
        self.batch_size = int(batch_size)
        self.group_size = int(group_size)
        self.seed = int(seed)
        self.epoch = 0
        buckets = defaultdict(list)
        for idx, code in enumerate(prefix_codes):
            buckets[tuple(int(value) for value in code)].append(idx)
        self.buckets = list(buckets.values())
        self.num_items = len(prefix_codes)

    def __len__(self):
        return math.ceil(self.num_items / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        chunks = []
        for bucket in self.buckets:
            values = list(bucket)
            rng.shuffle(values)
            chunks.extend(values[start : start + self.group_size] for start in range(0, len(values), self.group_size))
        rng.shuffle(chunks)
        batch = []
        for chunk in chunks:
            if batch and len(batch) + len(chunk) > self.batch_size:
                yield batch
                batch = []
            batch.extend(chunk)
        if batch:
            yield batch


def load_behavior_signal(path, item_ids):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, torch.Tensor):
        if len(checkpoint) != len(item_ids):
            raise ValueError(
                f"behavior embedding has {len(checkpoint)} rows, expected {len(item_ids)}"
            )
        source_x = checkpoint.float().numpy()
        return l2_normalize_np(source_x), {
            "mode": "precomputed_behavior",
            "path": str(path),
            "dim": int(source_x.shape[1]),
        }
    source_ids = checkpoint["item_ids"]
    source_x = checkpoint["embedding"].float().numpy()
    by_id = {item_id: source_x[idx] for idx, item_id in enumerate(source_ids)}
    missing = [item_id for item_id in item_ids if item_id not in by_id]
    if missing:
        raise KeyError(f"{len(missing)} items missing from behavior embedding: {missing[:3]}")
    return np.stack([by_id[item_id] for item_id in item_ids]).astype(np.float32), {
        "mode": "directed_item2vec",
        "path": str(path),
        "dim": int(source_x.shape[1]),
    }


def load_precomputed_embedding(path, item_ids, label):
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        value = value.get("embedding")
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{label} embedding must be a rank-2 tensor")
    if len(value) != len(item_ids):
        raise ValueError(f"{label} embedding has {len(value)} rows, expected {len(item_ids)}")
    return l2_normalize_np(value.float().numpy())


def fixed_prefix(shared, common_x, assignment):
    with torch.no_grad():
        z_shared = shared.encoder(torch.from_numpy(common_x).to(next(shared.parameters()).device)).cpu().numpy()
    codes = rq_assign_codes_np(z_shared, shared.quantizer.codebooks, assignment)
    q_shared = reconstruct_from_rq_codes(codes, shared.quantizer.codebooks)
    return np.asarray(codes, dtype=np.int64), q_shared.astype(np.float32), z_shared.astype(np.float32)


@torch.no_grad()
def initialize_suffix_codebooks(model, common_x, search_signal, rec_signal, device, seed):
    common_t = torch.from_numpy(common_x).to(device)
    search_t = torch.from_numpy(search_signal).to(device)
    rec_t = torch.from_numpy(rec_signal).to(device)
    z_shared = model.shared.encoder(common_t)
    search_delta, rec_delta, _, _ = model.task_deltas(z_shared, common_t, search_t, rec_t)
    init_rq_codebooks(model.search_quantizer, search_delta.cpu().numpy(), seed + 1000)
    init_rq_codebooks(model.rec_quantizer, rec_delta.cpu().numpy(), seed + 2000)


def cosine_recon_loss(prediction, target):
    return (1.0 - F.cosine_similarity(prediction, target, dim=1)).mean()


def collision_group_sinkhorn_loss(prefix_codes, values, quantizer, epsilon, iters, temperature):
    if epsilon <= 0:
        return values.new_tensor(0.0), 0
    residual = values
    all_indices = []
    final_distances = None
    for level, codebook in enumerate(quantizer.codebooks):
        distances = (
            residual.pow(2).sum(dim=1, keepdim=True)
            - 2 * residual @ codebook.t()
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = distances.argmin(dim=1)
        all_indices.append(indices)
        if level == len(quantizer.codebooks) - 1:
            final_distances = distances
        residual = residual - F.embedding(indices, codebook).detach()

    prefix_np = prefix_codes.detach().cpu().numpy()
    suffix_np = torch.stack(all_indices, dim=1).detach().cpu().numpy()
    groups = defaultdict(list)
    for idx, (prefix, suffix) in enumerate(zip(prefix_np, suffix_np)):
        groups[tuple(prefix.tolist() + suffix.tolist())].append(idx)
    losses = []
    collision_groups = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        collision_groups += 1
        group_distances = final_distances[group]
        targets, _ = sinkhorn_assign_np(group_distances.detach().cpu().numpy(), epsilon, iters)
        target_t = torch.from_numpy(targets).to(values.device)
        losses.append(F.cross_entropy(-group_distances / temperature, target_t))
    if not losses:
        return values.new_tensor(0.0), 0
    return torch.stack(losses).mean(), collision_groups


def quantizer_soft_assignments(values, quantizer, temperature):
    """Return hard-forward, soft-backward assignments for every RQ level."""
    residual = values
    assignments = []
    confidence_losses = []
    hard_quantized = values.new_zeros(values.shape)
    for codebook in quantizer.codebooks:
        distances = (
            residual.pow(2).sum(dim=1, keepdim=True)
            - 2 * residual @ codebook.t()
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        probabilities = torch.softmax(-distances / temperature, dim=1)
        normalized_entropy = -(
            probabilities * (probabilities + 1e-8).log()
        ).sum(dim=1).mean() / math.log(codebook.shape[0])
        confidence_losses.append(normalized_entropy)
        hard = F.one_hot(distances.argmin(dim=1), codebook.shape[0]).to(values.dtype)
        assignment = probabilities + (hard - probabilities).detach()
        assignments.append(assignment)
        hard_q = hard @ codebook
        hard_quantized = hard_quantized + hard_q
        soft_q = probabilities @ codebook
        residual = residual - (soft_q + (hard_q - soft_q).detach())
    return assignments, hard_quantized, torch.stack(confidence_losses).sum()


def full_sid_collision_loss(prefix_codes, values, quantizer, temperature, topk):
    assignments, hard_quantized, confidence = quantizer_soft_assignments(
        values, quantizer, temperature
    )
    losses = []
    pair_count = 0
    for prefix in torch.unique(prefix_codes, dim=0):
        members = torch.nonzero((prefix_codes == prefix).all(dim=1), as_tuple=False).flatten()
        if members.numel() < 2:
            continue
        collision = None
        for assignment in assignments:
            level_collision = assignment[members] @ assignment[members].t()
            collision = level_collision if collision is None else collision * level_collision
        collision = collision.masked_fill(
            torch.eye(members.numel(), device=values.device, dtype=torch.bool), 0.0
        )
        k = min(max(int(topk), 1), members.numel() - 1)
        hardest = collision.topk(k, dim=1).values
        losses.append(hardest.mean())
        pair_count += int(members.numel() * k)
    if not losses:
        return values.new_tensor(0.0), hard_quantized, confidence, 0
    return torch.stack(losses).mean(), hard_quantized, confidence, pair_count


def inprefix_contrastive_loss(prefix_codes, values, temperature):
    values = F.normalize(values, p=2, dim=1)
    losses = []
    for prefix in torch.unique(prefix_codes, dim=0):
        members = torch.nonzero((prefix_codes == prefix).all(dim=1), as_tuple=False).flatten()
        if members.numel() < 2:
            continue
        logits = values[members] @ values[members].t() / temperature
        targets = torch.arange(members.numel(), device=values.device)
        losses.append(F.cross_entropy(logits, targets))
    if not losses:
        return values.new_tensor(0.0)
    return torch.stack(losses).mean()


def variance_covariance_loss(values, std_target):
    centered = values - values.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(std_target - std).mean()
    if values.shape[0] < 2:
        return variance, values.new_tensor(0.0)
    covariance = centered.t() @ centered / (values.shape[0] - 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    covariance_loss = off_diagonal.pow(2).sum() / values.shape[1]
    return variance, covariance_loss


def inprefix_topology_loss(prefix_codes, student, teacher):
    student = F.normalize(student, p=2, dim=1)
    teacher = F.normalize(teacher, p=2, dim=1)
    losses = []
    for prefix in torch.unique(prefix_codes, dim=0):
        members = torch.nonzero((prefix_codes == prefix).all(dim=1), as_tuple=False).flatten()
        if members.numel() < 2:
            continue
        student_similarity = student[members] @ student[members].t()
        with torch.no_grad():
            teacher_similarity = teacher[members] @ teacher[members].t()
        mask = ~torch.eye(members.numel(), device=student.device, dtype=torch.bool)
        losses.append(F.mse_loss(student_similarity[mask], teacher_similarity[mask]))
    if not losses:
        return student.new_tensor(0.0)
    return torch.stack(losses).mean()


def inprefix_topology_mse(prefix_codes, student, teacher):
    student = student.astype(np.float64)
    teacher = teacher.astype(np.float64)
    student /= np.maximum(np.linalg.norm(student, axis=1, keepdims=True), 1e-12)
    teacher /= np.maximum(np.linalg.norm(teacher, axis=1, keepdims=True), 1e-12)
    squared_error = 0.0
    pair_count = 0
    for prefix in np.unique(prefix_codes, axis=0):
        members = np.flatnonzero(np.all(prefix_codes == prefix, axis=1))
        if len(members) < 2:
            continue
        student_similarity = student[members] @ student[members].T
        teacher_similarity = teacher[members] @ teacher[members].T
        mask = ~np.eye(len(members), dtype=bool)
        squared_error += float(np.square(student_similarity[mask] - teacher_similarity[mask]).sum())
        pair_count += int(mask.sum())
    return squared_error / max(pair_count, 1)


def effective_rank(values):
    values = values.astype(np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular * singular
    return float((variance.sum() ** 2) / max(float(np.square(variance).sum()), 1e-12))


def vector_summary(values):
    norms = np.linalg.norm(values, axis=1)
    return {
        "norm_mean": float(norms.mean()),
        "norm_p50": float(np.median(norms)),
        "norm_p90": float(np.quantile(norms, 0.9)),
        "effective_rank": effective_rank(values),
    }


def conditional_collision_stats(prefix_codes, suffix_codes):
    prefix_counter = Counter(tuple(row) for row in prefix_codes)
    full_counter = Counter(tuple(prefix) + tuple(suffix) for prefix, suffix in zip(prefix_codes, suffix_codes))
    total_pairs = sum(count * (count - 1) // 2 for count in prefix_counter.values())
    collision_pairs = sum(count * (count - 1) // 2 for count in full_counter.values())
    collided_items = sum(count for count in full_counter.values() if count > 1)
    return {
        "unique_sid_count": len(full_counter),
        "collided_item_count": collided_items,
        "collision_rate": collided_items / max(len(prefix_codes), 1),
        "duplicate_fraction": (len(prefix_codes) - len(full_counter)) / max(len(prefix_codes), 1),
        "collision_pairs": collision_pairs,
        "within_prefix_pair_count": total_pairs,
        "within_prefix_pair_collision_rate": collision_pairs / max(total_pairs, 1),
        "max_bucket_size": max(full_counter.values()) if full_counter else 0,
    }


def task_targets(common_x, signal_x, mode, scale):
    if mode == "signal":
        return signal_x.astype(np.float32), None
    full = make_task_targets(common_x, signal_x, scale, "common_task")
    return (full - common_x).astype(np.float32), full.astype(np.float32)


def restore_shared_checkpoint(args, common_dim, prefix_sizes, device):
    shared = SharedRQVAE(
        common_dim,
        args.hidden_dim,
        args.latent_dim,
        prefix_sizes,
        args.vq_beta,
        args.dropout,
        args.balance_temperature,
    ).to(device)
    if not args.shared_checkpoint:
        return None
    checkpoint = torch.load(args.shared_checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    shared_state = {key[len("shared.") :]: value for key, value in state.items() if key.startswith("shared.")}
    shared.load_state_dict(shared_state)
    return shared


def load_teacher_deltas(
    checkpoint_path,
    shared,
    common_x,
    search_signal,
    rec_signal,
    search_target_dim,
    rec_target_dim,
    device,
    batch_size,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    params = checkpoint["params"]
    teacher = AlignedDeltaTokenizer(
        shared=shared,
        common_dim=common_x.shape[1],
        search_signal_dim=search_signal.shape[1],
        rec_signal_dim=rec_signal.shape[1],
        hidden_dim=int(params["hidden_dim"]),
        latent_dim=int(params["latent_dim"]),
        search_sizes=checkpoint["search_sizes"],
        rec_sizes=checkpoint["rec_sizes"],
        beta=float(params["vq_beta"]),
        balance_temperature=float(params["balance_temperature"]),
        lora_rank=int(params["lora_rank"]),
        lora_alpha=float(params["lora_alpha"]),
        augment_scale=float(params["augment_scale"]),
        gate_bias=float(params["gate_bias"]),
        delta_max_ratio=float(params["delta_max_ratio"]),
        search_target_dim=search_target_dim,
        rec_target_dim=rec_target_dim,
        dropout=float(params["dropout"]),
    ).to(device)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()
    search_delta, rec_delta, _, _ = encode_task_deltas(
        teacher, common_x, search_signal, rec_signal, device, batch_size
    )
    return search_delta, rec_delta


def train_task_model(
    args,
    shared,
    common_x,
    search_signal,
    rec_signal,
    search_target,
    rec_target,
    common_recon,
    rec_base_recon,
    search_full_target,
    rec_full_target,
    rec_has_full_target,
    search_teacher,
    rec_teacher,
    prefix_codes,
    search_sizes,
    rec_sizes,
    device,
):
    for parameter in shared.parameters():
        parameter.requires_grad = False
    shared.eval()
    model = AlignedDeltaTokenizer(
        shared=shared,
        common_dim=common_x.shape[1],
        search_signal_dim=search_signal.shape[1],
        rec_signal_dim=rec_signal.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        search_sizes=search_sizes,
        rec_sizes=rec_sizes,
        beta=args.vq_beta,
        balance_temperature=args.balance_temperature,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        augment_scale=args.augment_scale,
        gate_bias=args.gate_bias,
        delta_max_ratio=args.delta_max_ratio,
        search_target_dim=search_target.shape[1],
        rec_target_dim=rec_target.shape[1],
        dropout=args.dropout,
    ).to(device)
    initialize_suffix_codebooks(model, common_x, search_signal, rec_signal, device, args.seed)

    dataset = TensorDataset(
        torch.from_numpy(common_x),
        torch.from_numpy(search_signal),
        torch.from_numpy(rec_signal),
        torch.from_numpy(search_target),
        torch.from_numpy(rec_target),
        torch.from_numpy(common_recon),
        torch.from_numpy(rec_base_recon),
        torch.from_numpy(search_full_target),
        torch.from_numpy(rec_full_target),
        torch.from_numpy(prefix_codes),
        torch.from_numpy(search_teacher),
        torch.from_numpy(rec_teacher),
    )
    needs_prefix_batches = any(
        weight > 0
        for weight in (
            args.conditional_sinkhorn_weight,
            args.inprefix_contrast_weight,
            args.full_sid_collision_weight,
        )
    )
    if needs_prefix_batches:
        loader = DataLoader(
            dataset,
            batch_sampler=PrefixBatchSampler(
                prefix_codes, args.batch_size, args.prefix_group_size, args.seed + 7
            ),
            num_workers=0,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + 7),
        )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    logs = []
    global_step = 0
    adaptive_scale = 0.0 if args.adaptive_collision else 1.0
    for epoch in range(1, args.task_epochs + 1):
        model.train()
        model.shared.eval()
        totals = Counter()
        steps = 0
        for (
            common_b,
            search_b,
            rec_b,
            search_target_b,
            rec_target_b,
            common_recon_b,
            rec_base_recon_b,
            search_full_target_b,
            rec_full_target_b,
            prefix_b,
            search_teacher_b,
            rec_teacher_b,
        ) in loader:
            common_b = common_b.to(device)
            search_b = search_b.to(device)
            rec_b = rec_b.to(device)
            search_target_b = search_target_b.to(device)
            rec_target_b = rec_target_b.to(device)
            common_recon_b = common_recon_b.to(device)
            rec_base_recon_b = rec_base_recon_b.to(device)
            search_full_target_b = search_full_target_b.to(device)
            rec_full_target_b = rec_full_target_b.to(device)
            prefix_b = prefix_b.to(device)
            search_teacher_b = search_teacher_b.to(device)
            rec_teacher_b = rec_teacher_b.to(device)
            out = model(common_b, search_b, rec_b)
            task_mse = F.mse_loss(out["search_recon"], search_target_b) + F.mse_loss(
                out["rec_recon"], rec_target_b
            )
            task_cos = cosine_recon_loss(out["search_recon"], search_target_b) + cosine_recon_loss(
                out["rec_recon"], rec_target_b
            )
            search_full_recon = F.normalize(common_recon_b + out["search_recon"], p=2, dim=1)
            full_mse = F.mse_loss(search_full_recon, search_full_target_b)
            if rec_has_full_target:
                rec_full_recon = F.normalize(rec_base_recon_b + out["rec_recon"], p=2, dim=1)
                full_mse = full_mse + F.mse_loss(rec_full_recon, rec_full_target_b)
            delta_norm = out["search_delta"].pow(2).mean() + out["rec_delta"].pow(2).mean()
            apply_conditional = (
                args.conditional_sinkhorn_weight > 0
                and global_step % max(args.conditional_every, 1) == 0
            )
            if apply_conditional:
                search_sk, search_groups = collision_group_sinkhorn_loss(
                    prefix_b,
                    out["search_delta"],
                    model.search_quantizer,
                    args.conditional_sinkhorn_epsilon,
                    args.conditional_sinkhorn_iters,
                    args.conditional_sinkhorn_temperature,
                )
                rec_sk, rec_groups = collision_group_sinkhorn_loss(
                    prefix_b,
                    out["rec_delta"],
                    model.rec_quantizer,
                    args.conditional_sinkhorn_epsilon,
                    args.conditional_sinkhorn_iters,
                    args.conditional_sinkhorn_temperature,
                )
                conditional_sk = search_sk + rec_sk
            else:
                conditional_sk = common_b.new_tensor(0.0)
                search_groups = 0
                rec_groups = 0
            search_collision, _, search_confidence, search_pair_count = full_sid_collision_loss(
                prefix_b,
                out["search_delta"],
                model.search_quantizer,
                args.full_sid_temperature,
                args.full_sid_topk,
            )
            rec_collision, _, rec_confidence, rec_pair_count = full_sid_collision_loss(
                prefix_b,
                out["rec_delta"],
                model.rec_quantizer,
                args.full_sid_temperature,
                args.full_sid_topk,
            )
            full_sid_collision = search_collision + rec_collision
            assignment_confidence = search_confidence + rec_confidence
            inprefix_contrast = inprefix_contrastive_loss(
                prefix_b,
                out["search_delta"],
                args.inprefix_contrast_temperature,
            ) + inprefix_contrastive_loss(
                prefix_b,
                out["rec_delta"],
                args.inprefix_contrast_temperature,
            )
            search_variance, search_covariance = variance_covariance_loss(
                out["search_delta"], args.variance_std_target
            )
            rec_variance, rec_covariance = variance_covariance_loss(
                out["rec_delta"], args.variance_std_target
            )
            variance_loss = search_variance + rec_variance
            covariance_loss = search_covariance + rec_covariance
            gate_floor_loss = F.relu(args.task_gate_min - out["search_gate"]).mean() + F.relu(
                args.task_gate_min - out["rec_gate"]
            ).mean()
            topology_loss = inprefix_topology_loss(
                prefix_b, out["search_delta"], search_teacher_b
            ) + inprefix_topology_loss(prefix_b, out["rec_delta"], rec_teacher_b)
            adaptive_weight = adaptive_scale if args.adaptive_collision else 1.0
            loss = (
                args.task_mse_weight * task_mse
                + args.task_cos_weight * task_cos
                + args.full_mse_weight * full_mse
                + args.commit_weight * out["commit"]
                + args.balance_weight * out["balance"]
                + args.delta_norm_weight * delta_norm
                + args.conditional_sinkhorn_weight * conditional_sk
                + adaptive_weight * args.inprefix_contrast_weight * inprefix_contrast
                + adaptive_weight * args.full_sid_collision_weight * full_sid_collision
                + adaptive_weight * args.assignment_confidence_weight * assignment_confidence
                + adaptive_weight * args.variance_weight * variance_loss
                + adaptive_weight * args.covariance_weight * covariance_loss
                + args.task_gate_floor_weight * gate_floor_loss
                + args.topology_weight * topology_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            global_step += 1
            totals["loss"] += float(loss.detach().cpu())
            totals["task_mse"] += float(task_mse.detach().cpu())
            totals["task_cos"] += float(task_cos.detach().cpu())
            totals["full_mse"] += float(full_mse.detach().cpu())
            totals["commit"] += float(out["commit"].detach().cpu())
            totals["balance"] += float(out["balance"].detach().cpu())
            totals["delta_norm"] += float(delta_norm.detach().cpu())
            totals["conditional_sinkhorn"] += float(conditional_sk.detach().cpu())
            totals["inprefix_contrast"] += float(inprefix_contrast.detach().cpu())
            totals["full_sid_collision"] += float(full_sid_collision.detach().cpu())
            totals["assignment_confidence"] += float(assignment_confidence.detach().cpu())
            totals["variance"] += float(variance_loss.detach().cpu())
            totals["covariance"] += float(covariance_loss.detach().cpu())
            totals["gate_floor"] += float(gate_floor_loss.detach().cpu())
            totals["topology"] += float(topology_loss.detach().cpu())
            totals["adaptive_scale"] += float(adaptive_weight)
            totals["full_sid_pairs"] += search_pair_count + rec_pair_count
            totals["search_collision_groups"] += search_groups
            totals["rec_collision_groups"] += rec_groups
            totals["search_gate"] += float(out["search_gate"].mean().detach().cpu())
            totals["rec_gate"] += float(out["rec_gate"].mean().detach().cpu())
            steps += 1
        row = {"epoch": epoch, **{key: value / max(steps, 1) for key, value in totals.items()}}
        if args.adaptive_collision:
            search_eval, rec_eval, _, _ = encode_task_deltas(
                model, common_x, search_signal, rec_signal, device, args.batch_size
            )
            search_codes = np.asarray(
                rq_assign_codes_np(search_eval, model.search_quantizer.codebooks, "argmin"),
                dtype=np.int64,
            )
            rec_codes = np.asarray(
                rq_assign_codes_np(rec_eval, model.rec_quantizer.codebooks, "argmin"),
                dtype=np.int64,
            )
            search_duplicate = conditional_collision_stats(prefix_codes, search_codes)[
                "duplicate_fraction"
            ]
            rec_duplicate = conditional_collision_stats(prefix_codes, rec_codes)[
                "duplicate_fraction"
            ]
            observed_duplicate = max(search_duplicate, rec_duplicate)
            if epoch >= args.collision_warmup_epochs:
                adaptive_scale = float(
                    np.clip(
                        adaptive_scale
                        + args.collision_controller_gain
                        * (observed_duplicate - args.collision_target_duplicate),
                        0.0,
                        1.0,
                    )
                )
            row.update(
                {
                    "search_duplicate": search_duplicate,
                    "rec_duplicate": rec_duplicate,
                    "observed_duplicate": observed_duplicate,
                    "next_adaptive_scale": adaptive_scale,
                }
            )
        logs.append(row)
        if epoch == 1 or epoch == args.task_epochs or epoch % args.log_every == 0:
            print(json.dumps({"stage": "task", **row}))
    return model, logs


@torch.no_grad()
def encode_task_deltas(model, common_x, search_signal, rec_signal, device, batch_size):
    search_values = []
    rec_values = []
    search_gates = []
    rec_gates = []
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(common_x), torch.from_numpy(search_signal), torch.from_numpy(rec_signal)
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    for common_b, search_b, rec_b in loader:
        common_b = common_b.to(device)
        search_b = search_b.to(device)
        rec_b = rec_b.to(device)
        z_shared = model.shared.encoder(common_b)
        search_delta, rec_delta, search_gate, rec_gate = model.task_deltas(
            z_shared, common_b, search_b, rec_b
        )
        search_values.append(search_delta.cpu().numpy())
        rec_values.append(rec_delta.cpu().numpy())
        search_gates.append(search_gate.cpu().numpy())
        rec_gates.append(rec_gate.cpu().numpy())
    return (
        np.concatenate(search_values).astype(np.float32),
        np.concatenate(rec_values).astype(np.float32),
        np.concatenate(search_gates).astype(np.float32),
        np.concatenate(rec_gates).astype(np.float32),
    )


def build(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_sizes = parse_sizes(args.prefix_codebook_sizes)
    search_sizes = parse_sizes(args.search_codebook_sizes)
    rec_sizes = parse_sizes(args.rec_codebook_sizes)

    item_ids, item_rows, has_metadata = load_item_rows(data_dir, args.item_ids, args.max_items)
    query_docs, query_counter = aggregate_search_queries(data_dir, item_ids, args.max_queries_per_item)
    rec_docs, neighbor_counter = aggregate_rec_neighbors(
        data_dir, item_ids, args.neighbor_window, args.max_neighbors_per_item
    )
    text_docs, _, _ = make_docs(item_ids, item_rows, query_docs, rec_docs)
    cache_dir = Path(args.embedding_cache_dir) if args.embedding_cache_dir else data_dir / "text_embedding_cache"
    if args.common_embedding:
        common_x = load_precomputed_embedding(args.common_embedding, item_ids, "common")
    else:
        common_x = encode_text_docs(args, text_docs, "common", device, cache_dir)
    args.text_embedding_dim = int(common_x.shape[1])
    search_signal, search_signal_stats = build_search_signal_embeddings(
        args, item_ids, query_counter, device, cache_dir
    )
    if args.rec_signal == "behavior":
        if not args.rec_behavior_embedding:
            raise ValueError("--rec-behavior-embedding is required for --rec-signal behavior")
        rec_signal, rec_signal_stats = load_behavior_signal(args.rec_behavior_embedding, item_ids)
        rec_target_mode = "signal"
    else:
        rec_signal, rec_signal_stats = build_rec_signal_embeddings(
            item_ids, common_x, neighbor_counter, args.max_neighbors_per_item
        )
        rec_signal_stats = {"mode": "text_neighbor", **rec_signal_stats}
        rec_target_mode = "delta"

    search_target, search_full_target = task_targets(
        common_x, search_signal, "delta", args.task_target_scale
    )
    rec_target, rec_full_target = task_targets(
        common_x, rec_signal, rec_target_mode, args.task_target_scale
    )

    shared = restore_shared_checkpoint(args, common_x.shape[1], prefix_sizes, device)
    if shared is None:
        shared, shared_logs = train_shared(args, common_x.astype(np.float32), prefix_sizes, device)
    else:
        shared_logs = [{"source": args.shared_checkpoint}]
    shared.eval()
    prefix_codes, q_shared, z_shared = fixed_prefix(shared, common_x, args.prefix_assignment)
    with torch.no_grad():
        common_recon = (
            shared.decoder(torch.from_numpy(q_shared).to(device)).cpu().numpy().astype(np.float32)
        )
    rec_has_full_target = rec_full_target is not None
    if rec_full_target is None:
        rec_full_target = np.zeros_like(common_x, dtype=np.float32)
    if args.topology_weight > 0 and args.topology_teacher_mode == "checkpoint":
        if not args.topology_teacher_checkpoint:
            raise ValueError(
                "--topology-teacher-checkpoint is required when --topology-weight > 0"
            )
        search_teacher, rec_teacher = load_teacher_deltas(
            args.topology_teacher_checkpoint,
            shared,
            common_x.astype(np.float32),
            search_signal,
            rec_signal,
            search_target.shape[1],
            rec_target.shape[1],
            device,
            args.batch_size,
        )
    elif args.topology_weight > 0:
        # Preserve neighborhoods from the representation views being quantized.
        search_teacher = search_target.astype(np.float32)
        rec_teacher = rec_signal.astype(np.float32)
    else:
        search_teacher = np.zeros((len(common_x), args.latent_dim), dtype=np.float32)
        rec_teacher = np.zeros((len(common_x), args.latent_dim), dtype=np.float32)
    model, task_logs = train_task_model(
        args,
        shared,
        common_x.astype(np.float32),
        search_signal,
        rec_signal,
        search_target,
        rec_target,
        common_recon,
        common_recon,
        search_full_target,
        rec_full_target,
        rec_has_full_target,
        search_teacher,
        rec_teacher,
        prefix_codes,
        search_sizes,
        rec_sizes,
        device,
    )
    search_delta, rec_delta, search_gate, rec_gate = encode_task_deltas(
        model, common_x, search_signal, rec_signal, device, args.batch_size
    )
    raw_search_codes = np.asarray(
        rq_assign_codes_np(search_delta, model.search_quantizer.codebooks, "argmin"), dtype=np.int64
    )
    raw_rec_codes = np.asarray(
        rq_assign_codes_np(rec_delta, model.rec_quantizer.codebooks, "argmin"), dtype=np.int64
    )
    search_codes = raw_search_codes.copy()
    rec_codes = raw_rec_codes.copy()
    refinement = {}
    if args.export_sinkhorn_rounds > 0:
        search_codes, search_refine = refine_last_suffix_layer(
            prefix_codes,
            search_codes,
            search_delta,
            model.search_quantizer.codebooks,
            args.conditional_sinkhorn_epsilon,
            args.conditional_sinkhorn_iters,
            args.export_sinkhorn_rounds,
        )
        rec_codes, rec_refine = refine_last_suffix_layer(
            prefix_codes,
            rec_codes,
            rec_delta,
            model.rec_quantizer.codebooks,
            args.conditional_sinkhorn_epsilon,
            args.conditional_sinkhorn_iters,
            args.export_sinkhorn_rounds,
        )
        search_codes = np.asarray(search_codes, dtype=np.int64)
        rec_codes = np.asarray(rec_codes, dtype=np.int64)
        refinement = {"search": search_refine, "rec": rec_refine}

    mapping = {}
    search_sid_by_item = {}
    rec_sid_by_item = {}
    for item_id, prefix_code, search_code, rec_code in zip(
        item_ids, prefix_codes, search_codes, rec_codes
    ):
        prefix = token_list(prefix_code, "P")
        search_suffix = token_list(search_code, "S")
        rec_suffix = token_list(rec_code, "R")
        mapping[item_id] = {
            "prefix": prefix,
            "search_sid": prefix + search_suffix,
            "rec_sid": prefix + rec_suffix,
            "search_suffix": search_suffix,
            "rec_suffix": rec_suffix,
            "prefix_codes": prefix_code.tolist(),
            "search_suffix_codes": search_code.tolist(),
            "rec_suffix_codes": rec_code.tolist(),
        }
        search_sid_by_item[item_id] = prefix + search_suffix
        rec_sid_by_item[item_id] = prefix + rec_suffix

    mapping_path = output_dir / "sid_mapping.v3_aligned_delta.json"
    write_json(mapping_path, mapping)
    write_json(output_dir / "sid_mapping.v2.json", mapping)
    write_json(output_dir / "sid_mapping.v1.json", mapping)
    write_json(output_dir / "shared_train_log.json", shared_logs)
    write_json(output_dir / "task_train_log.json", task_logs)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "params": vars(args),
            "prefix_sizes": prefix_sizes,
            "search_sizes": search_sizes,
            "rec_sizes": rec_sizes,
            "fixed_prefix_codes": prefix_codes,
        },
        output_dir / "aligned_delta_sid_model.pt",
    )

    prefix_labels = ["_".join(map(str, row)) for row in prefix_codes]
    search_labels = ["_".join(map(str, row)) for row in search_codes]
    rec_labels = ["_".join(map(str, row)) for row in rec_codes]
    prefix_counter = Counter(prefix_labels)
    stats = {
        "method": (
            "M7b-v6 BGE-UniSAR Representation-Enhanced LoRA PLC"
            if args.common_embedding and args.rec_signal == "behavior"
            else "M7b-v5 Adaptive Collision + Topology Distillation"
            if args.adaptive_collision or args.topology_weight > 0
            else "M7b-v4 In-Prefix Differentiable Discrimination"
            if any(
                weight > 0
                for weight in (
                    args.inprefix_contrast_weight,
                    args.full_sid_collision_weight,
                    args.variance_weight,
                    args.covariance_weight,
                )
            )
            else "M7b-v3 Aligned-LoRA-Delta"
        ),
        "num_items": len(item_ids),
        "mapping_path": str(mapping_path),
        "has_metadata": has_metadata,
        "field_coverage": field_coverage(item_rows),
        "codebook_sizes": {
            "prefix": prefix_sizes,
            "search_suffix": search_sizes,
            "rec_suffix": rec_sizes,
        },
        "prefix": {
            "assignment": args.prefix_assignment,
            "unique_count": len(prefix_counter),
            "bucket_size": count_summary(list(prefix_counter.values())),
            "usage": code_usage(prefix_codes, prefix_sizes),
            "training_export_changed_rate": 0.0,
        },
        "signals": {
            "search": {**search_signal_stats, **vector_summary(search_signal)},
            "rec": {**rec_signal_stats, **vector_summary(rec_signal)},
            "search_target_mode": "delta_from_common",
            "rec_target_mode": rec_target_mode,
            "search_target": vector_summary(search_target),
            "rec_target": vector_summary(rec_target),
        },
        "latent": {
            "z_shared": vector_summary(z_shared),
            "prefix_quantization_error": vector_summary(z_shared - q_shared),
            "search_delta": vector_summary(search_delta),
            "rec_delta": vector_summary(rec_delta),
            "search_delta_to_shared_norm_ratio": float(
                np.mean(np.linalg.norm(search_delta, axis=1) / np.maximum(np.linalg.norm(z_shared, axis=1), 1e-12))
            ),
            "rec_delta_to_shared_norm_ratio": float(
                np.mean(np.linalg.norm(rec_delta, axis=1) / np.maximum(np.linalg.norm(z_shared, axis=1), 1e-12))
            ),
            "search_gate_mean": float(search_gate.mean()),
            "rec_gate_mean": float(rec_gate.mean()),
        },
        "topology_distillation": {
            "teacher_mode": args.topology_teacher_mode,
            "teacher_checkpoint": args.topology_teacher_checkpoint or None,
            "weight": args.topology_weight,
            "search_inprefix_pairwise_mse": (
                inprefix_topology_mse(prefix_codes, search_delta, search_teacher)
                if args.topology_weight > 0
                else None
            ),
            "rec_inprefix_pairwise_mse": (
                inprefix_topology_mse(prefix_codes, rec_delta, rec_teacher)
                if args.topology_weight > 0
                else None
            ),
        },
        "raw_assignment": {
            "search_collision": conditional_collision_stats(prefix_codes, raw_search_codes),
            "rec_collision": conditional_collision_stats(prefix_codes, raw_rec_codes),
            "search_suffix_usage": code_usage(raw_search_codes, search_sizes),
            "rec_suffix_usage": code_usage(raw_rec_codes, rec_sizes),
        },
        "final_assignment": {
            "search_collision": conditional_collision_stats(prefix_codes, search_codes),
            "rec_collision": conditional_collision_stats(prefix_codes, rec_codes),
            "search_suffix_usage": code_usage(search_codes, search_sizes),
            "rec_suffix_usage": code_usage(rec_codes, rec_sizes),
            "search_changed_from_raw": float(np.any(search_codes != raw_search_codes, axis=1).mean()),
            "rec_changed_from_raw": float(np.any(rec_codes != raw_rec_codes, axis=1).mean()),
            "prefix_search_suffix_nmi": normalized_mutual_info_score(prefix_labels, search_labels),
            "prefix_rec_suffix_nmi": normalized_mutual_info_score(prefix_labels, rec_labels),
        },
        "legacy_collision": {
            "search": collision_rate(search_sid_by_item),
            "rec": collision_rate(rec_sid_by_item),
        },
        "refinement": refinement,
        "final_task_train_log": task_logs[-1],
        "params": vars(args),
    }
    write_json(output_dir / "sid_stats.v3_aligned_delta.json", stats)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/cjk/Search_Reconmendation/stage3/data/gensar_lite_5k")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--item-ids", default="")
    parser.add_argument("--max-items", type=int, default=5000)
    parser.add_argument("--prefix-codebook-sizes", default="64,64")
    parser.add_argument("--search-codebook-sizes", default="64,64")
    parser.add_argument("--rec-codebook-sizes", default="64,64")
    parser.add_argument("--prefix-assignment", choices=["argmin", "balanced"], default="balanced")
    parser.add_argument("--shared-checkpoint", default="")
    parser.add_argument("--rec-signal", choices=["neighbor", "behavior"], default="neighbor")
    parser.add_argument("--rec-behavior-embedding", default="")
    parser.add_argument("--common-embedding", default="")
    parser.add_argument("--max-queries-per-item", type=int, default=20)
    parser.add_argument("--neighbor-window", type=int, default=2)
    parser.add_argument("--max-neighbors-per-item", type=int, default=200)
    parser.add_argument("--text-encoder-model", default="intfloat/e5-small-v2")
    parser.add_argument("--text-encoder-prefix", default="passage: ")
    parser.add_argument("--query-encoder-prefix", default="query: ")
    parser.add_argument("--text-encoder-max-length", type=int, default=256)
    parser.add_argument("--text-encoder-batch-size", type=int, default=128)
    parser.add_argument("--text-encoder-pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--embedding-cache-dir", default="")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--augment-scale", type=float, default=0.25)
    parser.add_argument("--gate-bias", type=float, default=-2.0)
    parser.add_argument("--delta-max-ratio", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--shared-epochs", type=int, default=50)
    parser.add_argument("--task-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--vq-beta", type=float, default=0.25)
    parser.add_argument("--balance-temperature", type=float, default=0.05)
    parser.add_argument("--task-target-scale", type=float, default=0.5)
    parser.add_argument("--task-mse-weight", type=float, default=10.0)
    parser.add_argument("--task-cos-weight", type=float, default=0.1)
    parser.add_argument("--full-mse-weight", type=float, default=1.0)
    parser.add_argument("--commit-weight", type=float, default=0.05)
    parser.add_argument("--balance-weight", type=float, default=0.02)
    parser.add_argument("--delta-norm-weight", type=float, default=0.001)
    parser.add_argument("--prefix-sk-epsilons", default="")
    parser.add_argument("--sk-iters", type=int, default=100)
    parser.add_argument("--init-codebooks", choices=["kmeans", "random"], default="kmeans")
    parser.add_argument("--conditional-sinkhorn-weight", type=float, default=0.0)
    parser.add_argument("--conditional-sinkhorn-epsilon", type=float, default=0.03)
    parser.add_argument("--conditional-sinkhorn-iters", type=int, default=50)
    parser.add_argument("--conditional-sinkhorn-temperature", type=float, default=0.1)
    parser.add_argument("--conditional-every", type=int, default=1)
    parser.add_argument("--prefix-group-size", type=int, default=16)
    parser.add_argument("--inprefix-contrast-weight", type=float, default=0.0)
    parser.add_argument("--inprefix-contrast-temperature", type=float, default=0.1)
    parser.add_argument("--full-sid-collision-weight", type=float, default=0.0)
    parser.add_argument("--full-sid-temperature", type=float, default=0.05)
    parser.add_argument("--full-sid-topk", type=int, default=4)
    parser.add_argument("--assignment-confidence-weight", type=float, default=0.0)
    parser.add_argument("--variance-weight", type=float, default=0.0)
    parser.add_argument("--variance-std-target", type=float, default=0.08)
    parser.add_argument("--covariance-weight", type=float, default=0.0)
    parser.add_argument("--task-gate-min", type=float, default=0.0)
    parser.add_argument("--task-gate-floor-weight", type=float, default=0.0)
    parser.add_argument("--topology-teacher-checkpoint", default="")
    parser.add_argument(
        "--topology-teacher-mode", choices=["checkpoint", "signal"], default="checkpoint"
    )
    parser.add_argument("--topology-weight", type=float, default=0.0)
    parser.add_argument("--adaptive-collision", action="store_true")
    parser.add_argument("--collision-warmup-epochs", type=int, default=10)
    parser.add_argument("--collision-target-duplicate", type=float, default=0.08)
    parser.add_argument("--collision-controller-gain", type=float, default=2.0)
    parser.add_argument("--export-sinkhorn-rounds", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    stats = build(args)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
