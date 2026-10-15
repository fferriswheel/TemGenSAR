#!/usr/bin/env python3
import argparse
import hashlib
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
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

from build_neural_rq_sid_v2 import (
    MLP,
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


class SharedRQVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        prefix_sizes,
        beta,
        dropout,
        balance_temperature,
        sk_epsilons=None,
        sk_iters=100,
    ):
        super().__init__()
        self.encoder = MLP(input_dim, hidden_dim, latent_dim, dropout)
        self.quantizer = ResidualVectorQuantizer(
            prefix_sizes,
            latent_dim,
            beta,
            balance_temperature,
            sk_epsilons=sk_epsilons,
            sk_iters=sk_iters,
        )
        self.decoder = MLP(latent_dim, hidden_dim, input_dim, dropout)

    def forward(self, x):
        z = self.encoder(x)
        q, indices, commit, balance = self.quantizer(z)
        recon = self.decoder(q)
        return {"z": z, "q": q, "indices": indices, "commit": commit, "balance": balance, "recon": recon}


@torch.no_grad()
def encode_text_docs(args, docs, tag, device, cache_dir, prefix=None):
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.text_encoder_prefix if prefix is None else prefix
    h = hashlib.sha1()
    h.update(args.text_encoder_model.encode("utf-8"))
    h.update(str(args.text_encoder_max_length).encode("utf-8"))
    h.update(prefix.encode("utf-8"))
    h.update(tag.encode("utf-8"))
    for doc in docs:
        h.update(doc.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    cache_path = cache_dir / f"{tag}.{h.hexdigest()[:16]}.npy"
    if cache_path.exists():
        return np.load(cache_path).astype(np.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_model)
    model = AutoModel.from_pretrained(args.text_encoder_model).to(device)
    model.eval()
    vectors = []
    for start in range(0, len(docs), args.text_encoder_batch_size):
        batch_docs = docs[start : start + args.text_encoder_batch_size]
        texts = [f"{prefix}{doc}" for doc in batch_docs]
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.text_encoder_max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        out = model(**batch)
        hidden = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = F.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.cpu().numpy().astype(np.float32))
    x = np.concatenate(vectors, axis=0).astype(np.float32)
    np.save(cache_path, x)
    return x


def l2_normalize_np(x):
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(denom, 1e-12)).astype(np.float32)


def build_search_signal_embeddings(args, item_ids, query_counter, device, cache_dir):
    query_to_idx = {}
    queries = []
    item_query_weights = {}
    for item_id in item_ids:
        rows = query_counter[item_id].most_common(args.max_queries_per_item)
        item_query_weights[item_id] = rows
        for query, _ in rows:
            if query not in query_to_idx:
                query_to_idx[query] = len(queries)
                queries.append(query)

    if queries:
        query_x = encode_text_docs(args, queries, "m7_query_signal", device, cache_dir, prefix=args.query_encoder_prefix)
    else:
        query_x = np.zeros((0, args.text_embedding_dim), dtype=np.float32)

    out = np.zeros((len(item_ids), query_x.shape[1] if len(query_x) else args.text_embedding_dim), dtype=np.float32)
    for row_idx, item_id in enumerate(item_ids):
        weighted = []
        weights = []
        for query, count in item_query_weights[item_id]:
            weighted.append(query_x[query_to_idx[query]])
            weights.append(float(count))
        if weighted:
            w = np.asarray(weights, dtype=np.float32)
            w = w / max(float(w.sum()), 1e-12)
            out[row_idx] = (np.stack(weighted, axis=0) * w[:, None]).sum(axis=0)
    return l2_normalize_np(out), {"unique_queries": len(queries)}


def build_rec_signal_embeddings(item_ids, common_x, neighbor_counter, max_neighbors_per_item):
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    out = np.zeros_like(common_x, dtype=np.float32)
    used_counts = []
    dropped_counts = []
    for row_idx, item_id in enumerate(item_ids):
        weighted = []
        weights = []
        dropped = 0
        for neighbor, count in neighbor_counter[item_id].most_common(max_neighbors_per_item):
            if neighbor not in item_to_idx:
                dropped += int(count)
                continue
            weighted.append(common_x[item_to_idx[neighbor]])
            weights.append(float(count))
        used_counts.append(int(sum(weights)))
        dropped_counts.append(dropped)
        if weighted:
            w = np.asarray(weights, dtype=np.float32)
            w = w / max(float(w.sum()), 1e-12)
            out[row_idx] = (np.stack(weighted, axis=0) * w[:, None]).sum(axis=0)
    return l2_normalize_np(out), {"used_neighbor_count": count_summary(used_counts), "dropped_neighbor_count": count_summary(dropped_counts)}


def build_rec_transition_embeddings(data_dir, item_ids, common_x, max_next_per_item):
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    next_counter = defaultdict(Counter)
    cur_user = None
    cur_rows = []
    for row in read_jsonl(Path(data_dir) / "rec_interactions_chrono.jsonl"):
        user_id = row["user_id"]
        if cur_user is None:
            cur_user = user_id
        if user_id != cur_user:
            update_transition_counts(cur_rows, set(item_ids), next_counter)
            cur_user = user_id
            cur_rows = []
        cur_rows.append(row)
    if cur_rows:
        update_transition_counts(cur_rows, set(item_ids), next_counter)

    out = np.zeros_like(common_x, dtype=np.float32)
    used_counts = []
    dropped_counts = []
    for row_idx, item_id in enumerate(item_ids):
        weighted = []
        weights = []
        dropped = 0
        for next_item, count in next_counter[item_id].most_common(max_next_per_item):
            if next_item not in item_to_idx:
                dropped += int(count)
                continue
            weighted.append(common_x[item_to_idx[next_item]])
            weights.append(float(count))
        used_counts.append(int(sum(weights)))
        dropped_counts.append(dropped)
        if weighted:
            w = np.asarray(weights, dtype=np.float32)
            w = w / max(float(w.sum()), 1e-12)
            out[row_idx] = (np.stack(weighted, axis=0) * w[:, None]).sum(axis=0)
    return l2_normalize_np(out), {"used_next_count": count_summary(used_counts), "dropped_next_count": count_summary(dropped_counts)}


def update_transition_counts(user_rows, item_set, counters):
    rows = sorted(user_rows, key=lambda x: (x.get("timestamp", 0), x.get("position", 0), x["item_id"]))
    for pos in range(len(rows) - 1):
        item_id = rows[pos]["item_id"]
        next_item = rows[pos + 1]["item_id"]
        if item_id in item_set:
            counters[item_id][next_item] += 1


def make_task_targets(common_x, task_signal_x, scale, mode):
    if mode == "common_task":
        return l2_normalize_np(common_x + scale * task_signal_x)
    if mode == "task_signal":
        return task_signal_x.astype(np.float32)
    if mode == "delta_from_common":
        return l2_normalize_np(l2_normalize_np(common_x + scale * task_signal_x) - common_x)
    raise ValueError(f"unknown suffix target mode: {mode}")


def parse_float_sizes(value):
    if value is None or value == "":
        return None
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


class LoRAEncoder(nn.Module):
    def __init__(self, base_encoder, rank, alpha):
        super().__init__()
        self.base = base_encoder
        self.rank = rank
        self.scale = alpha / max(rank, 1)
        linear1 = self.base.net[0]
        linear2 = self.base.net[4]
        self.lora1_a = nn.Linear(linear1.in_features, rank, bias=False)
        self.lora1_b = nn.Linear(rank, linear1.out_features, bias=False)
        self.lora2_a = nn.Linear(linear2.in_features, rank, bias=False)
        self.lora2_b = nn.Linear(rank, linear2.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora1_a.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora2_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora1_b.weight)
        nn.init.zeros_(self.lora2_b.weight)

    def forward(self, x):
        linear1, norm, act, dropout, linear2 = self.base.net
        h = linear1(x) + self.scale * self.lora1_b(self.lora1_a(x))
        h = norm(h)
        h = act(h)
        h = dropout(h)
        z = linear2(h) + self.scale * self.lora2_b(self.lora2_a(h))
        return z


class GatedEmbeddingFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.common_proj = nn.Linear(input_dim, input_dim)
        self.task_proj = nn.Linear(input_dim, input_dim)
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x_common, x_task_signal):
        features = torch.cat(
            [x_common, x_task_signal, x_common * x_task_signal, (x_common - x_task_signal).abs()],
            dim=1,
        )
        gate = self.gate(features)
        return self.norm(self.common_proj(x_common) + gate * self.task_proj(x_task_signal))


class TextEncoderFusionPLCTokenizer(nn.Module):
    def __init__(
        self,
        shared_model,
        task_input_dim,
        hidden_dim,
        latent_dim,
        search_sizes,
        rec_sizes,
        beta,
        dropout,
        balance_temperature,
        adapter_scale,
        task_encoder,
        lora_rank,
        lora_alpha,
        suffix_decode_input,
        suffix_sk_epsilons=None,
        sk_iters=100,
    ):
        super().__init__()
        self.shared = shared_model
        self.adapter_scale = adapter_scale
        self.task_encoder = task_encoder
        self.suffix_decode_input = suffix_decode_input
        shared_input_dim = self.shared.encoder.net[0].in_features
        self.search_fusion = GatedEmbeddingFusion(shared_input_dim, hidden_dim, dropout)
        self.rec_fusion = GatedEmbeddingFusion(shared_input_dim, hidden_dim, dropout)
        if task_encoder == "adapter":
            self.search_adapter = MLP(shared_input_dim, hidden_dim, latent_dim, dropout)
            self.rec_adapter = MLP(shared_input_dim, hidden_dim, latent_dim, dropout)
        elif task_encoder == "lora":
            if task_input_dim != shared_input_dim:
                raise ValueError(f"LoRA task encoder requires task_input_dim={shared_input_dim}, got {task_input_dim}")
            self.search_encoder = LoRAEncoder(self.shared.encoder, lora_rank, lora_alpha)
            self.rec_encoder = LoRAEncoder(self.shared.encoder, lora_rank, lora_alpha)
        else:
            raise ValueError(f"unknown task encoder: {task_encoder}")
        self.search_quantizer = ResidualVectorQuantizer(
            search_sizes,
            latent_dim,
            beta,
            balance_temperature,
            sk_epsilons=suffix_sk_epsilons,
            sk_iters=sk_iters,
        )
        self.rec_quantizer = ResidualVectorQuantizer(
            rec_sizes,
            latent_dim,
            beta,
            balance_temperature,
            sk_epsilons=suffix_sk_epsilons,
            sk_iters=sk_iters,
        )
        self.search_decoder = MLP(latent_dim, hidden_dim, task_input_dim, dropout)
        self.rec_decoder = MLP(latent_dim, hidden_dim, task_input_dim, dropout)

    def shared_codes(self, x_common):
        z_shared = self.shared.encoder(x_common)
        q_shared, prefix_idx, _, _ = self.shared.quantizer(z_shared)
        return z_shared, q_shared, prefix_idx

    def task_latents(self, z_shared, x_common, x_search_signal, x_rec_signal):
        x_search_fused = self.search_fusion(x_common, x_search_signal)
        x_rec_fused = self.rec_fusion(x_common, x_rec_signal)
        if self.task_encoder == "adapter":
            z_search0 = z_shared + self.adapter_scale * self.search_adapter(x_search_fused)
            z_rec0 = z_shared + self.adapter_scale * self.rec_adapter(x_rec_fused)
        else:
            z_search0 = self.search_encoder(x_search_fused)
            z_rec0 = self.rec_encoder(x_rec_fused)
        return z_search0, z_rec0

    def forward(self, x_common, x_search_signal, x_rec_signal):
        with torch.no_grad():
            z_shared, q_shared, prefix_idx = self.shared_codes(x_common)
        z_search0, z_rec0 = self.task_latents(z_shared, x_common, x_search_signal, x_rec_signal)
        r_search = z_search0 - q_shared
        r_rec = z_rec0 - q_shared

        q_search, search_idx, search_commit, search_balance = self.search_quantizer(r_search)
        q_rec, rec_idx, rec_commit, rec_balance = self.rec_quantizer(r_rec)
        if self.suffix_decode_input == "full":
            search_decode_input = q_shared + q_search
            rec_decode_input = q_shared + q_rec
        elif self.suffix_decode_input == "suffix":
            search_decode_input = q_search
            rec_decode_input = q_rec
        else:
            raise ValueError(f"unknown suffix decode input: {self.suffix_decode_input}")
        search_recon = self.search_decoder(search_decode_input)
        rec_recon = self.rec_decoder(rec_decode_input)
        return {
            "q_shared": q_shared,
            "z_shared": z_shared,
            "prefix_idx": prefix_idx,
            "z_search0": z_search0,
            "z_rec0": z_rec0,
            "r_search": r_search,
            "r_rec": r_rec,
            "q_search": q_search,
            "q_rec": q_rec,
            "search_idx": search_idx,
            "rec_idx": rec_idx,
            "search_recon": search_recon,
            "rec_recon": rec_recon,
            "commit": search_commit + rec_commit,
            "balance": search_balance + rec_balance,
        }


def train_shared(args, x_common, prefix_sizes, device):
    dataset = TensorDataset(torch.from_numpy(x_common))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = SharedRQVAE(
        input_dim=x_common.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        prefix_sizes=prefix_sizes,
        beta=args.vq_beta,
        dropout=args.dropout,
        balance_temperature=args.balance_temperature,
        sk_epsilons=parse_float_sizes(args.prefix_sk_epsilons),
        sk_iters=args.sk_iters,
    ).to(device)
    if args.init_codebooks == "kmeans":
        with torch.no_grad():
            z = model.encoder(torch.from_numpy(x_common).to(device)).cpu().numpy()
        init_rq_codebooks(model.quantizer, z, args.seed)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logs = []
    for epoch in range(1, args.shared_epochs + 1):
        model.train()
        totals = Counter()
        steps = 0
        for (batch_common,) in loader:
            batch_common = batch_common.to(device)
            out = model(batch_common)
            recon = F.mse_loss(out["recon"], batch_common)
            loss = recon + args.commit_weight * out["commit"] + args.balance_weight * out["balance"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["recon"] += float(recon.detach().cpu())
            totals["commit"] += float(out["commit"].detach().cpu())
            totals["balance"] += float(out["balance"].detach().cpu())
            steps += 1
        row = {"epoch": epoch, **{k: v / max(steps, 1) for k, v in totals.items()}}
        logs.append(row)
        if epoch == 1 or epoch == args.shared_epochs or epoch % args.log_every == 0:
            print(json.dumps({"stage": "shared", **row}, ensure_ascii=False))
    return model, logs


@torch.no_grad()
def init_suffix_codebooks(model, x_common, x_search_signal, x_rec_signal, device, seed):
    model.eval()
    x_common_t = torch.from_numpy(x_common).to(device)
    x_search_t = torch.from_numpy(x_search_signal).to(device)
    x_rec_t = torch.from_numpy(x_rec_signal).to(device)
    z_shared, q_shared, _ = model.shared_codes(x_common_t)
    z_search0, z_rec0 = model.task_latents(z_shared, x_common_t, x_search_t, x_rec_t)
    r_search = z_search0 - q_shared
    r_rec = z_rec0 - q_shared
    init_rq_codebooks(model.search_quantizer, r_search.cpu().numpy(), seed + 1000)
    init_rq_codebooks(model.rec_quantizer, r_rec.cpu().numpy(), seed + 2000)


def train_task(
    args,
    shared_model,
    x_common,
    x_search_signal,
    x_rec_signal,
    x_search_target,
    x_rec_target,
    search_sizes,
    rec_sizes,
    device,
):
    for param in shared_model.parameters():
        param.requires_grad = False
    shared_model.eval()

    dataset = TensorDataset(
        torch.from_numpy(x_common),
        torch.from_numpy(x_search_signal),
        torch.from_numpy(x_rec_signal),
        torch.from_numpy(x_search_target),
        torch.from_numpy(x_rec_target),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 7),
    )
    model = TextEncoderFusionPLCTokenizer(
        shared_model=shared_model,
        task_input_dim=x_search_signal.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        search_sizes=search_sizes,
        rec_sizes=rec_sizes,
        beta=args.vq_beta,
        dropout=args.dropout,
        balance_temperature=args.balance_temperature,
        adapter_scale=args.adapter_scale,
        task_encoder=args.task_encoder,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        suffix_decode_input=args.suffix_decode_input,
        suffix_sk_epsilons=parse_float_sizes(args.suffix_sk_epsilons),
        sk_iters=args.sk_iters,
    ).to(device)
    if args.init_codebooks == "kmeans":
        init_suffix_codebooks(model, x_common, x_search_signal, x_rec_signal, device, args.seed)

    trainable = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    logs = []
    for epoch in range(1, args.task_epochs + 1):
        model.train()
        model.shared.eval()
        totals = Counter()
        steps = 0
        for batch_common, batch_search_signal, batch_rec_signal, batch_search_target, batch_rec_target in loader:
            batch_common = batch_common.to(device)
            batch_search_signal = batch_search_signal.to(device)
            batch_rec_signal = batch_rec_signal.to(device)
            batch_search_target = batch_search_target.to(device)
            batch_rec_target = batch_rec_target.to(device)
            out = model(batch_common, batch_search_signal, batch_rec_signal)
            recon = F.mse_loss(out["search_recon"], batch_search_target) + F.mse_loss(
                out["rec_recon"], batch_rec_target
            )
            orth = F.cosine_similarity(out["q_shared"], out["q_search"], dim=1).abs().mean()
            orth = orth + F.cosine_similarity(out["q_shared"], out["q_rec"], dim=1).abs().mean()
            anchor = F.mse_loss(out["z_search0"], out["z_shared"].detach())
            anchor = anchor + F.mse_loss(out["z_rec0"], out["z_shared"].detach())
            norm = out["r_search"].pow(2).mean() + out["r_rec"].pow(2).mean()
            collision = prefix_bucket_repulsion(out["prefix_idx"], out["r_search"], args.collision_margin)
            collision = collision + prefix_bucket_repulsion(out["prefix_idx"], out["r_rec"], args.collision_margin)
            inbucket = prefix_inbucket_contrast(out["prefix_idx"], out["q_search"], args.inbucket_contrast_temperature)
            inbucket = inbucket + prefix_inbucket_contrast(
                out["prefix_idx"], out["q_rec"], args.inbucket_contrast_temperature
            )
            search_contrast = task_signal_contrast(
                out["search_recon"],
                batch_search_signal,
                args.contrast_temperature,
            )
            rec_contrast = task_signal_contrast(
                out["rec_recon"],
                batch_rec_signal,
                args.contrast_temperature,
            )
            loss = (
                recon
                + args.commit_weight * out["commit"]
                + args.balance_weight * out["balance"]
                + args.orth_weight * orth
                + args.anchor_weight * anchor
                + args.residual_norm_weight * norm
                + args.collision_weight * collision
                + args.inbucket_contrast_weight * inbucket
                + args.search_contrast_weight * search_contrast
                + args.rec_contrast_weight * rec_contrast
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["recon"] += float(recon.detach().cpu())
            totals["commit"] += float(out["commit"].detach().cpu())
            totals["balance"] += float(out["balance"].detach().cpu())
            totals["orth"] += float(orth.detach().cpu())
            totals["anchor"] += float(anchor.detach().cpu())
            totals["residual_norm"] += float(norm.detach().cpu())
            totals["collision"] += float(collision.detach().cpu())
            totals["inbucket"] += float(inbucket.detach().cpu())
            totals["search_contrast"] += float(search_contrast.detach().cpu())
            totals["rec_contrast"] += float(rec_contrast.detach().cpu())
            steps += 1
        row = {"epoch": epoch, **{k: v / max(steps, 1) for k, v in totals.items()}}
        logs.append(row)
        if epoch == 1 or epoch == args.task_epochs or epoch % args.log_every == 0:
            print(json.dumps({"stage": "task", **row}, ensure_ascii=False))
    return model, logs


def prefix_bucket_repulsion(prefix_idx, values, margin):
    if margin <= 0:
        return values.new_tensor(0.0)
    prefix = torch.stack(prefix_idx, dim=1)
    same_prefix = (prefix[:, None, :] == prefix[None, :, :]).all(dim=-1)
    same_prefix.fill_diagonal_(False)
    if not bool(same_prefix.any()):
        return values.new_tensor(0.0)
    normalized = F.normalize(values, p=2, dim=1)
    dist = 2.0 - 2.0 * (normalized @ normalized.t()).clamp(min=-1.0, max=1.0)
    return F.relu(margin - dist)[same_prefix].pow(2).mean()


def prefix_inbucket_contrast(prefix_idx, values, temperature):
    if temperature <= 0:
        return values.new_tensor(0.0)
    prefix = torch.stack(prefix_idx, dim=1)
    same_prefix = (prefix[:, None, :] == prefix[None, :, :]).all(dim=-1)
    same_prefix.fill_diagonal_(False)
    if not bool(same_prefix.any()):
        return values.new_tensor(0.0)
    normalized = F.normalize(values, p=2, dim=1)
    sim = (normalized @ normalized.t()) / temperature
    return F.softplus(sim[same_prefix]).mean()


def task_signal_contrast(pred, target, temperature):
    if temperature <= 0:
        return pred.new_tensor(0.0)
    mask = (pred.norm(dim=1) > 1e-8) & (target.norm(dim=1) > 1e-8)
    if int(mask.sum().detach().cpu()) < 2:
        return pred.new_tensor(0.0)
    pred_norm = F.normalize(pred[mask], p=2, dim=1)
    target_norm = F.normalize(target[mask], p=2, dim=1)
    logits = (pred_norm @ target_norm.t()) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


@torch.no_grad()
def encode_codes(model, x_common, x_search_signal, x_rec_signal, device, batch_size):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_common), torch.from_numpy(x_search_signal), torch.from_numpy(x_rec_signal)),
        batch_size=batch_size,
        shuffle=False,
    )
    prefix_codes = []
    search_codes = []
    rec_codes = []
    for batch_common, batch_search, batch_rec in loader:
        out = model(batch_common.to(device), batch_search.to(device), batch_rec.to(device))
        prefix_codes.extend(torch.stack(out["prefix_idx"], dim=1).cpu().numpy().tolist())
        search_codes.extend(torch.stack(out["search_idx"], dim=1).cpu().numpy().tolist())
        rec_codes.extend(torch.stack(out["rec_idx"], dim=1).cpu().numpy().tolist())
    return prefix_codes, search_codes, rec_codes


def token_list(codes, tag):
    return [f"<{tag}{idx + 1}_{int(code)}>" for idx, code in enumerate(codes)]


def reconstruct_from_rq_codes(codes, codebooks):
    codes_np = np.asarray(codes, dtype=np.int64)
    vectors = None
    for level, codebook in enumerate(codebooks):
        centers = codebook.detach().cpu().numpy().astype(np.float32)
        picked = centers[codes_np[:, level]]
        vectors = picked if vectors is None else vectors + picked
    return vectors.astype(np.float32)


def squared_distances(x, centers):
    return ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)


def rq_top_paths(value, codebooks, beam_size):
    beams = [((), 0.0, value.astype(np.float32))]
    for codebook in codebooks:
        centers = codebook.detach().cpu().numpy().astype(np.float32)
        next_beams = []
        for path, score, residual in beams:
            dist = squared_distances(residual[None, :], centers)[0]
            for idx in np.argsort(dist)[: min(beam_size, len(dist))]:
                idx = int(idx)
                next_beams.append((path + (idx,), score + float(dist[idx]), residual - centers[idx]))
        next_beams.sort(key=lambda row: row[1])
        beams = next_beams[:beam_size]
    return [(path, score) for path, score, _ in beams]


def full_sid_key(prefix_codes, suffix_codes):
    return tuple(int(x) for x in prefix_codes) + tuple(int(x) for x in suffix_codes)


def unique_suffix_assign(prefix_codes, original_suffix_codes, residuals, codebooks, beam_size):
    rows = []
    for item_idx, (prefix_code, original_suffix_code) in enumerate(zip(prefix_codes, original_suffix_codes)):
        paths = rq_top_paths(residuals[item_idx], codebooks, beam_size)
        original = tuple(int(x) for x in original_suffix_code)
        path_to_score = {path: score for path, score in paths}
        if original not in path_to_score:
            paths.append((original, float("inf")))
            path_to_score[original] = float("inf")
        rows.append(
            {
                "item_idx": item_idx,
                "prefix": tuple(int(x) for x in prefix_code),
                "original": original,
                "original_score": path_to_score[original],
                "paths": paths,
            }
        )

    original_buckets = defaultdict(list)
    for row in rows:
        original_buckets[full_sid_key(row["prefix"], row["original"])].append(row)

    ordered = []
    for bucket_rows in original_buckets.values():
        bucket_rows.sort(key=lambda row: (row["original_score"], row["item_idx"]))
        ordered.append(bucket_rows[0])
    for bucket_rows in original_buckets.values():
        ordered.extend(bucket_rows[1:])

    used = set()
    assigned = {}
    changed = 0
    failed = 0
    selected_rank_sum = 0
    for row in ordered:
        selected = None
        selected_rank = None
        for rank, (path, _) in enumerate(row["paths"]):
            key = full_sid_key(row["prefix"], path)
            if key not in used:
                selected = path
                selected_rank = rank
                break
        if selected is None:
            selected = row["original"]
            selected_rank = len(row["paths"])
            failed += 1
        used.add(full_sid_key(row["prefix"], selected))
        changed += int(selected != row["original"])
        selected_rank_sum += selected_rank
        assigned[row["item_idx"]] = list(selected)

    return [assigned[idx] for idx in range(len(prefix_codes))], {
        "changed": changed,
        "unchanged": len(prefix_codes) - changed,
        "failed": failed,
        "avg_selected_rank": selected_rank_sum / max(len(prefix_codes), 1),
        "beam_size": beam_size,
    }


@torch.no_grad()
def encode_codes_with_assignment(
    model,
    x_common,
    x_search_signal,
    x_rec_signal,
    device,
    batch_size,
    assignment,
    unique_beam_size,
):
    if assignment == "argmin":
        prefix_codes, search_codes, rec_codes = encode_codes(model, x_common, x_search_signal, x_rec_signal, device, batch_size)
        return prefix_codes, search_codes, rec_codes, {"mode": "argmin"}

    model.eval()
    z_shared_all = []
    z_search0_all = []
    z_rec0_all = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_common), torch.from_numpy(x_search_signal), torch.from_numpy(x_rec_signal)),
        batch_size=batch_size,
        shuffle=False,
    )
    for batch_common, batch_search, batch_rec in loader:
        batch_common = batch_common.to(device)
        batch_search = batch_search.to(device)
        batch_rec = batch_rec.to(device)
        z_shared = model.shared.encoder(batch_common)
        z_search0, z_rec0 = model.task_latents(z_shared, batch_common, batch_search, batch_rec)
        z_shared_all.append(z_shared.cpu().numpy())
        z_search0_all.append(z_search0.cpu().numpy())
        z_rec0_all.append(z_rec0.cpu().numpy())

    z_shared_np = np.concatenate(z_shared_all, axis=0)
    z_search0_np = np.concatenate(z_search0_all, axis=0)
    z_rec0_np = np.concatenate(z_rec0_all, axis=0)
    prefix_assignment = "balanced" if assignment in ("balanced", "balanced_unique") else "argmin"
    suffix_assignment = "balanced" if assignment == "balanced_unique" else "argmin"
    prefix_codes = rq_assign_codes_np(z_shared_np, model.shared.quantizer.codebooks, prefix_assignment)
    q_shared_np = reconstruct_from_rq_codes(prefix_codes, model.shared.quantizer.codebooks)
    r_search = z_search0_np - q_shared_np
    r_rec = z_rec0_np - q_shared_np
    search_codes = rq_assign_codes_np(r_search, model.search_quantizer.codebooks, suffix_assignment)
    rec_codes = rq_assign_codes_np(r_rec, model.rec_quantizer.codebooks, suffix_assignment)

    assignment_log = {
        "mode": assignment,
        "prefix_assignment": prefix_assignment,
        "suffix_initial_assignment": suffix_assignment,
    }
    if assignment in ("unique", "balanced_unique"):
        search_codes, search_log = unique_suffix_assign(
            prefix_codes,
            search_codes,
            r_search,
            model.search_quantizer.codebooks,
            unique_beam_size,
        )
        rec_codes, rec_log = unique_suffix_assign(
            prefix_codes,
            rec_codes,
            r_rec,
            model.rec_quantizer.codebooks,
            unique_beam_size,
        )
        assignment_log["unique"] = {"search": search_log, "rec": rec_log}
    return prefix_codes, search_codes, rec_codes, assignment_log


def unique_capacity(prefix_codes, suffix_sizes):
    capacity_per_prefix = int(np.prod(suffix_sizes))
    prefix_counts = Counter(tuple(int(x) for x in codes) for codes in prefix_codes)
    overloaded = {prefix: count for prefix, count in prefix_counts.items() if count > capacity_per_prefix}
    return {
        "capacity_per_prefix": capacity_per_prefix,
        "max_prefix_bucket": max(prefix_counts.values()) if prefix_counts else 0,
        "overloaded_prefix_count": len(overloaded),
        "overloaded_item_count": int(sum(count - capacity_per_prefix for count in overloaded.values())),
    }


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
    if not item_ids:
        raise RuntimeError("no items loaded; check --data-dir/--item-ids")

    query_docs, query_counter = aggregate_search_queries(data_dir, item_ids, args.max_queries_per_item)
    rec_docs_by_id, neighbor_counter = aggregate_rec_neighbors(
        data_dir,
        item_ids,
        args.neighbor_window,
        args.max_neighbors_per_item,
    )
    text_docs, _, _ = make_docs(item_ids, item_rows, query_docs, rec_docs_by_id)

    cache_dir = Path(args.embedding_cache_dir) if args.embedding_cache_dir else data_dir / "text_embedding_cache"
    common_x = encode_text_docs(args, text_docs, "common", device, cache_dir)
    args.text_embedding_dim = int(common_x.shape[1])
    x_search_signal, search_signal_stats = build_search_signal_embeddings(args, item_ids, query_counter, device, cache_dir)
    x_rec_signal, rec_signal_stats = build_rec_signal_embeddings(
        item_ids,
        common_x,
        neighbor_counter,
        args.max_neighbors_per_item,
    )
    if args.rec_target_signal == "neighbor":
        x_rec_target_signal = x_rec_signal
        rec_target_stats = {"mode": "neighbor", **rec_signal_stats}
    elif args.rec_target_signal == "transition":
        x_rec_target_signal, transition_stats = build_rec_transition_embeddings(
            data_dir,
            item_ids,
            common_x,
            args.max_neighbors_per_item,
        )
        rec_target_stats = {"mode": "transition", **transition_stats}
    else:
        raise ValueError(f"unknown rec target signal: {args.rec_target_signal}")
    x_search_target = make_task_targets(common_x, x_search_signal, args.task_target_scale, args.suffix_target_mode)
    x_rec_target = make_task_targets(common_x, x_rec_target_signal, args.task_target_scale, args.suffix_target_mode)

    shared_model, shared_log = train_shared(args, common_x.astype(np.float32), prefix_sizes, device)
    model, task_log = train_task(
        args,
        shared_model,
        common_x.astype(np.float32),
        x_search_signal,
        x_rec_signal,
        x_search_target,
        x_rec_target,
        search_sizes,
        rec_sizes,
        device,
    )
    prefix_codes, search_codes, rec_codes, assignment_log = encode_codes_with_assignment(
        model,
        common_x.astype(np.float32),
        x_search_signal,
        x_rec_signal,
        device,
        args.batch_size,
        args.assignment,
        args.unique_beam_size,
    )

    sid_mapping = {}
    search_sid_by_item = {}
    rec_sid_by_item = {}
    prefix_counter = Counter()
    for item_id, prefix_code, search_code, rec_code in zip(item_ids, prefix_codes, search_codes, rec_codes):
        prefix = token_list(prefix_code, "P")
        search_suffix = token_list(search_code, "S")
        rec_suffix = token_list(rec_code, "R")
        search_sid = prefix + search_suffix
        rec_sid = prefix + rec_suffix
        sid_mapping[item_id] = {
            "prefix": prefix,
            "search_sid": search_sid,
            "rec_sid": rec_sid,
            "search_suffix": search_suffix,
            "rec_suffix": rec_suffix,
            "prefix_codes": [int(x) for x in prefix_code],
            "search_suffix_codes": [int(x) for x in search_code],
            "rec_suffix_codes": [int(x) for x in rec_code],
        }
        search_sid_by_item[item_id] = search_sid
        rec_sid_by_item[item_id] = rec_sid
        prefix_counter[tuple(prefix)] += 1

    mapping_v2_path = output_dir / "sid_mapping.v2_text_encoder_fusion_plc.json"
    write_json(mapping_v2_path, sid_mapping)
    write_json(output_dir / "sid_mapping.v2.json", sid_mapping)
    write_json(output_dir / "sid_mapping.v1.json", sid_mapping)
    write_json(output_dir / "shared_train_log.json", shared_log)
    write_json(output_dir / "task_train_log.json", task_log)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "params": vars(args),
            "prefix_sizes": prefix_sizes,
            "search_sizes": search_sizes,
            "rec_sizes": rec_sizes,
        },
        output_dir / "text_encoder_fusion_plc_sid_model.pt",
    )

    prefix_labels = ["_".join(map(str, codes)) for codes in prefix_codes]
    search_suffix_labels = ["_".join(map(str, codes)) for codes in search_codes]
    rec_suffix_labels = ["_".join(map(str, codes)) for codes in rec_codes]
    query_counts = [sum(query_counter[item_id].values()) for item_id in item_ids]
    neighbor_counts = [sum(neighbor_counter[item_id].values()) for item_id in item_ids]
    stats = {
        "num_items": len(item_ids),
        "has_metadata": has_metadata,
        "mapping_path": str(mapping_v2_path),
        "field_coverage": field_coverage(item_rows),
        "query_interaction_count": count_summary(query_counts),
        "rec_neighbor_count": count_summary(neighbor_counts),
        "codebook_sizes": {
            "prefix": prefix_sizes,
            "search_suffix": search_sizes,
            "rec_suffix": rec_sizes,
        },
        "sinkhorn": {
            "prefix_sk_epsilons": parse_float_sizes(args.prefix_sk_epsilons),
            "suffix_sk_epsilons": parse_float_sizes(args.suffix_sk_epsilons),
            "sk_iters": args.sk_iters,
            "note": "Sinkhorn is used inside RQ training only for layers with epsilon > 0; export assignment is still controlled separately by --assignment.",
        },
        "assignment": {
            **assignment_log,
            "search_unique_capacity": unique_capacity(prefix_codes, search_sizes),
            "rec_unique_capacity": unique_capacity(prefix_codes, rec_sizes),
        },
        "embedding": {
            "text_encoder_model": args.text_encoder_model,
            "dim": int(common_x.shape[1]),
            "max_length": args.text_encoder_max_length,
            "query_encoder_prefix": args.query_encoder_prefix,
            "cache_dir": str(cache_dir),
        },
        "task_signal": {
            "search": search_signal_stats,
            "rec": rec_signal_stats,
            "rec_target": rec_target_stats,
            "task_target_scale": args.task_target_scale,
            "suffix_target_mode": args.suffix_target_mode,
            "suffix_decode_input": args.suffix_decode_input,
            "search_contrast_weight": args.search_contrast_weight,
            "rec_contrast_weight": args.rec_contrast_weight,
            "contrast_temperature": args.contrast_temperature,
            "search_nonzero_ratio": float((np.linalg.norm(x_search_signal, axis=1) > 0).mean()),
            "rec_nonzero_ratio": float((np.linalg.norm(x_rec_signal, axis=1) > 0).mean()),
            "rec_target_nonzero_ratio": float((np.linalg.norm(x_rec_target_signal, axis=1) > 0).mean()),
        },
        "prefix": {
            "unique_count": len(prefix_counter),
            "bucket_size": count_summary(list(prefix_counter.values())),
            "usage": code_usage(prefix_codes, prefix_sizes),
        },
        "search_suffix_usage": code_usage(search_codes, search_sizes),
        "rec_suffix_usage": code_usage(rec_codes, rec_sizes),
        "search_sid_collision": collision_rate(search_sid_by_item),
        "rec_sid_collision": collision_rate(rec_sid_by_item),
        "redundancy_proxy_nmi": {
            "prefix_search_suffix": normalized_mutual_info_score(prefix_labels, search_suffix_labels),
            "prefix_rec_suffix": normalized_mutual_info_score(prefix_labels, rec_suffix_labels),
        },
        "final_shared_train_log": shared_log[-1] if shared_log else {},
        "final_task_train_log": task_log[-1] if task_log else {},
        "params": vars(args),
        "note": f"M7b-v2 TextEnc-Fusion-PLC-Behavior: frozen text encoder maps common text, queries, and neighbor items; task signals are pooled then fused with common vectors before Adapter/LoRA; task_encoder={args.task_encoder}; suffix codebooks continue RQ on z_task - q_prefix; assignment={args.assignment}; suffix_target_mode={args.suffix_target_mode}; suffix_decode_input={args.suffix_decode_input}; rec_target_signal={args.rec_target_signal}; optional task-signal InfoNCE is controlled by search_contrast_weight/rec_contrast_weight.",
    }
    write_json(output_dir / "sid_stats.v2_text_encoder_fusion_plc.json", stats)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics")
    parser.add_argument(
        "--output-dir",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/sid_v2_text_encoder_fusion_plc",
    )
    parser.add_argument("--item-ids", default="")
    parser.add_argument("--max-items", type=int, default=10000)
    parser.add_argument("--prefix-codebook-sizes", default="8,4")
    parser.add_argument("--search-codebook-sizes", default="32,8")
    parser.add_argument("--rec-codebook-sizes", default="32,8")
    parser.add_argument("--max-queries-per-item", type=int, default=20)
    parser.add_argument("--neighbor-window", type=int, default=2)
    parser.add_argument("--max-neighbors-per-item", type=int, default=200)
    parser.add_argument("--text-max-features", type=int, default=50000)
    parser.add_argument("--hash-features", type=int, default=32768)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--text-encoder-model", default="intfloat/e5-small-v2")
    parser.add_argument("--text-encoder-prefix", default="passage: ")
    parser.add_argument("--query-encoder-prefix", default="query: ")
    parser.add_argument("--text-encoder-max-length", type=int, default=256)
    parser.add_argument("--text-encoder-batch-size", type=int, default=64)
    parser.add_argument("--embedding-cache-dir", default="")
    parser.add_argument("--task-encoder", choices=["adapter", "lora"], default="adapter")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--shared-epochs", type=int, default=50)
    parser.add_argument("--task-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--vq-beta", type=float, default=0.25)
    parser.add_argument("--commit-weight", type=float, default=0.25)
    parser.add_argument("--balance-weight", type=float, default=0.05)
    parser.add_argument("--balance-temperature", type=float, default=0.05)
    parser.add_argument("--prefix-sk-epsilons", default="")
    parser.add_argument("--suffix-sk-epsilons", default="")
    parser.add_argument("--sk-iters", type=int, default=100)
    parser.add_argument("--orth-weight", type=float, default=0.02)
    parser.add_argument("--anchor-weight", type=float, default=0.01)
    parser.add_argument("--residual-norm-weight", type=float, default=0.001)
    parser.add_argument("--collision-weight", type=float, default=0.0)
    parser.add_argument("--collision-margin", type=float, default=0.5)
    parser.add_argument("--inbucket-contrast-weight", type=float, default=0.0)
    parser.add_argument("--inbucket-contrast-temperature", type=float, default=0.2)
    parser.add_argument("--search-contrast-weight", type=float, default=0.0)
    parser.add_argument("--rec-contrast-weight", type=float, default=0.0)
    parser.add_argument("--contrast-temperature", type=float, default=0.07)
    parser.add_argument("--adapter-scale", type=float, default=0.5)
    parser.add_argument("--task-target-scale", type=float, default=0.5)
    parser.add_argument("--suffix-target-mode", choices=["common_task", "task_signal", "delta_from_common"], default="common_task")
    parser.add_argument("--suffix-decode-input", choices=["full", "suffix"], default="full")
    parser.add_argument("--rec-target-signal", choices=["neighbor", "transition"], default="neighbor")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-codebooks", choices=["kmeans", "random"], default="kmeans")
    parser.add_argument("--assignment", choices=["argmin", "balanced", "unique", "balanced_unique"], default="argmin")
    parser.add_argument("--unique-beam-size", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
