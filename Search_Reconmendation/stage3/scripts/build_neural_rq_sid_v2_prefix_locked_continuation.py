#!/usr/bin/env python3
import argparse
import json
import math
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
    collision_rate,
    count_summary,
    field_coverage,
    init_rq_codebooks,
    load_item_rows,
    make_docs,
    parse_sizes,
    rq_assign_codes_np,
    vectorize_reduce_hash,
    vectorize_reduce_text,
    write_json,
)


class SharedRQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, prefix_sizes, beta, dropout, balance_temperature):
        super().__init__()
        self.encoder = MLP(input_dim, hidden_dim, latent_dim, dropout)
        self.quantizer = ResidualVectorQuantizer(prefix_sizes, latent_dim, beta, balance_temperature)
        self.decoder = MLP(latent_dim, hidden_dim, input_dim, dropout)

    def forward(self, x):
        z = self.encoder(x)
        q, indices, commit, balance = self.quantizer(z)
        recon = self.decoder(q)
        return {"z": z, "q": q, "indices": indices, "commit": commit, "balance": balance, "recon": recon}


class PrefixLockedContinuationTokenizer(nn.Module):
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
    ):
        super().__init__()
        self.shared = shared_model
        self.adapter_scale = adapter_scale
        self.search_adapter = MLP(task_input_dim, hidden_dim, latent_dim, dropout)
        self.rec_adapter = MLP(task_input_dim, hidden_dim, latent_dim, dropout)
        self.search_quantizer = ResidualVectorQuantizer(search_sizes, latent_dim, beta, balance_temperature)
        self.rec_quantizer = ResidualVectorQuantizer(rec_sizes, latent_dim, beta, balance_temperature)
        self.search_decoder = MLP(latent_dim, hidden_dim, task_input_dim, dropout)
        self.rec_decoder = MLP(latent_dim, hidden_dim, task_input_dim, dropout)

    def shared_codes(self, x_common):
        z_shared = self.shared.encoder(x_common)
        q_shared, prefix_idx, _, _ = self.shared.quantizer(z_shared)
        return z_shared, q_shared, prefix_idx

    def forward(self, x_common, x_search_task, x_rec_task):
        with torch.no_grad():
            z_shared, q_shared, prefix_idx = self.shared_codes(x_common)
        z_search0 = z_shared + self.adapter_scale * self.search_adapter(x_search_task)
        z_rec0 = z_shared + self.adapter_scale * self.rec_adapter(x_rec_task)
        r_search = z_search0 - q_shared
        r_rec = z_rec0 - q_shared

        q_search, search_idx, search_commit, search_balance = self.search_quantizer(r_search)
        q_rec, rec_idx, rec_commit, rec_balance = self.rec_quantizer(r_rec)
        search_recon = self.search_decoder(q_shared + q_search)
        rec_recon = self.rec_decoder(q_shared + q_rec)
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
def init_suffix_codebooks(model, x_common, x_search_task, x_rec_task, device, seed):
    model.eval()
    x_common_t = torch.from_numpy(x_common).to(device)
    x_search_t = torch.from_numpy(x_search_task).to(device)
    x_rec_t = torch.from_numpy(x_rec_task).to(device)
    z_shared, q_shared, _ = model.shared_codes(x_common_t)
    r_search = z_shared + model.adapter_scale * model.search_adapter(x_search_t) - q_shared
    r_rec = z_shared + model.adapter_scale * model.rec_adapter(x_rec_t) - q_shared
    init_rq_codebooks(model.search_quantizer, r_search.cpu().numpy(), seed + 1000)
    init_rq_codebooks(model.rec_quantizer, r_rec.cpu().numpy(), seed + 2000)


def train_task(args, shared_model, x_common, x_search_task, x_rec_task, search_sizes, rec_sizes, device):
    for param in shared_model.parameters():
        param.requires_grad = False
    shared_model.eval()

    dataset = TensorDataset(torch.from_numpy(x_common), torch.from_numpy(x_search_task), torch.from_numpy(x_rec_task))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 7),
    )
    model = PrefixLockedContinuationTokenizer(
        shared_model=shared_model,
        task_input_dim=x_search_task.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        search_sizes=search_sizes,
        rec_sizes=rec_sizes,
        beta=args.vq_beta,
        dropout=args.dropout,
        balance_temperature=args.balance_temperature,
        adapter_scale=args.adapter_scale,
    ).to(device)
    if args.init_codebooks == "kmeans":
        init_suffix_codebooks(model, x_common, x_search_task, x_rec_task, device, args.seed)

    trainable = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    logs = []
    for epoch in range(1, args.task_epochs + 1):
        model.train()
        model.shared.eval()
        totals = Counter()
        steps = 0
        for batch_common, batch_search, batch_rec in loader:
            batch_common = batch_common.to(device)
            batch_search = batch_search.to(device)
            batch_rec = batch_rec.to(device)
            out = model(batch_common, batch_search, batch_rec)
            recon = F.mse_loss(out["search_recon"], batch_search) + F.mse_loss(out["rec_recon"], batch_rec)
            orth = F.cosine_similarity(out["q_shared"], out["q_search"], dim=1).abs().mean()
            orth = orth + F.cosine_similarity(out["q_shared"], out["q_rec"], dim=1).abs().mean()
            anchor = F.mse_loss(out["z_search0"], out["z_shared"].detach())
            anchor = anchor + F.mse_loss(out["z_rec0"], out["z_shared"].detach())
            norm = out["r_search"].pow(2).mean() + out["r_rec"].pow(2).mean()
            loss = (
                recon
                + args.commit_weight * out["commit"]
                + args.balance_weight * out["balance"]
                + args.orth_weight * orth
                + args.anchor_weight * anchor
                + args.residual_norm_weight * norm
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
            steps += 1
        row = {"epoch": epoch, **{k: v / max(steps, 1) for k, v in totals.items()}}
        logs.append(row)
        if epoch == 1 or epoch == args.task_epochs or epoch % args.log_every == 0:
            print(json.dumps({"stage": "task", **row}, ensure_ascii=False))
    return model, logs


@torch.no_grad()
def encode_codes(model, x_common, x_search_task, x_rec_task, device, batch_size):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_common), torch.from_numpy(x_search_task), torch.from_numpy(x_rec_task)),
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


@torch.no_grad()
def encode_codes_with_assignment(model, x_common, x_search_task, x_rec_task, device, batch_size, assignment):
    if assignment == "argmin":
        return encode_codes(model, x_common, x_search_task, x_rec_task, device, batch_size)

    model.eval()
    z_shared_all = []
    z_search0_all = []
    z_rec0_all = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_common), torch.from_numpy(x_search_task), torch.from_numpy(x_rec_task)),
        batch_size=batch_size,
        shuffle=False,
    )
    for batch_common, batch_search, batch_rec in loader:
        batch_common = batch_common.to(device)
        batch_search = batch_search.to(device)
        batch_rec = batch_rec.to(device)
        z_shared = model.shared.encoder(batch_common)
        z_search0 = z_shared + model.adapter_scale * model.search_adapter(batch_search)
        z_rec0 = z_shared + model.adapter_scale * model.rec_adapter(batch_rec)
        z_shared_all.append(z_shared.cpu().numpy())
        z_search0_all.append(z_search0.cpu().numpy())
        z_rec0_all.append(z_rec0.cpu().numpy())

    z_shared_np = np.concatenate(z_shared_all, axis=0)
    z_search0_np = np.concatenate(z_search0_all, axis=0)
    z_rec0_np = np.concatenate(z_rec0_all, axis=0)
    prefix_codes = rq_assign_codes_np(z_shared_np, model.shared.quantizer.codebooks, assignment)
    q_shared_np = reconstruct_from_rq_codes(prefix_codes, model.shared.quantizer.codebooks)
    search_codes = rq_assign_codes_np(z_search0_np - q_shared_np, model.search_quantizer.codebooks, assignment)
    rec_codes = rq_assign_codes_np(z_rec0_np - q_shared_np, model.rec_quantizer.codebooks, assignment)
    return prefix_codes, search_codes, rec_codes


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
    text_docs, search_docs, rec_docs = make_docs(item_ids, item_rows, query_docs, rec_docs_by_id)

    common_x = vectorize_reduce_text(text_docs, args.text_max_features, args.feature_dim, args.seed)
    search_x = vectorize_reduce_text(search_docs, args.text_max_features, args.feature_dim, args.seed + 100)
    rec_x = vectorize_reduce_hash(rec_docs, args.hash_features, args.feature_dim, args.seed + 200)
    x_search_task = np.concatenate([common_x, search_x], axis=1).astype(np.float32)
    x_rec_task = np.concatenate([common_x, rec_x], axis=1).astype(np.float32)

    shared_model, shared_log = train_shared(args, common_x.astype(np.float32), prefix_sizes, device)
    model, task_log = train_task(
        args,
        shared_model,
        common_x.astype(np.float32),
        x_search_task,
        x_rec_task,
        search_sizes,
        rec_sizes,
        device,
    )
    prefix_codes, search_codes, rec_codes = encode_codes_with_assignment(
        model,
        common_x.astype(np.float32),
        x_search_task,
        x_rec_task,
        device,
        args.batch_size,
        args.assignment,
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

    mapping_v2_path = output_dir / "sid_mapping.v2_prefix_locked_continuation.json"
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
        output_dir / "prefix_locked_continuation_sid_model.pt",
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
        "note": "M5 PrefixLocked-Continuation: shared prefix codes/codebooks are frozen; task adapters make a small z_task shift and task suffix codebooks continue RQ on z_task - q_prefix.",
    }
    write_json(output_dir / "sid_stats.v2_prefix_locked_continuation.json", stats)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics")
    parser.add_argument(
        "--output-dir",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/sid_v2_prefix_locked_continuation",
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
    parser.add_argument("--orth-weight", type=float, default=0.02)
    parser.add_argument("--anchor-weight", type=float, default=0.01)
    parser.add_argument("--residual-norm-weight", type=float, default=0.001)
    parser.add_argument("--adapter-scale", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-codebooks", choices=["kmeans", "random"], default="kmeans")
    parser.add_argument("--assignment", choices=["argmin", "balanced"], default="argmin")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
