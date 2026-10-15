#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.data import DataLoader, TensorDataset

from build_neural_rq_sid_v2 import (
    aggregate_rec_neighbors,
    aggregate_search_queries,
    code_usage,
    collision_rate,
    count_summary,
    field_coverage,
    load_item_rows,
    make_docs,
    write_json,
)
from build_neural_rq_sid_v2_text_encoder_fusion_plc import (
    SharedRQVAE,
    TextEncoderFusionPLCTokenizer,
    build_rec_signal_embeddings,
    build_search_signal_embeddings,
    encode_text_docs,
    reconstruct_from_rq_codes,
    token_list,
)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def as_namespace(values):
    return argparse.Namespace(**values)


def center_distance_for_constraint(distances):
    max_distance = distances.max()
    min_distance = distances.min()
    middle = (max_distance + min_distance) / 2
    amplitude = max_distance - middle + 1e-5
    return (distances - middle) / amplitude


@torch.no_grad()
def sinkhorn_assign_np(distances_np, epsilon, iters):
    distances = torch.from_numpy(distances_np.astype(np.float64))
    distances = center_distance_for_constraint(distances)
    q = torch.exp(-distances / epsilon)
    if torch.isnan(q).any() or torch.isinf(q).any() or float(q.sum()) <= 0:
        return distances_np.argmin(axis=1).astype(np.int64), {"fallback": "initial_exp"}
    q = q / q.sum().clamp(min=1e-300)
    bsz = q.shape[0]
    n_codes = q.shape[1]
    fallback = ""
    for _ in range(iters):
        row_sum = q.sum(dim=1, keepdim=True)
        if bool((row_sum <= 0).any()):
            fallback = "row_sum"
            break
        q = q / row_sum / bsz
        col_sum = q.sum(dim=0, keepdim=True)
        if bool((col_sum <= 0).any()):
            fallback = "col_sum"
            break
        q = q / col_sum / n_codes
    if fallback or torch.isnan(q).any() or torch.isinf(q).any():
        return distances_np.argmin(axis=1).astype(np.int64), {"fallback": fallback or "nan_inf"}
    q = q * bsz
    return torch.argmax(q, dim=-1).cpu().numpy().astype(np.int64), {"fallback": ""}


def full_key(prefix, suffix):
    return tuple(int(x) for x in prefix) + tuple(int(x) for x in suffix)


def collision_groups(prefix_codes, suffix_codes):
    buckets = defaultdict(list)
    for idx, (prefix, suffix) in enumerate(zip(prefix_codes, suffix_codes)):
        buckets[full_key(prefix, suffix)].append(idx)
    return [items for items in buckets.values() if len(items) > 1], buckets


def squared_distances_np(x, centers):
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    c_norm = np.sum(centers * centers, axis=1, keepdims=True).T
    return x_norm + c_norm - 2 * x @ centers.T


def refine_last_suffix_layer(prefix_codes, suffix_codes, residuals, codebooks, epsilon, sk_iters, max_rounds):
    if len(codebooks) < 1:
        raise ValueError("suffix codebooks must not be empty")
    suffix_codes = np.asarray(suffix_codes, dtype=np.int64).copy()
    prefix_codes = np.asarray(prefix_codes, dtype=np.int64)
    residuals = residuals.astype(np.float32)
    centers = [codebook.detach().cpu().numpy().astype(np.float32) for codebook in codebooks]
    logs = []
    total_changed = 0
    total_fallback_groups = 0

    for round_idx in range(1, max_rounds + 1):
        groups, _ = collision_groups(prefix_codes.tolist(), suffix_codes.tolist())
        if not groups:
            break
        changed = 0
        fallback_groups = 0
        group_sizes = []
        for group in groups:
            group_sizes.append(len(group))
            first_part = suffix_codes[group, :-1]
            residual_after_fixed = residuals[group].copy()
            for level, codebook in enumerate(centers[:-1]):
                residual_after_fixed -= codebook[first_part[:, level]]
            distances = squared_distances_np(residual_after_fixed, centers[-1])
            assigned, sk_log = sinkhorn_assign_np(distances, epsilon, sk_iters)
            if sk_log.get("fallback"):
                fallback_groups += 1
            before = suffix_codes[group, -1].copy()
            suffix_codes[group, -1] = assigned
            changed += int((before != assigned).sum())
        total_changed += changed
        total_fallback_groups += fallback_groups
        next_groups, buckets = collision_groups(prefix_codes.tolist(), suffix_codes.tolist())
        logs.append(
            {
                "round": round_idx,
                "input_collision_groups": len(groups),
                "input_collision_group_size": count_summary(group_sizes),
                "changed_last_codes": changed,
                "fallback_groups": fallback_groups,
                "remaining_collision_groups": len(next_groups),
                "unique_sid_count": len(buckets),
                "gensar_collision_rate": (len(prefix_codes) - len(buckets)) / max(len(prefix_codes), 1),
            }
        )
        if changed == 0:
            break

    return suffix_codes.tolist(), {
        "epsilon": epsilon,
        "sk_iters": sk_iters,
        "max_rounds": max_rounds,
        "rounds": logs,
        "total_changed_last_codes": total_changed,
        "total_fallback_groups": total_fallback_groups,
    }


def restore_model(params, checkpoint, input_dim, device):
    shared = SharedRQVAE(
        input_dim=input_dim,
        hidden_dim=params.hidden_dim,
        latent_dim=params.latent_dim,
        prefix_sizes=checkpoint["prefix_sizes"],
        beta=params.vq_beta,
        dropout=params.dropout,
        balance_temperature=params.balance_temperature,
    )
    model = TextEncoderFusionPLCTokenizer(
        shared_model=shared,
        task_input_dim=input_dim,
        hidden_dim=params.hidden_dim,
        latent_dim=params.latent_dim,
        search_sizes=checkpoint["search_sizes"],
        rec_sizes=checkpoint["rec_sizes"],
        beta=params.vq_beta,
        dropout=params.dropout,
        balance_temperature=params.balance_temperature,
        adapter_scale=params.adapter_scale,
        task_encoder=params.task_encoder,
        lora_rank=params.lora_rank,
        lora_alpha=params.lora_alpha,
        suffix_decode_input=getattr(params, "suffix_decode_input", "full"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def compute_task_residuals(model, x_common, x_search_signal, x_rec_signal, prefix_codes, device, batch_size):
    search_latents = []
    rec_latents = []
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
        search_latents.append(z_search0.cpu().numpy())
        rec_latents.append(z_rec0.cpu().numpy())
    q_prefix = reconstruct_from_rq_codes(prefix_codes, model.shared.quantizer.codebooks)
    return (
        np.concatenate(search_latents, axis=0).astype(np.float32) - q_prefix,
        np.concatenate(rec_latents, axis=0).astype(np.float32) - q_prefix,
    )


def gensar_collision_stats(sid_by_item):
    unique_count = len({tuple(sid) for sid in sid_by_item.values()})
    n_items = len(sid_by_item)
    return {
        "unique_sid_count": unique_count,
        "collision_rate": (n_items - unique_count) / max(n_items, 1),
    }


def build_mapping(item_ids, prefix_codes, search_codes, rec_codes):
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
    return sid_mapping, search_sid_by_item, rec_sid_by_item, prefix_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/cjk/Search_Reconmendation/stage3/data/gensar_lite_5k")
    parser.add_argument("--sid-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--item-ids", default="")
    parser.add_argument("--max-items", type=int, default=10000)
    parser.add_argument("--epsilon", type=float, default=0.003)
    parser.add_argument("--sk-iters", type=int, default=100)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--tasks", choices=["search", "rec", "both"], default="both")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sid_dir = Path(args.sid_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint = torch.load(sid_dir / "text_encoder_fusion_plc_sid_model.pt", map_location="cpu")
    params = as_namespace(checkpoint["params"])
    params.data_dir = str(data_dir)
    params.item_ids = args.item_ids or getattr(params, "item_ids", "")
    params.max_items = args.max_items

    item_ids, item_rows, has_metadata = load_item_rows(data_dir, params.item_ids, params.max_items)
    if not item_ids:
        raise RuntimeError("no items loaded")
    query_docs, query_counter = aggregate_search_queries(data_dir, item_ids, params.max_queries_per_item)
    rec_docs_by_id, neighbor_counter = aggregate_rec_neighbors(
        data_dir,
        item_ids,
        params.neighbor_window,
        params.max_neighbors_per_item,
    )
    text_docs, _, _ = make_docs(item_ids, item_rows, query_docs, rec_docs_by_id)
    cache_dir = Path(params.embedding_cache_dir) if getattr(params, "embedding_cache_dir", "") else data_dir / "text_embedding_cache"
    common_x = encode_text_docs(params, text_docs, "common", device, cache_dir)
    params.text_embedding_dim = int(common_x.shape[1])
    x_search_signal, search_signal_stats = build_search_signal_embeddings(params, item_ids, query_counter, device, cache_dir)
    x_rec_signal, rec_signal_stats = build_rec_signal_embeddings(
        item_ids,
        common_x,
        neighbor_counter,
        params.max_neighbors_per_item,
    )

    original = load_json(sid_dir / "sid_mapping.v2.json")
    prefix_codes = [original[item_id]["prefix_codes"] for item_id in item_ids]
    search_codes = [original[item_id]["search_suffix_codes"] for item_id in item_ids]
    rec_codes = [original[item_id]["rec_suffix_codes"] for item_id in item_ids]

    model = restore_model(params, checkpoint, common_x.shape[1], device)
    r_search, r_rec = compute_task_residuals(
        model,
        common_x.astype(np.float32),
        x_search_signal.astype(np.float32),
        x_rec_signal.astype(np.float32),
        prefix_codes,
        device,
        params.batch_size,
    )

    refine_log = {}
    if args.tasks in ("search", "both"):
        search_codes, refine_log["search"] = refine_last_suffix_layer(
            prefix_codes,
            search_codes,
            r_search,
            model.search_quantizer.codebooks,
            args.epsilon,
            args.sk_iters,
            args.max_rounds,
        )
    if args.tasks in ("rec", "both"):
        rec_codes, refine_log["rec"] = refine_last_suffix_layer(
            prefix_codes,
            rec_codes,
            r_rec,
            model.rec_quantizer.codebooks,
            args.epsilon,
            args.sk_iters,
            args.max_rounds,
        )

    sid_mapping, search_sid_by_item, rec_sid_by_item, prefix_counter = build_mapping(
        item_ids,
        prefix_codes,
        search_codes,
        rec_codes,
    )
    write_json(output_dir / "sid_mapping.v2.json", sid_mapping)
    write_json(output_dir / "sid_mapping.v1.json", sid_mapping)
    write_json(output_dir / "sid_mapping.m7b_cg_sinkhorn.json", sid_mapping)

    prefix_labels = ["_".join(map(str, codes)) for codes in prefix_codes]
    search_suffix_labels = ["_".join(map(str, codes)) for codes in search_codes]
    rec_suffix_labels = ["_".join(map(str, codes)) for codes in rec_codes]
    query_counts = [sum(query_counter[item_id].values()) for item_id in item_ids]
    neighbor_counts = [sum(neighbor_counter[item_id].values()) for item_id in item_ids]
    stats = {
        "method": "M7b + Collision-Group Sinkhorn Refinement",
        "num_items": len(item_ids),
        "source_sid_dir": str(sid_dir),
        "mapping_path": str(output_dir / "sid_mapping.v2.json"),
        "has_metadata": has_metadata,
        "field_coverage": field_coverage(item_rows),
        "query_interaction_count": count_summary(query_counts),
        "rec_neighbor_count": count_summary(neighbor_counts),
        "codebook_sizes": {
            "prefix": checkpoint["prefix_sizes"],
            "search_suffix": checkpoint["search_sizes"],
            "rec_suffix": checkpoint["rec_sizes"],
        },
        "refinement": {
            "scope": "full-SID collision groups only",
            "fixed": "shared prefix and all suffix layers except final suffix layer",
            "search": refine_log.get("search"),
            "rec": refine_log.get("rec"),
        },
        "prefix": {
            "unique_count": len(prefix_counter),
            "bucket_size": count_summary(list(prefix_counter.values())),
            "usage": code_usage(prefix_codes, checkpoint["prefix_sizes"]),
        },
        "search_suffix_usage": code_usage(search_codes, checkpoint["search_sizes"]),
        "rec_suffix_usage": code_usage(rec_codes, checkpoint["rec_sizes"]),
        "search_sid_collision_collided_item_rate": collision_rate(search_sid_by_item),
        "rec_sid_collision_collided_item_rate": collision_rate(rec_sid_by_item),
        "search_sid_collision_gensar_rate": gensar_collision_stats(search_sid_by_item),
        "rec_sid_collision_gensar_rate": gensar_collision_stats(rec_sid_by_item),
        "redundancy_proxy_nmi": {
            "prefix_search_suffix": normalized_mutual_info_score(prefix_labels, search_suffix_labels),
            "prefix_rec_suffix": normalized_mutual_info_score(prefix_labels, rec_suffix_labels),
        },
        "embedding": {
            "text_encoder_model": params.text_encoder_model,
            "dim": int(common_x.shape[1]),
            "cache_dir": str(cache_dir),
        },
        "task_signal": {
            "search": search_signal_stats,
            "rec": rec_signal_stats,
            "search_nonzero_ratio": float((np.linalg.norm(x_search_signal, axis=1) > 0).mean()),
            "rec_nonzero_ratio": float((np.linalg.norm(x_rec_signal, axis=1) > 0).mean()),
        },
    }
    write_json(output_dir / "sid_stats.m7b_cg_sinkhorn.json", stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
