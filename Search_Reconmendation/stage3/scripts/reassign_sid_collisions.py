#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.data import DataLoader, TensorDataset

from build_neural_rq_sid_v2 import (
    aggregate_rec_neighbors,
    aggregate_search_queries,
    code_usage,
    collision_rate,
    count_summary,
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


def sid_key(prefix_codes, suffix_codes):
    return tuple(int(x) for x in prefix_codes) + tuple(int(x) for x in suffix_codes)


def greedy_reassign(item_ids, prefix_codes, original_suffix_codes, residuals, codebooks, beam_size):
    rows = []
    for item_idx, item_id in enumerate(item_ids):
        paths = rq_top_paths(residuals[item_idx], codebooks, beam_size)
        original = tuple(int(x) for x in original_suffix_codes[item_idx])
        if original not in [path for path, _ in paths]:
            paths.append((original, 0.0))
        original_score = next((score for path, score in paths if path == original), 0.0)
        rows.append(
            {
                "item_idx": item_idx,
                "item_id": item_id,
                "prefix": tuple(int(x) for x in prefix_codes[item_idx]),
                "original": original,
                "original_score": original_score,
                "paths": paths,
            }
        )

    buckets = defaultdict(list)
    for row in rows:
        buckets[sid_key(row["prefix"], row["original"])].append(row)

    ordered = []
    for bucket_rows in buckets.values():
        bucket_rows.sort(key=lambda row: (row["original_score"], row["item_id"]))
        ordered.append(bucket_rows[0])
    for bucket_rows in buckets.values():
        ordered.extend(bucket_rows[1:])

    assigned = {}
    used = set()
    changed = 0
    failed = 0
    rank_sum = 0
    for row in ordered:
        selected = None
        selected_rank = None
        for rank, (path, _) in enumerate(row["paths"]):
            key = sid_key(row["prefix"], path)
            if key not in used:
                selected = path
                selected_rank = rank
                break
        if selected is None:
            failed += 1
            selected = row["original"]
            selected_rank = len(row["paths"])
        used.add(sid_key(row["prefix"], selected))
        changed += int(selected != row["original"])
        rank_sum += selected_rank
        assigned[row["item_idx"]] = list(selected)

    return [assigned[idx] for idx in range(len(item_ids))], {
        "changed": changed,
        "unchanged": len(item_ids) - changed,
        "failed": failed,
        "avg_selected_rank": rank_sum / max(len(item_ids), 1),
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
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def compute_residuals(model, x_common, x_search_signal, x_rec_signal, prefix_codes, device, batch_size):
    search_latents = []
    rec_latents = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_common), torch.from_numpy(x_search_signal), torch.from_numpy(x_rec_signal)),
        batch_size=batch_size,
        shuffle=False,
    )
    for batch_common, batch_search, batch_rec in loader:
        batch_common = batch_common.to(device)
        z_shared = model.shared.encoder(batch_common)
        z_search, z_rec = model.task_latents(z_shared, batch_common, batch_search.to(device), batch_rec.to(device))
        search_latents.append(z_search.cpu().numpy())
        rec_latents.append(z_rec.cpu().numpy())
    q_prefix = reconstruct_from_rq_codes(prefix_codes, model.shared.quantizer.codebooks)
    return np.concatenate(search_latents, axis=0) - q_prefix, np.concatenate(rec_latents, axis=0) - q_prefix


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics")
    parser.add_argument("--sid-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--item-ids", default="")
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--beam-size", type=int, default=128)
    parser.add_argument("--tasks", choices=["search", "rec", "both"], default="both")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sid_dir = Path(args.sid_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(sid_dir / "text_encoder_fusion_plc_sid_model.pt", map_location="cpu")
    params = argparse.Namespace(**checkpoint["params"])
    params.data_dir = str(data_dir)
    params.item_ids = args.item_ids or params.item_ids
    params.max_items = args.max_items
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    item_ids, item_rows, _ = load_item_rows(data_dir, params.item_ids, params.max_items)
    query_docs, query_counter = aggregate_search_queries(data_dir, item_ids, params.max_queries_per_item)
    rec_docs_by_id, neighbor_counter = aggregate_rec_neighbors(
        data_dir,
        item_ids,
        params.neighbor_window,
        params.max_neighbors_per_item,
    )
    text_docs, _, _ = make_docs(item_ids, item_rows, query_docs, rec_docs_by_id)
    cache_dir = Path(params.embedding_cache_dir) if params.embedding_cache_dir else data_dir / "text_embedding_cache"
    common_x = encode_text_docs(params, text_docs, "common", device, cache_dir)
    params.text_embedding_dim = int(common_x.shape[1])
    x_search_signal, _ = build_search_signal_embeddings(params, item_ids, query_counter, device, cache_dir)
    x_rec_signal, _ = build_rec_signal_embeddings(item_ids, common_x, neighbor_counter, params.max_neighbors_per_item)

    model = restore_model(params, checkpoint, common_x.shape[1], device)
    original = load_json(sid_dir / "sid_mapping.v2.json")
    prefix_codes = [original[item_id]["prefix_codes"] for item_id in item_ids]
    search_codes = [original[item_id]["search_suffix_codes"] for item_id in item_ids]
    rec_codes = [original[item_id]["rec_suffix_codes"] for item_id in item_ids]

    r_search, r_rec = compute_residuals(
        model,
        common_x.astype(np.float32),
        x_search_signal.astype(np.float32),
        x_rec_signal.astype(np.float32),
        prefix_codes,
        device,
        params.batch_size,
    )

    logs = {}
    if args.tasks in ("search", "both"):
        search_codes, logs["search"] = greedy_reassign(
            item_ids,
            prefix_codes,
            search_codes,
            r_search,
            model.search_quantizer.codebooks,
            args.beam_size,
        )
    if args.tasks in ("rec", "both"):
        rec_codes, logs["rec"] = greedy_reassign(
            item_ids,
            prefix_codes,
            rec_codes,
            r_rec,
            model.rec_quantizer.codebooks,
            args.beam_size,
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

    write_json(output_dir / "sid_mapping.v2.json", sid_mapping)
    write_json(output_dir / "sid_mapping.v1.json", sid_mapping)
    write_json(output_dir / "sid_mapping.v2_reassigned.json", sid_mapping)

    prefix_labels = ["_".join(map(str, codes)) for codes in prefix_codes]
    search_suffix_labels = ["_".join(map(str, codes)) for codes in search_codes]
    rec_suffix_labels = ["_".join(map(str, codes)) for codes in rec_codes]
    stats = {
        "num_items": len(item_ids),
        "source_sid_dir": str(sid_dir),
        "beam_size": args.beam_size,
        "reassign_logs": logs,
        "prefix": {
            "unique_count": len(prefix_counter),
            "bucket_size": count_summary(list(prefix_counter.values())),
            "usage": code_usage(prefix_codes, checkpoint["prefix_sizes"]),
        },
        "search_suffix_usage": code_usage(search_codes, checkpoint["search_sizes"]),
        "rec_suffix_usage": code_usage(rec_codes, checkpoint["rec_sizes"]),
        "search_sid_collision": collision_rate(search_sid_by_item),
        "rec_sid_collision": collision_rate(rec_sid_by_item),
        "redundancy_proxy_nmi": {
            "prefix_search_suffix": normalized_mutual_info_score(prefix_labels, search_suffix_labels),
            "prefix_rec_suffix": normalized_mutual_info_score(prefix_labels, rec_suffix_labels),
        },
        "note": "Collision-aware reassignment keeps SID length fixed and greedily moves collided items to their nearest unused full prefix+suffix path.",
    }
    write_json(output_dir / "sid_stats.v2_reassigned.json", stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
