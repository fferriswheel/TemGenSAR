#!/usr/bin/env python3
import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from build_neural_rq_sid_v2 import (
    aggregate_rec_neighbors,
    aggregate_search_queries,
    load_item_rows,
    make_docs,
    rq_assign_codes_np,
    write_json,
)
from build_neural_rq_sid_v2_text_encoder_fusion_plc import (
    SharedRQVAE,
    build_rec_signal_embeddings,
    build_search_signal_embeddings,
    encode_text_docs,
    token_list,
)
from build_neural_rq_sid_v3_aligned_delta import (
    AlignedDeltaTokenizer,
    encode_task_deltas,
    load_behavior_signal,
    load_precomputed_embedding,
)
from refine_m7b_collision_group_sinkhorn import refine_last_suffix_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sid-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument(
        "--prefix-assignment", choices=["checkpoint", "argmin", "balanced"], default="checkpoint"
    )
    parser.add_argument("--epsilon", type=float, default=0.003)
    parser.add_argument("--sk-iters", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    sid_dir = Path(args.sid_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(sid_dir / "aligned_delta_sid_model.pt", map_location="cpu", weights_only=False)
    params = Namespace(**checkpoint["params"])
    params.data_dir = args.data_dir
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    item_ids, item_rows, _ = load_item_rows(data_dir, params.item_ids, params.max_items)
    query_docs, query_counter = aggregate_search_queries(data_dir, item_ids, params.max_queries_per_item)
    rec_docs, neighbor_counter = aggregate_rec_neighbors(
        data_dir, item_ids, params.neighbor_window, params.max_neighbors_per_item
    )
    text_docs, _, _ = make_docs(item_ids, item_rows, query_docs, rec_docs)
    cache_dir = Path(params.embedding_cache_dir) if params.embedding_cache_dir else data_dir / "text_embedding_cache"
    if getattr(params, "common_embedding", ""):
        common_x = load_precomputed_embedding(params.common_embedding, item_ids, "common")
    else:
        common_x = encode_text_docs(params, text_docs, "common", device, cache_dir)
    search_signal, _ = build_search_signal_embeddings(params, item_ids, query_counter, device, cache_dir)
    if params.rec_signal == "behavior":
        rec_signal, _ = load_behavior_signal(params.rec_behavior_embedding, item_ids)
    else:
        rec_signal, _ = build_rec_signal_embeddings(
            item_ids, common_x, neighbor_counter, params.max_neighbors_per_item
        )
    state = checkpoint["model_state_dict"]
    search_target_dim = int(state["search_decoder.4.weight"].shape[0])
    rec_target_dim = int(state["rec_decoder.4.weight"].shape[0])
    shared = SharedRQVAE(
        input_dim=common_x.shape[1], hidden_dim=params.hidden_dim, latent_dim=params.latent_dim,
        prefix_sizes=checkpoint["prefix_sizes"], beta=params.vq_beta,
        dropout=params.dropout, balance_temperature=params.balance_temperature,
    )
    model = AlignedDeltaTokenizer(
        shared=shared, common_dim=common_x.shape[1], search_signal_dim=search_signal.shape[1],
        rec_signal_dim=rec_signal.shape[1], hidden_dim=params.hidden_dim, latent_dim=params.latent_dim,
        search_sizes=checkpoint["search_sizes"], rec_sizes=checkpoint["rec_sizes"],
        beta=params.vq_beta, balance_temperature=params.balance_temperature,
        lora_rank=params.lora_rank, lora_alpha=params.lora_alpha, augment_scale=params.augment_scale,
        gate_bias=params.gate_bias, delta_max_ratio=params.delta_max_ratio,
        search_target_dim=search_target_dim, rec_target_dim=rec_target_dim, dropout=params.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if args.prefix_assignment == "checkpoint":
        prefix_codes = np.asarray(checkpoint["fixed_prefix_codes"], dtype=np.int64)
    else:
        with torch.no_grad():
            z_shared = model.shared.encoder(torch.from_numpy(common_x).to(device)).cpu().numpy()
        prefix_codes = np.asarray(
            rq_assign_codes_np(z_shared, model.shared.quantizer.codebooks, args.prefix_assignment),
            dtype=np.int64,
        )
    search_delta, rec_delta, _, _ = encode_task_deltas(
        model, common_x.astype(np.float32), search_signal, rec_signal, device, params.batch_size
    )
    search_codes = np.asarray(rq_assign_codes_np(search_delta, model.search_quantizer.codebooks, "argmin"))
    rec_codes = np.asarray(rq_assign_codes_np(rec_delta, model.rec_quantizer.codebooks, "argmin"))
    search_codes, search_log = refine_last_suffix_layer(
        prefix_codes, search_codes, search_delta, model.search_quantizer.codebooks,
        args.epsilon, args.sk_iters, args.max_rounds,
    )
    rec_codes, rec_log = refine_last_suffix_layer(
        prefix_codes, rec_codes, rec_delta, model.rec_quantizer.codebooks,
        args.epsilon, args.sk_iters, args.max_rounds,
    )
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
    write_json(output_dir / "sid_mapping.cg_sinkhorn.json", mapping)
    summary = {
        "prefix_assignment": args.prefix_assignment,
        "prefix_unique_count": len({tuple(row) for row in prefix_codes.tolist()}),
        "search": search_log,
        "rec": rec_log,
    }
    write_json(output_dir / "sid_stats.cg_sinkhorn.json", summary)
    print(json.dumps({
        "search_final": search_log["rounds"][-1] if search_log["rounds"] else {},
        "rec_final": rec_log["rounds"][-1] if rec_log["rounds"] else {},
    }, indent=2))


if __name__ == "__main__":
    main()
