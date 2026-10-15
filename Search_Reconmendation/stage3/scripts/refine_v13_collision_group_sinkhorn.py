#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from build_neural_rq_sid_v2 import (
    aggregate_search_queries,
    load_item_rows,
    parse_sizes,
    write_json,
)
from build_neural_rq_sid_v2_text_encoder_fusion_plc import (
    build_search_signal_embeddings,
    l2_normalize_np,
)
from build_neural_rq_sid_v3_aligned_delta import (
    load_behavior_signal,
    load_precomputed_embedding,
)
from build_neural_rq_sid_v7_anchored_cross_view import mapping_from_codes
from build_neural_rq_sid_v9_direct_private import DirectPrivateTokenizer, encode_private
from refine_m7b_collision_group_sinkhorn import refine_last_suffix_layer
from train_multiview_anchor_prefix_v13 import MultiViewAnchorRQVAE


def collision_stats(prefix, suffix):
    labels = [tuple(p) + tuple(s) for p, s in zip(prefix, suffix)]
    counts = Counter(labels)
    collided = sum(value for value in counts.values() if value > 1)
    return {
        "unique_sid_count": len(counts),
        "duplicate_fraction": (len(labels) - len(counts)) / len(labels),
        "collided_item_fraction": collided / len(labels),
        "collision_groups": sum(value > 1 for value in counts.values()),
        "max_bucket_size": max(counts.values()),
    }


def codes_from_mapping(mapping, item_ids, key):
    return np.asarray([mapping[item_id][key] for item_id in item_ids], dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=0.003)
    parser.add_argument("--sk-iters", type=int, default=100)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    params = argparse.Namespace(**checkpoint["params"])
    device = torch.device(args.device)
    data_dir = Path(params.data_dir)
    item_ids, _, _ = load_item_rows(data_dir, params.item_ids, params.max_items)
    _, query_counter = aggregate_search_queries(
        data_dir, item_ids, params.max_queries_per_item
    )
    common_x = load_precomputed_embedding(params.common_embedding, item_ids, "common")
    behavior_x, _ = load_behavior_signal(params.rec_behavior_embedding, item_ids)
    search_dim = int(getattr(params, "text_embedding_dim", 768))
    if getattr(params, "search_embedding", ""):
        search_x = load_precomputed_embedding(
            params.search_embedding, item_ids, "search"
        )
    else:
        search_x, _ = build_search_signal_embeddings(
            params,
            item_ids,
            query_counter,
            device,
            Path(params.embedding_cache_dir),
        )
    if search_x.shape[1] != search_dim:
        raise ValueError(
            f"Search embedding dimension mismatch: checkpoint={search_dim}, "
            f"loaded={search_x.shape[1]}"
        )

    anchor_checkpoint = torch.load(
        params.anchor_checkpoint, map_location="cpu", weights_only=False
    )
    anchor_args = argparse.Namespace(**anchor_checkpoint["params"])
    anchor = MultiViewAnchorRQVAE(
        common_x.shape[1],
        behavior_x.shape[1],
        search_dim,
        parse_sizes(params.prefix_codebook_sizes),
        anchor_args,
    )
    model = DirectPrivateTokenizer(
        anchor,
        common_x.shape[1],
        search_dim,
        behavior_x.shape[1],
        checkpoint["search_sizes"],
        checkpoint["rec_sizes"],
        params,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    lookup = np.load(params.anchor_lookup)
    if [str(value) for value in lookup["item_ids"].tolist()] != item_ids:
        raise ValueError("anchor lookup item order does not match dataset")
    z_shared = lookup["z_anchor"].astype(np.float32)
    q_shared = lookup["q_anchor"].astype(np.float32)
    semantic_base = l2_normalize_np(lookup["common_base"].astype(np.float32))
    _, rec_private, _, search_identity = encode_private(
        model,
        common_x,
        search_x,
        behavior_x,
        device,
        args.batch_size,
        return_parts=True,
        fixed_z_shared=z_shared,
        fixed_q_shared=q_shared,
        fixed_semantic_base=semantic_base,
    )

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    prefix = codes_from_mapping(mapping, item_ids, "prefix_codes")
    raw_search = codes_from_mapping(mapping, item_ids, "search_suffix_codes")
    raw_rec = codes_from_mapping(mapping, item_ids, "rec_suffix_codes")

    search_route = np.column_stack([prefix, raw_search[:, 0]])
    refined_identity, search_log = refine_last_suffix_layer(
        search_route,
        raw_search[:, 1:],
        search_identity,
        model.search_identity_quantizer.codebooks,
        args.epsilon,
        args.sk_iters,
        args.max_rounds,
    )
    refined_search = raw_search.copy()
    refined_search[:, 1] = np.asarray(refined_identity, dtype=np.int64)[:, 0]
    refined_rec, rec_log = refine_last_suffix_layer(
        prefix,
        raw_rec,
        rec_private,
        model.rec_quantizer.codebooks,
        args.epsilon,
        args.sk_iters,
        args.max_rounds,
    )
    refined_rec = np.asarray(refined_rec, dtype=np.int64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    refined_mapping = mapping_from_codes(
        item_ids, prefix, refined_search, refined_rec
    )
    write_json(args.output_dir / "sid_mapping.v13b_cg_sinkhorn.json", refined_mapping)
    stats = {
        "method": "v13b raw + GenSAR-style collision-group Sinkhorn",
        "num_items": len(item_ids),
        "params": {
            "epsilon": args.epsilon,
            "sk_iters": args.sk_iters,
            "max_rounds": args.max_rounds,
            "fixed_search": "prefix and relevance code",
            "fixed_rec": "prefix and first Rec suffix code",
        },
        "raw": {
            "search": collision_stats(prefix, raw_search),
            "rec": collision_stats(prefix, raw_rec),
        },
        "refined": {
            "search": collision_stats(prefix, refined_search),
            "rec": collision_stats(prefix, refined_rec),
        },
        "changed": {
            "search_item_fraction": float(
                np.any(refined_search != raw_search, axis=1).mean()
            ),
            "rec_item_fraction": float(np.any(refined_rec != raw_rec, axis=1).mean()),
        },
        "logs": {"search": search_log, "rec": rec_log},
    }
    write_json(args.output_dir / "sid_stats.v13b_cg_sinkhorn.json", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
