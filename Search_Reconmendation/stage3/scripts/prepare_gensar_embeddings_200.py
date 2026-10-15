#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_neural_rq_sid_v2 import aggregate_rec_neighbors, aggregate_search_queries, load_item_rows, make_docs
from build_neural_rq_sid_v2_text_encoder_fusion_plc import (
    build_rec_signal_embeddings,
    build_search_signal_embeddings,
    encode_text_docs,
    l2_normalize_np,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics",
    )
    parser.add_argument(
        "--gensar-data-dir",
        default="/data/cjk/external/GenSAR/data/SR200_M7b_collision005_unique",
    )
    parser.add_argument("--semantic-name", default="ours_signal_semantic")
    parser.add_argument("--collaborate-name", default="ours_signal_collaborate")
    parser.add_argument("--max-queries-per-item", type=int, default=20)
    parser.add_argument("--neighbor-window", type=int, default=2)
    parser.add_argument("--max-neighbors-per-item", type=int, default=200)
    parser.add_argument("--text-encoder-model", default="intfloat/e5-small-v2")
    parser.add_argument("--text-encoder-prefix", default="passage: ")
    parser.add_argument("--query-encoder-prefix", default="query: ")
    parser.add_argument("--text-encoder-max-length", type=int, default=256)
    parser.add_argument("--text-encoder-batch-size", type=int, default=64)
    parser.add_argument("--task-scale", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embedding-cache-dir", default="")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    gensar_data_dir = Path(args.gensar_data_dir)
    id_maps = json.loads((gensar_data_dir / "id_maps.json").read_text(encoding="utf-8"))
    int_to_asin = {int(k): v for k, v in id_maps["int_to_asin"].items()}
    ordered_item_ids = [int_to_asin[i] for i in range(1, len(int_to_asin) + 1)]

    item_ids, item_rows, _ = load_item_rows(data_dir, "", 0)
    item_rows = {item_id: item_rows[item_id] for item_id in ordered_item_ids}
    item_ids = ordered_item_ids

    query_docs, query_counter = aggregate_search_queries(data_dir, item_ids, args.max_queries_per_item)
    rec_docs_by_id, neighbor_counter = aggregate_rec_neighbors(
        data_dir,
        item_ids,
        args.neighbor_window,
        args.max_neighbors_per_item,
    )
    text_docs, _, _ = make_docs(item_ids, item_rows, query_docs, rec_docs_by_id)

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    cache_dir = Path(args.embedding_cache_dir) if args.embedding_cache_dir else data_dir / "text_embedding_cache"

    common_x = encode_text_docs(args, text_docs, "common", device, cache_dir)
    args.text_embedding_dim = int(common_x.shape[1])
    search_signal, search_stats = build_search_signal_embeddings(args, item_ids, query_counter, device, cache_dir)
    rec_signal, rec_stats = build_rec_signal_embeddings(
        item_ids,
        common_x,
        neighbor_counter,
        args.max_neighbors_per_item,
    )

    semantic_x = l2_normalize_np(common_x + args.task_scale * search_signal)
    collaborate_x = l2_normalize_np(common_x + args.task_scale * rec_signal)

    emb_dir = gensar_data_dir / "emb"
    emb_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(semantic_x.astype(np.float32)), emb_dir / f"{args.semantic_name}.pt")
    torch.save(torch.from_numpy(collaborate_x.astype(np.float32)), emb_dir / f"{args.collaborate_name}.pt")
    stats = {
        "items": len(item_ids),
        "dim": int(semantic_x.shape[1]),
        "semantic_file": f"emb/{args.semantic_name}.pt",
        "collaborate_file": f"emb/{args.collaborate_name}.pt",
        "task_scale": args.task_scale,
        "search_signal": search_stats,
        "rec_signal": rec_stats,
    }
    (gensar_data_dir / "emb" / "ours_signal_embedding_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
