#!/usr/bin/env python
"""Build GenSAR item2query pickle from search interaction splits."""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gensar-data-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="item2query_div_nq-20.pkl")
    parser.add_argument("--max-queries-per-item", type=int, default=20)
    parser.add_argument(
        "--splits",
        default="src_train.pkl,src_val.pkl,src_test.pkl",
        help="Comma-separated dataset split files to aggregate.",
    )
    parser.add_argument(
        "--missing-fallback",
        choices=["error", "item_text"],
        default="error",
        help="How to handle src items that never occur in a search split.",
    )
    args = parser.parse_args()

    data_dir = args.gensar_data_dir
    query_vocab = load_pickle(data_dir / "vocab" / "query_vocab.pkl")
    src_item_set = set(load_pickle(data_dir / "vocab" / "src_item_set.pkl"))

    item2queries: dict[int, list[str]] = defaultdict(list)
    seen: dict[int, set[str]] = defaultdict(set)

    for split_name in [x.strip() for x in args.splits.split(",") if x.strip()]:
        frame = load_pickle(data_dir / "dataset" / split_name)
        for row in frame[["item_id", "query_id"]].itertuples(index=False):
            item_id = int(row.item_id)
            query_id = int(row.query_id)
            query = query_vocab[query_id]["query"]
            if query not in seen[item_id]:
                seen[item_id].add(query)
                item2queries[item_id].append(query)

    missing = sorted(src_item_set - set(item2queries))
    if missing:
        if args.missing_fallback == "error":
            raise RuntimeError(
                f"{len(missing)} src items have no query, first={missing[:10]}"
            )
        item_vocab = load_pickle(data_dir / "vocab" / "item_vocab.pkl")
        for item_id in missing:
            item = item_vocab[item_id]
            fallback = (item.get("title") or "").strip()
            if not fallback:
                fallback = (item.get("description") or "").strip()
            if not fallback:
                fallback = "product"
            item2queries[item_id].append(fallback)

    capped = {
        item_id: queries[: args.max_queries_per_item]
        for item_id, queries in sorted(item2queries.items())
        if item_id in src_item_set
    }

    output_path = data_dir / "vocab" / args.output_name
    with output_path.open("wb") as f:
        pickle.dump(capped, f)

    lens = [len(v) for v in capped.values()]
    print(f"saved: {output_path}")
    print(f"items: {len(capped)}")
    print(f"item-text fallbacks: {len(missing) if args.missing_fallback == 'item_text' else 0}")
    print(f"queries/item min={min(lens)} mean={sum(lens)/len(lens):.2f} max={max(lens)}")


if __name__ == "__main__":
    main()
