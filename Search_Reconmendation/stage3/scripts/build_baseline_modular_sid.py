#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_items(data_dir, item_ids_arg, max_items):
    data_dir = Path(data_dir)
    items_path = data_dir / "items.jsonl"
    item_ids_path = Path(item_ids_arg) if item_ids_arg else data_dir / "item_ids.jsonl"

    item_text = {}
    if items_path.exists():
        for row in read_jsonl(items_path):
            item_text[row["item_id"]] = row.get("item_text") or " ".join(
                part
                for part in [
                    row.get("title", ""),
                    row.get("description", ""),
                    row.get("brand", ""),
                ]
                if part
            )
    else:
        for row in read_jsonl(item_ids_path):
            item_text[row["item_id"]] = row["item_id"]

    selected_item_ids = [row["item_id"] for row in read_jsonl(item_ids_path)]
    item_ids = sorted(item_id for item_id in selected_item_ids if item_id in item_text)
    if max_items > 0:
        item_ids = item_ids[:max_items]
    item_text = {item_id: item_text[item_id] for item_id in item_ids}
    return item_ids, item_text, items_path.exists()


def aggregate_search_queries(data_dir, item_ids, max_queries_per_item):
    item_set = set(item_ids)
    query_counter = defaultdict(Counter)
    for row in read_jsonl(Path(data_dir) / "search_interactions.jsonl"):
        if row["split"] != "train":
            continue
        item_id = row["item_id"]
        if item_id in item_set:
            query_counter[item_id][row["query"]] += 1

    agg = {}
    for item_id in item_ids:
        queries = [query for query, _ in query_counter[item_id].most_common(max_queries_per_item)]
        agg[item_id] = " ".join(queries) if queries else item_id
    return agg


def aggregate_rec_users(data_dir, item_ids, max_users_per_item):
    item_set = set(item_ids)
    users_by_item = defaultdict(Counter)
    rec_path = Path(data_dir) / "rec_interactions_chrono.jsonl"
    for row in read_jsonl(rec_path):
        item_id = row["item_id"]
        if item_id in item_set:
            users_by_item[item_id][row["user_id"]] += 1

    agg = {}
    for item_id in item_ids:
        users = [user for user, _ in users_by_item[item_id].most_common(max_users_per_item)]
        agg[item_id] = " ".join(users) if users else item_id
    return agg


def fit_predict(texts, num_clusters, method, seed):
    n_clusters = min(num_clusters, len(texts))
    if method == "tfidf":
        vectorizer = TfidfVectorizer(min_df=1, max_features=50000, norm="l2")
    elif method == "hashing":
        vectorizer = HashingVectorizer(n_features=4096, alternate_sign=False, norm="l2")
    else:
        raise ValueError(f"Unknown vectorizer method: {method}")

    x = vectorizer.fit_transform(texts) if method == "tfidf" else vectorizer.transform(texts)
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=min(4096, max(256, len(texts))),
        n_init="auto",
        max_iter=100,
    )
    return model.fit_predict(x).tolist()


def hierarchical_codes(texts, level1_k, level2_k, method, seed):
    level1 = fit_predict(texts, level1_k, method, seed)
    groups = defaultdict(list)
    for idx, code in enumerate(level1):
        groups[code].append(idx)

    level2 = [0] * len(texts)
    for code, indices in groups.items():
        if len(indices) <= 1:
            continue
        sub_texts = [texts[idx] for idx in indices]
        sub_codes = fit_predict(sub_texts, min(level2_k, len(indices)), method, seed + code + 1)
        for idx, sub_code in zip(indices, sub_codes):
            level2[idx] = sub_code
    return level1, level2


def collision_rate(sid_by_item):
    reverse = defaultdict(list)
    for item_id, sid in sid_by_item.items():
        reverse[tuple(sid)].append(item_id)
    collided_items = sum(len(items) for items in reverse.values() if len(items) > 1)
    return {
        "unique_sid_count": len(reverse),
        "collided_item_count": collided_items,
        "collision_rate": collided_items / max(len(sid_by_item), 1),
        "max_bucket_size": max((len(items) for items in reverse.values()), default=0),
    }


def build(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    item_ids, item_text, has_metadata = load_items(data_dir, args.item_ids, args.max_items)
    search_text = aggregate_search_queries(data_dir, item_ids, args.max_queries_per_item)
    rec_text = aggregate_rec_users(data_dir, item_ids, args.max_users_per_item)

    # Include item_id as a tie-break token for the smoke baseline. Without metadata,
    # many products share the exact same category-derived query text.
    shared_docs = [
        " ".join([item_text[item_id], search_text[item_id], item_id])
        if has_metadata
        else " ".join([search_text[item_id], item_id])
        for item_id in item_ids
    ]
    search_docs = [" ".join([search_text[item_id], item_id]) for item_id in item_ids]
    rec_docs = [" ".join([rec_text[item_id], item_id]) for item_id in item_ids]

    p1, p2 = hierarchical_codes(shared_docs, args.prefix_k1, args.prefix_k2, "tfidf", args.seed)
    s1, s2 = hierarchical_codes(search_docs, args.search_k1, args.search_k2, "tfidf", args.seed + 1000)
    r1, r2 = hierarchical_codes(rec_docs, args.rec_k1, args.rec_k2, "hashing", args.seed + 2000)

    sid_mapping = {}
    search_sid_by_item = {}
    rec_sid_by_item = {}
    prefix_counter = Counter()
    for idx, item_id in enumerate(item_ids):
        prefix = [f"<P1_{p1[idx]}>", f"<P2_{p2[idx]}>"]
        search_sid = prefix + [f"<S1_{s1[idx]}>", f"<S2_{s2[idx]}>"]
        rec_sid = prefix + [f"<R1_{r1[idx]}>", f"<R2_{r2[idx]}>"]
        sid_mapping[item_id] = {
            "search_sid": search_sid,
            "rec_sid": rec_sid,
            "prefix": prefix,
        }
        search_sid_by_item[item_id] = search_sid
        rec_sid_by_item[item_id] = rec_sid
        prefix_counter[tuple(prefix)] += 1

    mapping_path = output_dir / "sid_mapping.baseline.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(sid_mapping, f, indent=2, ensure_ascii=False)

    prefix_sizes = sorted(prefix_counter.values())
    stats = {
        "num_items": len(item_ids),
        "has_metadata": has_metadata,
        "mapping_path": str(mapping_path),
        "prefix_unique_count": len(prefix_counter),
        "prefix_max_bucket_size": max(prefix_sizes) if prefix_sizes else 0,
        "prefix_mean_bucket_size": sum(prefix_sizes) / max(len(prefix_sizes), 1),
        "search_sid_collision": collision_rate(search_sid_by_item),
        "rec_sid_collision": collision_rate(rec_sid_by_item),
        "params": vars(args),
        "warning": "Baseline smoke tokenizer using text/query/user aggregations, not the final gated residual SID model.",
    }
    with (output_dir / "sid_mapping.baseline.stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/sid",
    )
    parser.add_argument("--item-ids", default="")
    parser.add_argument("--max-items", type=int, default=10000)
    parser.add_argument("--prefix-k1", type=int, default=256)
    parser.add_argument("--prefix-k2", type=int, default=256)
    parser.add_argument("--search-k1", type=int, default=256)
    parser.add_argument("--search-k2", type=int, default=256)
    parser.add_argument("--rec-k1", type=int, default=256)
    parser.add_argument("--rec-k2", type=int, default=256)
    parser.add_argument("--max-queries-per-item", type=int, default=20)
    parser.add_argument("--max-users-per-item", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    stats = build(args)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
