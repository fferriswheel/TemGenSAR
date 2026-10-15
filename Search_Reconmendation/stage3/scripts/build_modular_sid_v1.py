#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import normalize


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def flatten_categories(categories):
    if not categories:
        return ""
    terms = []
    for path in categories:
        if isinstance(path, list):
            terms.extend(str(part) for part in path if part)
        elif path:
            terms.append(str(path))
    return " ".join(terms)


def load_item_rows(data_dir, item_ids_arg, max_items):
    data_dir = Path(data_dir)
    item_ids_path = Path(item_ids_arg) if item_ids_arg else data_dir / "item_ids.jsonl"
    selected_ids = [row["item_id"] for row in read_jsonl(item_ids_path)]
    selected_set = set(selected_ids)

    rows_by_id = {}
    items_path = data_dir / "items.jsonl"
    if items_path.exists():
        for row in read_jsonl(items_path):
            item_id = row["item_id"]
            if item_id in selected_set:
                rows_by_id[item_id] = row
    else:
        for item_id in selected_ids:
            rows_by_id[item_id] = {
                "item_id": item_id,
                "title": "",
                "description": "",
                "categories": [],
                "brand": "",
                "item_text": item_id,
            }

    item_ids = [item_id for item_id in sorted(selected_ids) if item_id in rows_by_id]
    if max_items > 0:
        item_ids = item_ids[:max_items]
    item_rows = {item_id: rows_by_id[item_id] for item_id in item_ids}
    return item_ids, item_rows, items_path.exists()


def field_coverage(item_rows):
    total = len(item_rows)
    coverage = {}
    for field in ["title", "description", "brand", "categories", "item_text"]:
        count = 0
        for row in item_rows.values():
            value = row.get(field)
            if isinstance(value, str):
                count += bool(value.strip())
            else:
                count += bool(value)
        coverage[field] = {
            "count": count,
            "ratio": count / max(total, 1),
        }
    return coverage


def aggregate_search_queries(data_dir, item_ids, max_queries_per_item):
    item_set = set(item_ids)
    query_counter = defaultdict(Counter)
    for row in read_jsonl(Path(data_dir) / "search_interactions.jsonl"):
        if row["split"] != "train":
            continue
        item_id = row["item_id"]
        if item_id in item_set:
            query_counter[item_id][row["query"]] += 1

    query_docs = {}
    for item_id in item_ids:
        queries = [query for query, _ in query_counter[item_id].most_common(max_queries_per_item)]
        query_docs[item_id] = " ".join(queries)
    return query_docs, query_counter


def update_neighbor_counts(user_rows, item_set, neighbor_window, counters):
    items = [row["item_id"] for row in user_rows]
    for pos, item_id in enumerate(items):
        if item_id not in item_set:
            continue
        left = max(0, pos - neighbor_window)
        right = min(len(items), pos + neighbor_window + 1)
        for neighbor_pos in range(left, right):
            if neighbor_pos == pos:
                continue
            neighbor = items[neighbor_pos]
            counters[item_id][neighbor] += 1


def aggregate_rec_neighbors(data_dir, item_ids, neighbor_window, max_neighbors_per_item):
    item_set = set(item_ids)
    neighbor_counter = defaultdict(Counter)
    rec_path = Path(data_dir) / "rec_interactions_chrono.jsonl"

    cur_user = None
    cur_rows = []
    for row in read_jsonl(rec_path):
        user_id = row["user_id"]
        if cur_user is None:
            cur_user = user_id
        if user_id != cur_user:
            update_neighbor_counts(cur_rows, item_set, neighbor_window, neighbor_counter)
            cur_user = user_id
            cur_rows = []
        cur_rows.append(row)
    if cur_rows:
        update_neighbor_counts(cur_rows, item_set, neighbor_window, neighbor_counter)

    rec_docs = {}
    for item_id in item_ids:
        parts = []
        for neighbor, count in neighbor_counter[item_id].most_common(max_neighbors_per_item):
            parts.extend([neighbor] * min(count, 5))
        rec_docs[item_id] = " ".join(parts)
    return rec_docs, neighbor_counter


def make_docs(item_ids, item_rows, query_docs):
    shared_docs = []
    search_docs = []
    for item_id in item_ids:
        row = item_rows[item_id]
        title = row.get("title", "")
        description = row.get("description", "")
        brand = row.get("brand", "")
        categories = flatten_categories(row.get("categories", []))
        queries = query_docs.get(item_id, "")

        shared = " ".join(part for part in [title, categories, brand, queries, item_id] if part)
        search = " ".join(part for part in [title, description, queries, item_id] if part)
        shared_docs.append(shared if shared else item_id)
        search_docs.append(search if search else item_id)
    return shared_docs, search_docs


def vectorize_reduce_text(docs, max_features, svd_dim, seed):
    vectorizer = TfidfVectorizer(min_df=1, max_features=max_features, norm="l2")
    x = vectorizer.fit_transform(docs)
    dim = min(svd_dim, max(1, min(x.shape) - 1))
    if dim < 2:
        dense = x.toarray().astype(np.float32)
    else:
        svd = TruncatedSVD(n_components=dim, random_state=seed)
        dense = svd.fit_transform(x).astype(np.float32)
    return normalize(dense, norm="l2")


def vectorize_reduce_hash(docs, n_features, svd_dim, seed):
    vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False, norm="l2")
    x = vectorizer.transform(docs)
    dim = min(svd_dim, max(1, min(x.shape) - 1))
    if dim < 2:
        dense = x.toarray().astype(np.float32)
    else:
        svd = TruncatedSVD(n_components=dim, random_state=seed)
        dense = svd.fit_transform(x).astype(np.float32)
    return normalize(dense, norm="l2")


def squared_distances(x, centers):
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    c_norm = np.sum(centers * centers, axis=1, keepdims=True).T
    return x_norm + c_norm - 2 * x @ centers.T


def balanced_assign(x, centers):
    n_items = len(x)
    n_clusters = len(centers)
    base_capacity = n_items // n_clusters
    extra = n_items % n_clusters
    capacities = np.full(n_clusters, base_capacity, dtype=np.int64)
    capacities[:extra] += 1

    distances = squared_distances(x, centers)
    order = np.argsort(distances, axis=1)
    assigned = np.full(n_items, -1, dtype=np.int64)
    remaining = capacities.copy()

    # First pass: greedily let each item claim its nearest still-open cluster.
    item_order = np.argsort(distances[np.arange(n_items), order[:, 0]])
    for item_idx in item_order:
        for cluster_idx in order[item_idx]:
            if remaining[cluster_idx] > 0:
                assigned[item_idx] = cluster_idx
                remaining[cluster_idx] -= 1
                break

    # Defensive fallback for numerical/pathological cases.
    unassigned = np.where(assigned < 0)[0]
    if len(unassigned):
        open_clusters = np.repeat(np.arange(n_clusters), remaining)
        for item_idx, cluster_idx in zip(unassigned, open_clusters):
            assigned[item_idx] = cluster_idx

    return assigned


def fit_balanced_kmeans(x, n_clusters, seed, max_iter):
    n_clusters = min(n_clusters, len(x))
    if n_clusters <= 1:
        return [0] * len(x)

    init_model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=min(4096, max(256, len(x))),
        n_init="auto",
        max_iter=100,
    )
    init_model.fit(x)
    centers = init_model.cluster_centers_.astype(np.float32)
    labels = pairwise_distances_argmin_min(x, centers)[0]

    for _ in range(max_iter):
        new_labels = balanced_assign(x, centers)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for cluster_idx in range(n_clusters):
            members = x[labels == cluster_idx]
            if len(members):
                centers[cluster_idx] = members.mean(axis=0)
    return labels.tolist()


def fit_kmeans(x, n_clusters, seed, assignment, max_iter):
    n_clusters = min(n_clusters, len(x))
    if assignment == "balanced":
        return fit_balanced_kmeans(x, n_clusters, seed, max_iter)
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=min(4096, max(256, len(x))),
        n_init="auto",
        max_iter=100,
    )
    return model.fit_predict(x).tolist()


def hierarchical_dense_codes(x, level1_k, level2_k, seed, assignment, balanced_max_iter):
    level1 = fit_kmeans(x, level1_k, seed, assignment, balanced_max_iter)
    groups = defaultdict(list)
    for idx, code in enumerate(level1):
        groups[code].append(idx)

    level2 = [0] * len(x)
    for code, indices in groups.items():
        if len(indices) <= 1:
            continue
        sub_x = x[indices]
        sub_codes = fit_kmeans(
            sub_x,
            min(level2_k, len(indices)),
            seed + code + 1,
            assignment,
            balanced_max_iter,
        )
        for idx, sub_code in zip(indices, sub_codes):
            level2[idx] = sub_code
    return level1, level2


def residualize_by_prefix(x, prefix_codes):
    residual = np.array(x, copy=True)
    groups = defaultdict(list)
    for idx, code in enumerate(prefix_codes):
        groups[code].append(idx)
    for indices in groups.values():
        centroid = x[indices].mean(axis=0, keepdims=True)
        residual[indices] = x[indices] - centroid
    return normalize(residual, norm="l2")


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


def count_summary(values):
    values = sorted(values)
    if not values:
        return {"count": 0}

    def pct(q):
        return values[int(round((len(values) - 1) * q))]

    return {
        "count": len(values),
        "min": values[0],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": values[-1],
        "mean": sum(values) / len(values),
    }


def build(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    item_ids, item_rows, has_metadata = load_item_rows(data_dir, args.item_ids, args.max_items)
    query_docs, query_counter = aggregate_search_queries(data_dir, item_ids, args.max_queries_per_item)
    rec_docs_by_id, neighbor_counter = aggregate_rec_neighbors(
        data_dir,
        item_ids,
        args.neighbor_window,
        args.max_neighbors_per_item,
    )
    shared_docs, search_docs = make_docs(item_ids, item_rows, query_docs)
    rec_docs = [rec_docs_by_id.get(item_id, "") or item_id for item_id in item_ids]

    shared_x = vectorize_reduce_text(shared_docs, args.text_max_features, args.svd_dim, args.seed)
    search_x = vectorize_reduce_text(search_docs, args.text_max_features, args.svd_dim, args.seed + 100)
    rec_x = vectorize_reduce_hash(rec_docs, args.hash_features, args.svd_dim, args.seed + 200)

    p1, p2 = hierarchical_dense_codes(
        shared_x,
        args.prefix_k1,
        args.prefix_k2,
        args.seed,
        args.assignment,
        args.balanced_max_iter,
    )
    prefix_codes = list(zip(p1, p2))
    search_residual = residualize_by_prefix(search_x, prefix_codes)
    rec_residual = residualize_by_prefix(rec_x, prefix_codes)
    s1, s2 = hierarchical_dense_codes(
        search_residual,
        args.search_k1,
        args.search_k2,
        args.seed + 1000,
        args.assignment,
        args.balanced_max_iter,
    )
    r1, r2 = hierarchical_dense_codes(
        rec_residual,
        args.rec_k1,
        args.rec_k2,
        args.seed + 2000,
        args.assignment,
        args.balanced_max_iter,
    )

    sid_mapping = {}
    search_sid_by_item = {}
    rec_sid_by_item = {}
    prefix_counter = Counter()
    for idx, item_id in enumerate(item_ids):
        prefix = [f"<P1_{p1[idx]}>", f"<P2_{p2[idx]}>"]
        search_suffix = [f"<S1_{s1[idx]}>", f"<S2_{s2[idx]}>"]
        rec_suffix = [f"<R1_{r1[idx]}>", f"<R2_{r2[idx]}>"]
        search_sid = prefix + search_suffix
        rec_sid = prefix + rec_suffix
        sid_mapping[item_id] = {
            "prefix": prefix,
            "search_sid": search_sid,
            "rec_sid": rec_sid,
            "search_suffix": search_suffix,
            "rec_suffix": rec_suffix,
        }
        search_sid_by_item[item_id] = search_sid
        rec_sid_by_item[item_id] = rec_sid
        prefix_counter[tuple(prefix)] += 1

    mapping_path = output_dir / "sid_mapping.v1.json"
    write_json(mapping_path, sid_mapping)

    prefix_labels = [f"{a}_{b}" for a, b in prefix_codes]
    search_suffix_labels = [f"{a}_{b}" for a, b in zip(s1, s2)]
    rec_suffix_labels = [f"{a}_{b}" for a, b in zip(r1, r2)]

    query_counts = [sum(query_counter[item_id].values()) for item_id in item_ids]
    neighbor_counts = [sum(neighbor_counter[item_id].values()) for item_id in item_ids]
    prefix_sizes = list(prefix_counter.values())
    stats = {
        "num_items": len(item_ids),
        "has_metadata": has_metadata,
        "mapping_path": str(mapping_path),
        "field_coverage": field_coverage(item_rows),
        "query_interaction_count": count_summary(query_counts),
        "rec_neighbor_count": count_summary(neighbor_counts),
        "prefix": {
            "unique_count": len(prefix_counter),
            "bucket_size": count_summary(prefix_sizes),
        },
        "search_sid_collision": collision_rate(search_sid_by_item),
        "rec_sid_collision": collision_rate(rec_sid_by_item),
        "redundancy_proxy_nmi": {
            "prefix_search_suffix": normalized_mutual_info_score(prefix_labels, search_suffix_labels),
            "prefix_rec_suffix": normalized_mutual_info_score(prefix_labels, rec_suffix_labels),
        },
        "params": vars(args),
        "note": "v1 uses shared-prefix clustering plus prefix-centroid residual clustering for search and rec suffixes.",
    }
    write_json(output_dir / "sid_stats.v1.json", stats)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/sid_v1",
    )
    parser.add_argument("--item-ids", default="")
    parser.add_argument("--max-items", type=int, default=10000)
    parser.add_argument("--prefix-k1", type=int, default=256)
    parser.add_argument("--prefix-k2", type=int, default=256)
    parser.add_argument("--search-k1", type=int, default=256)
    parser.add_argument("--search-k2", type=int, default=256)
    parser.add_argument("--rec-k1", type=int, default=256)
    parser.add_argument("--rec-k2", type=int, default=256)
    parser.add_argument("--assignment", choices=["kmeans", "balanced"], default="kmeans")
    parser.add_argument("--balanced-max-iter", type=int, default=10)
    parser.add_argument("--max-queries-per-item", type=int, default=20)
    parser.add_argument("--neighbor-window", type=int, default=2)
    parser.add_argument("--max-neighbors-per-item", type=int, default=200)
    parser.add_argument("--text-max-features", type=int, default=50000)
    parser.add_argument("--hash-features", type=int, default=32768)
    parser.add_argument("--svd-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    stats = build(args)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
