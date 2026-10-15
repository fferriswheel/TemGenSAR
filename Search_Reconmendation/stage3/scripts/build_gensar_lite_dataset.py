#!/usr/bin/env python3
import argparse
import ast
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_queries(path):
    queries = []
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for query_id, line in enumerate(f):
            query = line.strip()
            if query:
                queries.append({"query_id": query_id, "query": query})
    return queries


def iter_qrels(path, split, query_by_id):
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            user_query, _, item_id, label = parts
            if int(label) <= 0:
                continue
            user_id, query_id_text = user_query.rsplit("_", 1)
            query_id = int(query_id_text)
            yield {
                "interaction_id": f"{split}_{line_no}",
                "user_id": user_id,
                "query_id": query_id,
                "query": query_by_id.get(query_id, ""),
                "item_id": item_id,
                "label": int(label),
                "split": split,
            }


def iter_reviews(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            yield line_no, row


def clean_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(clean_text(x) for x in value if x is not None)
    return str(value).replace("\n", " ").strip()


def normalize_meta(row):
    categories = row.get("categories") or row.get("category") or []
    if categories and isinstance(categories[0], list):
        category_text = " ".join(str(x) for x in categories[0])
    elif isinstance(categories, list):
        category_text = " ".join(str(x) for x in categories)
    else:
        category_text = str(categories)
    title = clean_text(row.get("title"))
    description = clean_text(row.get("description"))
    brand = clean_text(row.get("brand")).replace("by ", "").strip()
    item_text = " ".join(x for x in [title, description, brand, category_text] if x)
    return {
        "item_id": row["asin"],
        "title": title,
        "description": description,
        "categories": categories,
        "brand": brand,
        "price": clean_text(row.get("price")),
        "sales_rank": row.get("salesRank", {}),
        "related": row.get("related", {}),
        "item_text": item_text,
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def count_qrels(train_qrels, test_qrels, query_by_id):
    item_qrel_count = Counter()
    item_query_count = defaultdict(Counter)
    for split, path in [("train", train_qrels), ("test", test_qrels)]:
        for row in iter_qrels(path, split, query_by_id):
            item_id = row["item_id"]
            item_qrel_count[item_id] += 1
            item_query_count[item_id][row["query_id"]] += 1
    return item_qrel_count, item_query_count


def count_reviews(review_path):
    item_review_count = Counter()
    user_review_count = Counter()
    for _, row in iter_reviews(review_path):
        item_id = row.get("asin")
        user_id = row.get("reviewerID")
        if not item_id or not user_id:
            continue
        item_review_count[item_id] += 1
        user_review_count[user_id] += 1
    return item_review_count, user_review_count


def parse_ratios(value):
    ratios = [float(part) for part in value.split(",")]
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios) or sum(ratios) <= 0:
        raise ValueError("--strata-ratios must contain three non-negative values")
    total = sum(ratios)
    return [ratio / total for ratio in ratios]


def allocate_quotas(total, ratios):
    raw = [total * ratio for ratio in ratios]
    quotas = [int(value) for value in raw]
    for index in sorted(range(len(raw)), key=lambda i: raw[i] - quotas[i], reverse=True)[: total - sum(quotas)]:
        quotas[index] += 1
    return quotas


def select_items(
    item_qrel_count,
    item_query_count,
    item_review_count,
    max_items,
    min_qrels,
    min_reviews,
    seed,
    selection_policy,
    strata_ratios,
):
    candidates = []
    for item_id, qrel_count in item_qrel_count.items():
        review_count = item_review_count.get(item_id, 0)
        if qrel_count < min_qrels or review_count < min_reviews:
            continue
        query_div = len(item_query_count[item_id])
        score = min(qrel_count, 200) * 2.0 + min(review_count, 200) + query_div * 5.0
        candidates.append((score, qrel_count, review_count, query_div, item_id))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    candidates.sort(reverse=True)
    max_items = min(max_items, len(candidates))
    if selection_policy == "top":
        selected = candidates[:max_items]
        details = {
            "policy": "top_score",
            "candidate_strata_sizes": [len(candidates), 0, 0],
            "target_strata_counts": [max_items, 0, 0],
            "selected_strata_counts": [len(selected), 0, 0],
        }
    else:
        boundaries = [0, len(candidates) // 3, 2 * len(candidates) // 3, len(candidates)]
        strata = [candidates[boundaries[i] : boundaries[i + 1]] for i in range(3)]
        quotas = allocate_quotas(max_items, strata_ratios)
        selected_by_stratum = []
        deficit = 0
        for stratum, quota in zip(strata, quotas):
            count = min(quota, len(stratum))
            selected_by_stratum.append(rng.sample(stratum, count))
            deficit += quota - count
        if deficit:
            selected_ids = {row[-1] for rows in selected_by_stratum for row in rows}
            remaining = [row for row in candidates if row[-1] not in selected_ids]
            selected_by_stratum[0].extend(remaining[:deficit])
        selected_ids = {row[-1] for rows in selected_by_stratum for row in rows}
        selected = [row for row in candidates if row[-1] in selected_ids]
        details = {
            "policy": "score_rank_stratified",
            "strata": ["head", "middle", "tail"],
            "strata_ratios": strata_ratios,
            "candidate_strata_sizes": [len(stratum) for stratum in strata],
            "target_strata_counts": quotas,
            "selected_strata_counts": [len(rows) for rows in selected_by_stratum],
        }
    return [item_id for *_rest, item_id in selected], candidates, details


def load_selected_meta(meta_path, selected_items):
    selected = set(selected_items)
    item_rows = {}
    with gzip.open(meta_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = ast.literal_eval(line)
            except Exception:
                continue
            item_id = row.get("asin")
            if item_id in selected:
                item_rows[item_id] = normalize_meta(row)
                if len(item_rows) == len(selected):
                    break
    for item_id in selected_items:
        item_rows.setdefault(
            item_id,
            {
                "item_id": item_id,
                "title": item_id,
                "description": "",
                "categories": [],
                "brand": "",
                "price": "",
                "sales_rank": {},
                "related": {},
                "item_text": item_id,
            },
        )
    return item_rows


def build_selected_reviews(review_path, selected_items, min_user_selected_interactions):
    selected = set(selected_items)
    by_user = defaultdict(list)
    for line_no, row in iter_reviews(review_path):
        item_id = row.get("asin")
        user_id = row.get("reviewerID")
        if item_id not in selected or not user_id:
            continue
        by_user[user_id].append(
            {
                "review_line_id": line_no,
                "user_id": user_id,
                "item_id": item_id,
                "rating": float(row.get("overall", 0.0) or 0.0),
                "timestamp": int(row.get("unixReviewTime", 0) or 0),
                "review_time": row.get("reviewTime", ""),
            }
        )
    kept = {}
    for user_id, rows in by_user.items():
        rows.sort(key=lambda x: (x["timestamp"], x["review_line_id"], x["item_id"]))
        if len(rows) < min_user_selected_interactions:
            continue
        for position, row in enumerate(rows):
            row["position"] = position
        kept[user_id] = rows
    return kept


def build_selected_qrels(
    train_qrels,
    test_qrels,
    query_by_id,
    selected_items,
    selected_reviews_by_user,
    timestamp_policy,
    dedupe_user_item,
    seed,
):
    selected = set(selected_items)
    kept_users = set(selected_reviews_by_user)
    rows = []
    for split, path in [("train", train_qrels), ("test", test_qrels)]:
        for row in iter_qrels(path, split, query_by_id):
            if row["item_id"] not in selected:
                continue
            if row["user_id"] not in kept_users:
                continue
            rows.append(row)
    if dedupe_user_item:
        rng = random.Random(seed)
        rng.shuffle(rows)
        deduped = {}
        for row in rows:
            deduped.setdefault((row["user_id"], row["item_id"]), row)
        rows = list(deduped.values())
    if timestamp_policy == "review_match":
        review_lookup = {}
        for user_id, review_rows in selected_reviews_by_user.items():
            for review_row in review_rows:
                key = (user_id, review_row["item_id"])
                previous = review_lookup.get(key)
                if previous is None or (review_row["timestamp"], review_row["review_line_id"]) < (
                    previous["timestamp"],
                    previous["review_line_id"],
                ):
                    review_lookup[key] = review_row
        matched = []
        for row in rows:
            review_row = review_lookup.get((row["user_id"], row["item_id"]))
            if review_row is None:
                continue
            row = dict(row)
            row["timestamp"] = review_row["timestamp"]
            row["matched_review_line_id"] = review_row["review_line_id"]
            matched.append(row)
        rows = matched
    rows.sort(key=lambda row: (row["user_id"], row.get("timestamp", 0), row["interaction_id"]))
    return rows


def summarize_counts(values):
    values = list(values)
    if not values:
        return {"min": 0, "p50": 0, "mean": 0, "p90": 0, "max": 0}
    values = sorted(values)
    return {
        "min": values[0],
        "p50": values[len(values) // 2],
        "mean": sum(values) / len(values),
        "p90": values[int(0.9 * (len(values) - 1))],
        "max": values[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--product-search-dir",
        default="/data/cjk/Search_Reconmendation/external/Amazon-Product-Search-Datasets/reviews_Electronics_5.json.gz",
    )
    parser.add_argument(
        "--review-path",
        default="/data/cjk/Search_Reconmendation/stage3/raw/amazon_legacy_5core/reviews_Electronics_5.json.gz",
    )
    parser.add_argument(
        "--meta-path",
        default="/data/cjk/Search_Reconmendation/stage3/raw/amazon_legacy_5core/meta_Electronics.json.gz",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/cjk/Search_Reconmendation/stage3/data/gensar_lite_5k",
    )
    parser.add_argument("--max-items", type=int, default=5000)
    parser.add_argument("--min-qrels", type=int, default=2)
    parser.add_argument("--min-reviews", type=int, default=2)
    parser.add_argument("--min-user-selected-interactions", type=int, default=2)
    parser.add_argument("--selection-policy", choices=["top", "stratified"], default="top")
    parser.add_argument("--strata-ratios", default="0.4,0.4,0.2")
    parser.add_argument(
        "--search-timestamp-policy",
        choices=["none", "review_match"],
        default="none",
    )
    parser.add_argument("--dedupe-search-user-item", action="store_true")
    parser.add_argument("--require-user-task-overlap", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    product_search_dir = Path(args.product_search_dir)
    output_dir = Path(args.output_dir)
    query_path = product_search_dir / "query_text.txt.gz"
    train_qrels = product_search_dir / "train.qrels.gz"
    test_qrels = product_search_dir / "test.qrels.gz"

    queries = read_queries(query_path)
    query_by_id = {row["query_id"]: row["query"] for row in queries}
    item_qrel_count, item_query_count = count_qrels(train_qrels, test_qrels, query_by_id)
    item_review_count, _ = count_reviews(args.review_path)
    strata_ratios = parse_ratios(args.strata_ratios)
    selected_items, candidates, selection_details = select_items(
        item_qrel_count,
        item_query_count,
        item_review_count,
        args.max_items,
        args.min_qrels,
        args.min_reviews,
        args.seed,
        args.selection_policy,
        strata_ratios,
    )
    item_rows = load_selected_meta(args.meta_path, selected_items)
    selected_reviews_by_user = build_selected_reviews(
        args.review_path,
        selected_items,
        args.min_user_selected_interactions,
    )
    search_rows = build_selected_qrels(
        train_qrels,
        test_qrels,
        query_by_id,
        selected_items,
        selected_reviews_by_user,
        args.search_timestamp_policy,
        args.dedupe_search_user_item,
        args.seed,
    )

    rec_retained_items = set()
    for rows in selected_reviews_by_user.values():
        rec_retained_items.update(row["item_id"] for row in rows)
    search_retained_items = {row["item_id"] for row in search_rows}
    retained_items = rec_retained_items & search_retained_items
    final_items = [item_id for item_id in selected_items if item_id in retained_items]
    final_item_set = set(final_items)

    if args.require_user_task_overlap:
        search_users = {row["user_id"] for row in search_rows if row["item_id"] in final_item_set}
    else:
        search_users = set(selected_reviews_by_user)

    rec_rows = []
    for user_id in sorted(selected_reviews_by_user):
        if user_id not in search_users:
            continue
        rows = [row for row in selected_reviews_by_user[user_id] if row["item_id"] in final_item_set]
        if len(rows) < args.min_user_selected_interactions:
            continue
        for position, row in enumerate(rows):
            row = dict(row)
            row["position"] = position
            row["split"] = "train"
            rec_rows.append(row)
    rec_rows.sort(key=lambda x: (x["user_id"], x["timestamp"], x["position"], x["item_id"]))
    search_rows = [row for row in search_rows if row["item_id"] in final_item_set]
    if args.require_user_task_overlap:
        rec_users = {row["user_id"] for row in rec_rows}
        search_rows = [row for row in search_rows if row["user_id"] in rec_users]
    users = sorted({row["user_id"] for row in rec_rows} | {row["user_id"] for row in search_rows})
    queries_seen = sorted({row["query_id"] for row in search_rows})

    write_jsonl(output_dir / "queries.jsonl", queries)
    write_jsonl(output_dir / "users.jsonl", ({"user_id": user_id} for user_id in users))
    write_jsonl(output_dir / "item_ids.jsonl", ({"item_id": item_id} for item_id in final_items))
    write_jsonl(output_dir / "items.jsonl", (item_rows[item_id] for item_id in final_items))
    write_jsonl(output_dir / "rec_interactions_chrono.jsonl", rec_rows)
    write_jsonl(output_dir / "search_interactions.jsonl", search_rows)

    rec_item_counts = Counter(row["item_id"] for row in rec_rows)
    search_item_counts = Counter(row["item_id"] for row in search_rows)
    stats = {
        "format_version": 1,
        "source": {
            "product_search_dir": str(product_search_dir),
            "review_path": args.review_path,
            "meta_path": args.meta_path,
        },
        "selection": {
            "requested_max_items": args.max_items,
            "min_qrels": args.min_qrels,
            "min_reviews": args.min_reviews,
            "min_user_selected_interactions": args.min_user_selected_interactions,
            "candidate_items_before_truncation": len(candidates),
            "seed": args.seed,
            "search_timestamp_policy": args.search_timestamp_policy,
            "dedupe_search_user_item": args.dedupe_search_user_item,
            "require_user_task_overlap": args.require_user_task_overlap,
            **selection_details,
            "description": "Items appear in both Product Search qrels and Amazon 5-core reviews; users are retained only if they have at least two selected review interactions.",
        },
        "num_items": len(final_items),
        "num_users": len(users),
        "num_queries": len(queries),
        "num_queries_in_search": len(queries_seen),
        "search_interactions": {
            "total": len(search_rows),
            "splits": dict(Counter(row["split"] for row in search_rows)),
            "per_item": summarize_counts(search_item_counts.values()),
        },
        "rec_interactions": {
            "total": len(rec_rows),
            "per_user": summarize_counts(len(rows) for rows in selected_reviews_by_user.values()),
            "per_item": summarize_counts(rec_item_counts.values()),
        },
        "overlap": {
            "items_with_search": len(search_item_counts),
            "items_with_rec": len(rec_item_counts),
            "items_with_both": len(set(search_item_counts) & set(rec_item_counts)),
        },
        "top_items": [
            {
                "item_id": item_id,
                "search_count": search_item_counts.get(item_id, 0),
                "rec_count": rec_item_counts.get(item_id, 0),
                "query_diversity": len(item_query_count[item_id]),
            }
            for item_id in final_items[:20]
        ],
    }
    write_json(output_dir / "stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
