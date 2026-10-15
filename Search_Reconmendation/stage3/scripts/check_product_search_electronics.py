#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def percentile(sorted_values, q):
    if not sorted_values:
        return 0
    idx = int(round((len(sorted_values) - 1) * q))
    return sorted_values[idx]


def summarize_counts(counter):
    values = sorted(counter.values())
    if not values:
        return {
            "num_keys": 0,
            "min": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            "mean": 0,
        }
    return {
        "num_keys": len(values),
        "min": values[0],
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": values[-1],
        "mean": sum(values) / len(values),
    }


def top_items(counter, k=10):
    return [{"key": key, "count": count} for key, count in counter.most_common(k)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    query_path = data_dir / "queries.jsonl"
    interactions_path = data_dir / "search_interactions.jsonl"
    output_path = data_dir / "sanity_report.json"

    queries = {row["query_id"]: row["query"] for row in read_jsonl(query_path)}

    split_rows = Counter()
    split_users = defaultdict(set)
    split_items = defaultdict(set)
    split_queries = defaultdict(set)
    user_counts = defaultdict(Counter)
    item_counts = defaultdict(Counter)
    query_counts = defaultdict(Counter)
    user_item_seen_order = defaultdict(list)
    duplicate_keys = Counter()
    seen_by_split = defaultdict(set)
    bad_rows = []

    for row_idx, row in enumerate(read_jsonl(interactions_path)):
        split = row.get("split")
        user_id = row.get("user_id")
        item_id = row.get("item_id")
        query_id = row.get("query_id")
        label = row.get("label")

        if (
            split not in {"train", "test"}
            or not user_id
            or not item_id
            or query_id not in queries
            or label not in {0, 1}
        ):
            bad_rows.append({"row_idx": row_idx, "row": row})
            if len(bad_rows) >= 20:
                continue

        split_rows[split] += 1
        split_users[split].add(user_id)
        split_items[split].add(item_id)
        split_queries[split].add(query_id)
        user_counts[split][user_id] += 1
        item_counts[split][item_id] += 1
        query_counts[split][query_id] += 1

        key = (user_id, query_id, item_id)
        duplicate_keys[(split, key)] += 1
        seen_by_split[split].add(key)

        if split == "train":
            # This order is qrels file order, not chronological order.
            user_item_seen_order[user_id].append(item_id)

    train_keys = seen_by_split["train"]
    test_keys = seen_by_split["test"]
    overlap_keys = train_keys & test_keys

    train_users = split_users["train"]
    test_users = split_users["test"]
    train_items = split_items["train"]
    test_items = split_items["test"]
    train_queries = split_queries["train"]
    test_queries = split_queries["test"]

    rec_seq_len_counter = Counter()
    rec_unique_len_counter = Counter()
    for user_id, items in user_item_seen_order.items():
        rec_seq_len_counter[user_id] = len(items)
        rec_unique_len_counter[user_id] = len(set(items))

    duplicate_summary = Counter()
    for (split, _), count in duplicate_keys.items():
        if count > 1:
            duplicate_summary[split] += 1

    report = {
        "format": {
            "bad_row_count_capped": len(bad_rows),
            "bad_row_examples": bad_rows[:20],
        },
        "global": {
            "num_queries_file": len(queries),
            "num_interactions": sum(split_rows.values()),
        },
        "splits": {
            split: {
                "interactions": split_rows[split],
                "users": len(split_users[split]),
                "items": len(split_items[split]),
                "queries": len(split_queries[split]),
                "user_interactions": summarize_counts(user_counts[split]),
                "item_interactions": summarize_counts(item_counts[split]),
                "query_interactions": summarize_counts(query_counts[split]),
                "top_users": top_items(user_counts[split], args.top_k),
                "top_items": top_items(item_counts[split], args.top_k),
                "top_queries": [
                    {
                        "query_id": query_id,
                        "query": queries[query_id],
                        "count": count,
                    }
                    for query_id, count in query_counts[split].most_common(args.top_k)
                ],
            }
            for split in ["train", "test"]
        },
        "overlap": {
            "train_test_user_overlap": len(train_users & test_users),
            "test_user_in_train_ratio": len(train_users & test_users) / max(len(test_users), 1),
            "train_test_item_overlap": len(train_items & test_items),
            "test_item_in_train_ratio": len(train_items & test_items) / max(len(test_items), 1),
            "train_test_query_overlap": len(train_queries & test_queries),
            "test_query_in_train_ratio": len(train_queries & test_queries) / max(len(test_queries), 1),
            "exact_user_query_item_overlap": len(overlap_keys),
        },
        "duplicates": {
            "duplicate_user_query_item_keys_by_split": dict(duplicate_summary),
        },
        "qrels_derived_rec_sequences": {
            "warning": "These sequences use qrels file order, not chronological order.",
            "sequence_length": summarize_counts(rec_seq_len_counter),
            "unique_item_length": summarize_counts(rec_unique_len_counter),
            "users_len_ge_2": sum(1 for value in rec_seq_len_counter.values() if value >= 2),
            "users_len_ge_5": sum(1 for value in rec_seq_len_counter.values() if value >= 5),
            "users_len_ge_10": sum(1 for value in rec_seq_len_counter.values() if value >= 10),
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
