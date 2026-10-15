#!/usr/bin/env python3
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def read_queries(path):
    queries = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for query_id, line in enumerate(f):
            query = line.strip()
            if query:
                queries.append({"query_id": query_id, "query": query})
    return queries


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_qrels(path, split, query_by_id):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 4:
                raise ValueError(f"Invalid qrels line {line_no + 1} in {path}: {line!r}")

            user_query, _, item_id, label = parts
            user_id, query_id_text = user_query.rsplit("_", 1)
            query_id = int(query_id_text)

            yield {
                "interaction_id": f"{split}_{line_no}",
                "user_id": user_id,
                "query_id": query_id,
                "query": query_by_id[query_id],
                "item_id": item_id,
                "label": int(label),
                "split": split,
            }


def write_limited_jsonl(path, rows, limit):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if count >= limit:
                break
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def convert(input_dir, output_dir, sample_size):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_path = input_dir / "query_text.txt.gz"
    train_qrels_path = input_dir / "train.qrels.gz"
    test_qrels_path = input_dir / "test.qrels.gz"

    queries = read_queries(query_path)
    query_by_id = {row["query_id"]: row["query"] for row in queries}
    write_jsonl(output_dir / "queries.jsonl", queries)

    stats = {
        "num_queries": len(queries),
        "splits": {},
    }
    users = set()
    items = set()
    query_ids_seen = set()
    split_counters = {}

    search_path = output_dir / "search_interactions.jsonl"
    with search_path.open("w", encoding="utf-8") as out:
        for split, qrels_path in [("train", train_qrels_path), ("test", test_qrels_path)]:
            counter = Counter()
            for row in iter_qrels(qrels_path, split, query_by_id):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                users.add(row["user_id"])
                items.add(row["item_id"])
                query_ids_seen.add(row["query_id"])
                counter["interactions"] += 1
            split_counters[split] = dict(counter)

    write_jsonl(output_dir / "users.jsonl", [{"user_id": user_id} for user_id in sorted(users)])
    write_jsonl(output_dir / "item_ids.jsonl", [{"item_id": item_id} for item_id in sorted(items)])

    stats["num_users"] = len(users)
    stats["num_items"] = len(items)
    stats["num_queries_in_qrels"] = len(query_ids_seen)
    stats["splits"] = split_counters
    stats["source"] = str(input_dir)
    stats["format_version"] = 1

    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    sample_count = write_limited_jsonl(
        output_dir / "sample" / "search_interactions.sample.jsonl",
        (json.loads(line) for line in search_path.open("r", encoding="utf-8")),
        sample_size,
    )

    sample_stats = {
        "sample_size": sample_count,
        "source": str(search_path),
    }
    with (output_dir / "sample" / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(sample_stats, f, indent=2, ensure_ascii=False)

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="/data/cjk/Search_Reconmendation/external/Amazon-Product-Search-Datasets/reviews_Electronics_5.json.gz",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics",
    )
    parser.add_argument("--sample-size", type=int, default=10000)
    args = parser.parse_args()

    stats = convert(args.input_dir, args.output_dir, args.sample_size)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
