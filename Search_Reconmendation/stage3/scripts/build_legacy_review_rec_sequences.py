#!/usr/bin/env python3
import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_train_review_line_ids(path):
    line_ids = set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                line_ids.add(int(line.split("_", 1)[1]))
    return line_ids


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(values):
    values = sorted(values)
    if not values:
        return {"count": 0}

    def pct(q):
        idx = int(round((len(values) - 1) * q))
        return values[idx]

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


def build(raw_reviews, train_review_ids, output_dir, max_smoke_examples, max_history_len):
    raw_reviews = Path(raw_reviews)
    output_dir = Path(output_dir)
    train_ids = load_train_review_line_ids(train_review_ids)

    train_by_user = defaultdict(list)
    total_reviews = 0
    train_reviews = 0
    fields_counter = Counter()

    with gzip.open(raw_reviews, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            total_reviews += 1
            if line_no not in train_ids:
                continue
            obj = json.loads(line)
            row = {
                "review_line_id": line_no,
                "user_id": obj["reviewerID"],
                "item_id": obj["asin"],
                "rating": obj.get("overall"),
                "timestamp": obj.get("unixReviewTime"),
                "review_time": obj.get("reviewTime"),
                "split": "train",
            }
            for field in row:
                if row[field] is not None:
                    fields_counter[field] += 1
            train_by_user[row["user_id"]].append(row)
            train_reviews += 1

    rec_rows = []
    for user_id in sorted(train_by_user):
        rows = sorted(
            train_by_user[user_id],
            key=lambda row: (
                row["timestamp"] if row["timestamp"] is not None else -1,
                row["review_line_id"],
            ),
        )
        for pos, row in enumerate(rows):
            row = dict(row)
            row["position"] = pos
            rec_rows.append(row)

    rec_path = output_dir / "rec_interactions_chrono.jsonl"
    write_jsonl(rec_path, rec_rows)

    smoke_examples = []
    by_user = defaultdict(list)
    for row in rec_rows:
        by_user[row["user_id"]].append(row["item_id"])

    for user_id in sorted(by_user):
        items = by_user[user_id]
        if len(items) < 2:
            continue
        for pos in range(1, len(items)):
            history = items[max(0, pos - max_history_len) : pos]
            target = items[pos]
            smoke_examples.append(
                {
                    "task": "rec",
                    "source": {
                        "user_id": user_id,
                        "history_item_ids": history,
                        "history_order": "unixReviewTime",
                    },
                    "input_text": "<REC> " + " ".join(history),
                    "target_item_id": target,
                    "split": "train",
                }
            )
            if len(smoke_examples) >= max_smoke_examples:
                break
        if len(smoke_examples) >= max_smoke_examples:
            break

    write_jsonl(output_dir / "smoke" / "rec_chrono_smoke_examples.jsonl", smoke_examples)

    seq_lens = [len(rows) for rows in by_user.values()]
    stats = {
        "raw_reviews": str(raw_reviews),
        "train_review_ids": str(train_review_ids),
        "total_raw_reviews": total_reviews,
        "num_train_review_ids": len(train_ids),
        "num_train_reviews_loaded": train_reviews,
        "num_users": len(by_user),
        "num_items": len({row["item_id"] for row in rec_rows}),
        "sequence_length": summarize(seq_lens),
        "users_len_ge_2": sum(1 for value in seq_lens if value >= 2),
        "users_len_ge_5": sum(1 for value in seq_lens if value >= 5),
        "users_len_ge_10": sum(1 for value in seq_lens if value >= 10),
        "field_presence": dict(fields_counter),
        "num_rec_chrono_smoke_examples": len(smoke_examples),
    }
    with (output_dir / "rec_chrono_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-reviews",
        default="/data/cjk/Search_Reconmendation/stage3/raw/amazon_legacy_5core/reviews_Electronics_5.json.gz",
    )
    parser.add_argument(
        "--train-review-ids",
        default="/data/cjk/Search_Reconmendation/external/Amazon-Product-Search-Datasets/reviews_Electronics_5.json.gz/train_review_id.txt.gz",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics",
    )
    parser.add_argument("--max-smoke-examples", type=int, default=10000)
    parser.add_argument("--max-history-len", type=int, default=5)
    args = parser.parse_args()

    stats = build(
        args.raw_reviews,
        args.train_review_ids,
        args.output_dir,
        args.max_smoke_examples,
        args.max_history_len,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
