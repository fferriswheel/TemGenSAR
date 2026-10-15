#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_placeholder_sid(item_id, task):
    # Temporary deterministic placeholder. This will be replaced by learned modular SID.
    value = 0
    for ch in item_id:
        value = (value * 131 + ord(ch)) % (256**4)
    tokens = [
        value % 256,
        (value // 256) % 256,
        (value // (256**2)) % 256,
        (value // (256**3)) % 256,
    ]
    prefix = [f"<P{idx}_{tok}>" for idx, tok in enumerate(tokens[:2], start=1)]
    if task == "search":
        suffix = [f"<S{idx}_{tok}>" for idx, tok in enumerate(tokens[2:], start=1)]
    elif task == "rec":
        suffix = [f"<R{idx}_{tok}>" for idx, tok in enumerate(tokens[2:], start=1)]
    else:
        raise ValueError(f"Unknown task: {task}")
    return prefix + suffix


def build_examples(data_dir, output_dir, max_search_examples, max_rec_examples, max_history_len):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    interactions_path = data_dir / "search_interactions.jsonl"

    search_examples = []
    user_train_items = defaultdict(list)

    for row in read_jsonl(interactions_path):
        if row["split"] != "train":
            continue

        item_id = row["item_id"]
        user_id = row["user_id"]
        user_train_items[user_id].append(item_id)

        if len(search_examples) < max_search_examples:
            search_examples.append(
                {
                    "task": "search",
                    "source": {
                        "user_id": user_id,
                        "query_id": row["query_id"],
                        "query": row["query"],
                    },
                    "input_text": f"<SEARCH> {row['query']}",
                    "target_item_id": item_id,
                    "target_sid": make_placeholder_sid(item_id, "search"),
                    "split": "train",
                }
            )

    rec_examples = []
    for user_id, items in user_train_items.items():
        if len(items) < 2:
            continue
        for pos in range(1, len(items)):
            history = items[max(0, pos - max_history_len) : pos]
            target = items[pos]
            rec_examples.append(
                {
                    "task": "rec",
                    "source": {
                        "user_id": user_id,
                        "history_item_ids": history,
                        "history_order": "qrels_file_order_not_chronological",
                    },
                    "input_text": "<REC> " + " ".join(history),
                    "target_item_id": target,
                    "target_sid": make_placeholder_sid(target, "rec"),
                    "split": "train",
                }
            )
            if len(rec_examples) >= max_rec_examples:
                break
        if len(rec_examples) >= max_rec_examples:
            break

    write_jsonl(output_dir / "search_smoke_examples.jsonl", search_examples)
    write_jsonl(output_dir / "rec_smoke_examples.jsonl", rec_examples)
    write_jsonl(output_dir / "mixed_smoke_examples.jsonl", search_examples + rec_examples)

    stats = {
        "max_search_examples": max_search_examples,
        "max_rec_examples": max_rec_examples,
        "max_history_len": max_history_len,
        "num_search_examples": len(search_examples),
        "num_rec_examples": len(rec_examples),
        "num_users_with_train_interactions": len(user_train_items),
        "warning": "Rec examples use qrels file order, not chronological order. Use only for smoke tests.",
    }
    with (output_dir / "smoke_stats.json").open("w", encoding="utf-8") as f:
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
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke",
    )
    parser.add_argument("--max-search-examples", type=int, default=10000)
    parser.add_argument("--max-rec-examples", type=int, default=10000)
    parser.add_argument("--max-history-len", type=int, default=5)
    args = parser.parse_args()

    stats = build_examples(
        args.data_dir,
        args.output_dir,
        args.max_search_examples,
        args.max_rec_examples,
        args.max_history_len,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
