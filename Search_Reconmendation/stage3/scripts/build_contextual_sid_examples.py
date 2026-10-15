#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter, defaultdict
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


def load_mapping(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def rec_sid_text(item_id, sid_mapping):
    return " ".join(sid_mapping[item_id]["rec_sid"])


def build_rec_index(rec_path, sid_mapping):
    by_user = defaultdict(list)
    for row in read_jsonl(rec_path):
        item_id = row["item_id"]
        if item_id not in sid_mapping:
            continue
        by_user[row["user_id"]].append(
            {
                "item_id": item_id,
                "timestamp": row.get("timestamp", 0),
                "position": row.get("position", 0),
            }
        )

    for rows in by_user.values():
        rows.sort(key=lambda x: (x["timestamp"], x["position"], x["item_id"]))
    return by_user


def build_item_positions(by_user):
    positions = {}
    for user_id, rows in by_user.items():
        item_to_pos = defaultdict(list)
        for idx, row in enumerate(rows):
            item_to_pos[row["item_id"]].append(idx)
        positions[user_id] = item_to_pos
    return positions


def history_before_target(user_rows, item_positions, target_item, max_history_len):
    positions = item_positions.get(target_item, [])
    if positions:
        pos = positions[0]
    else:
        pos = len(user_rows)
    history = [row["item_id"] for row in user_rows[max(0, pos - max_history_len) : pos]]
    return history


def format_history(history_items, sid_mapping):
    return " ".join(rec_sid_text(item_id, sid_mapping) for item_id in history_items)


def build_search_examples(
    search_path,
    by_user,
    item_positions_by_user,
    sid_mapping,
    max_examples,
    max_history_len,
    min_history_len,
    sid_source,
):
    examples = []
    skipped = Counter()
    for row in read_jsonl(search_path):
        if row["split"] != "train":
            skipped["non_train"] += 1
            continue
        user_id = row["user_id"]
        target_item = row["item_id"]
        if target_item not in sid_mapping:
            skipped["missing_target_sid"] += 1
            continue
        user_rows = by_user.get(user_id, [])
        if not user_rows:
            skipped["missing_user_rec_history"] += 1
            continue
        history = history_before_target(
            user_rows,
            item_positions_by_user.get(user_id, {}),
            target_item,
            max_history_len,
        )
        history = [item for item in history if item in sid_mapping and item != target_item]
        if len(history) < min_history_len:
            skipped["short_history"] += 1
            continue

        hist_text = format_history(history, sid_mapping)
        examples.append(
            {
                "task": "search",
                "source": {
                    "user_id": user_id,
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "history_item_ids": history,
                    "history_encoding": "rec_sid",
                },
                "input_text": f"<SEARCH> {row['query']} <HIST> {hist_text}",
                "target_item_id": target_item,
                "target_sid": sid_mapping[target_item]["search_sid"],
                "split": "train",
                "sid_source": sid_source,
            }
        )
        if len(examples) >= max_examples:
            break
    return examples, skipped


def build_rec_examples(by_user, sid_mapping, max_examples, max_history_len, min_history_len, sid_source):
    examples = []
    skipped = Counter()
    for user_id in sorted(by_user):
        rows = by_user[user_id]
        for pos in range(1, len(rows)):
            target = rows[pos]["item_id"]
            if target not in sid_mapping:
                skipped["missing_target_sid"] += 1
                continue
            history = [row["item_id"] for row in rows[max(0, pos - max_history_len) : pos]]
            history = [item for item in history if item in sid_mapping and item != target]
            if len(history) < min_history_len:
                skipped["short_history"] += 1
                continue
            hist_text = format_history(history, sid_mapping)
            examples.append(
                {
                    "task": "rec",
                    "source": {
                        "user_id": user_id,
                        "history_item_ids": history,
                        "history_encoding": "rec_sid",
                    },
                    "input_text": f"<REC> {hist_text}",
                    "target_item_id": target,
                    "target_sid": sid_mapping[target]["rec_sid"],
                    "split": "train",
                    "sid_source": sid_source,
                }
            )
            if len(examples) >= max_examples:
                return examples, skipped
    return examples, skipped


def split_rows(rows, valid_ratio, test_ratio, seed):
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)

    splits = {"train": [], "valid": [], "test": []}
    for task, task_rows in sorted(by_task.items()):
        rng.shuffle(task_rows)
        n = len(task_rows)
        n_test = int(round(n * test_ratio))
        n_valid = int(round(n * valid_ratio))
        splits["test"].extend(task_rows[:n_test])
        splits["valid"].extend(task_rows[n_test : n_test + n_valid])
        splits["train"].extend(task_rows[n_test + n_valid :])

    for rows_ in splits.values():
        rng.shuffle(rows_)
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics")
    parser.add_argument(
        "--sid-mapping",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/sid_v1_smoke_p128_32/sid_mapping.v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/v1_contextual",
    )
    parser.add_argument("--max-search-examples", type=int, default=10000)
    parser.add_argument("--max-rec-examples", type=int, default=10000)
    parser.add_argument("--max-history-len", type=int, default=5)
    parser.add_argument("--min-history-len", type=int, default=1)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--sid-source", default="v1_contextual")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    sid_mapping = load_mapping(args.sid_mapping)
    by_user = build_rec_index(data_dir / "rec_interactions_chrono.jsonl", sid_mapping)
    item_positions_by_user = build_item_positions(by_user)

    search_examples, search_skipped = build_search_examples(
        data_dir / "search_interactions.jsonl",
        by_user,
        item_positions_by_user,
        sid_mapping,
        args.max_search_examples,
        args.max_history_len,
        args.min_history_len,
        args.sid_source,
    )
    rec_examples, rec_skipped = build_rec_examples(
        by_user,
        sid_mapping,
        args.max_rec_examples,
        args.max_history_len,
        args.min_history_len,
        args.sid_source,
    )
    all_examples = search_examples + rec_examples
    splits = split_rows(all_examples, args.valid_ratio, args.test_ratio, args.seed)

    write_jsonl(output_dir / "search.jsonl", search_examples)
    write_jsonl(output_dir / "rec.jsonl", rec_examples)
    write_jsonl(output_dir / "mixed.jsonl", all_examples)
    for split, rows in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
        write_jsonl(output_dir / f"{split}.search.jsonl", [row for row in rows if row["task"] == "search"])
        write_jsonl(output_dir / f"{split}.rec.jsonl", [row for row in rows if row["task"] == "rec"])

    stats = {
        "sid_mapping": args.sid_mapping,
        "max_search_examples": args.max_search_examples,
        "max_rec_examples": args.max_rec_examples,
        "max_history_len": args.max_history_len,
        "min_history_len": args.min_history_len,
        "sid_source": args.sid_source,
        "num_search_examples": len(search_examples),
        "num_rec_examples": len(rec_examples),
        "search_skipped": dict(search_skipped),
        "rec_skipped": dict(rec_skipped),
        "splits": {
            split: {
                "num_examples": len(rows),
                "task_counts": dict(Counter(row["task"] for row in rows)),
            }
            for split, rows in splits.items()
        },
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
