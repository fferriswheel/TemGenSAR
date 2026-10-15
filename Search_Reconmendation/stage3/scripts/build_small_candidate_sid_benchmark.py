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

    for split_rows_ in splits.values():
        rng.shuffle(split_rows_)
    return splits


def limit_per_task(rows, max_examples_per_task, seed):
    if max_examples_per_task <= 0:
        return rows
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    limited = []
    for task_rows in by_task.values():
        rng.shuffle(task_rows)
        limited.extend(task_rows[:max_examples_per_task])
    rng.shuffle(limited)
    return limited


def summarize(rows):
    summary = {
        "num_examples": len(rows),
        "task_counts": dict(Counter(row["task"] for row in rows)),
        "unique_target_items": len({row["target_item_id"] for row in rows}),
    }
    for task in ["search", "rec"]:
        task_rows = [row for row in rows if row["task"] == task]
        summary[task] = {
            "num_examples": len(task_rows),
            "unique_target_items": len({row["target_item_id"] for row in task_rows}),
            "unique_full_sids": len({tuple(row["target_sid"]) for row in task_rows}),
            "top_target_items": Counter(row["target_item_id"] for row in task_rows).most_common(10),
            "top_suffixes": Counter(tuple(row["target_sid"][2:]) for row in task_rows).most_common(10),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/v1_contextual/mixed.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/v1_contextual_small_1000",
    )
    parser.add_argument("--candidate-items", type=int, default=1000)
    parser.add_argument("--max-examples-per-task", type=int, default=0)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    item_counts = Counter(row["target_item_id"] for row in rows)
    candidates = {item for item, _ in item_counts.most_common(args.candidate_items)}
    filtered = [row for row in rows if row["target_item_id"] in candidates]
    filtered = limit_per_task(filtered, args.max_examples_per_task, args.seed)
    splits = split_rows(filtered, args.valid_ratio, args.test_ratio, args.seed)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "mixed.jsonl", filtered)
    with (output_dir / "candidate_items.json").open("w", encoding="utf-8") as f:
        json.dump(sorted(candidates), f, indent=2, ensure_ascii=False)

    for split, split_rows_ in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", split_rows_)
        write_jsonl(output_dir / f"{split}.search.jsonl", [row for row in split_rows_ if row["task"] == "search"])
        write_jsonl(output_dir / f"{split}.rec.jsonl", [row for row in split_rows_ if row["task"] == "rec"])

    stats = {
        "input": args.input,
        "output_dir": str(output_dir),
        "candidate_items": args.candidate_items,
        "max_examples_per_task": args.max_examples_per_task,
        "valid_ratio": args.valid_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "all": summarize(filtered),
        "splits": {split: summarize(split_rows_) for split, split_rows_ in splits.items()},
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
