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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/mixed_smoke_examples.sid_v1.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/v1_split",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    splits = split_rows(rows, args.valid_ratio, args.test_ratio, args.seed)

    output_dir = Path(args.output_dir)
    stats = {
        "input": args.input,
        "output_dir": str(output_dir),
        "valid_ratio": args.valid_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "splits": {},
    }
    for split, split_rows_ in splits.items():
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, split_rows_)
        stats["splits"][split] = {
            "path": str(path),
            "num_examples": len(split_rows_),
            "task_counts": dict(Counter(row["task"] for row in split_rows_)),
        }

    with (output_dir / "split_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
