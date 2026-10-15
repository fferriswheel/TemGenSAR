#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader, Dataset


def make_placeholder_sid(item_id, task):
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


class JsonlGenerationDataset(Dataset):
    def __init__(self, path):
        self.rows = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        target_sid = row.get("target_sid")
        if target_sid is None:
            target_sid = make_placeholder_sid(row["target_item_id"], row["task"])
        return {
            "task": row["task"],
            "input_text": row["input_text"],
            "target_text": " ".join(target_sid),
            "target_item_id": row["target_item_id"],
        }


def collate_generation(batch):
    return {
        "task": [row["task"] for row in batch],
        "input_text": [row["input_text"] for row in batch],
        "target_text": [row["target_text"] for row in batch],
        "target_item_id": [row["target_item_id"] for row in batch],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--examples",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/mixed_smoke_examples.jsonl",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=2)
    args = parser.parse_args()

    dataset = JsonlGenerationDataset(args.examples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_generation,
    )

    print(json.dumps({"num_examples": len(dataset), "batch_size": args.batch_size}, indent=2))
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.num_batches:
            break
        print(json.dumps({"batch_idx": batch_idx, "batch": batch}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
