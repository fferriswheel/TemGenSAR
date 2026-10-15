#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from build_neural_rq_sid_v2 import load_item_rows, read_jsonl, write_json


class SkipGramNS(nn.Module):
    def __init__(self, num_items, dim):
        super().__init__()
        self.source = nn.Embedding(num_items, dim)
        self.context = nn.Embedding(num_items, dim)
        nn.init.normal_(self.source.weight, std=0.02)
        nn.init.zeros_(self.context.weight)

    def forward(self, source, positive, negative):
        source_x = self.source(source)
        positive_x = self.context(positive)
        negative_x = self.context(negative)
        positive_score = (source_x * positive_x).sum(dim=-1)
        negative_score = torch.einsum("bd,bnd->bn", source_x, negative_x)
        return -F.logsigmoid(positive_score).mean() - F.logsigmoid(-negative_score).mean()


def build_directed_pairs(data_dir, item_to_idx, max_gap):
    pairs = []
    cur_user = None
    cur_rows = []

    def flush(rows):
        rows.sort(key=lambda row: (row.get("timestamp", 0), row.get("position", 0), row["item_id"]))
        items = [item_to_idx[row["item_id"]] for row in rows if row["item_id"] in item_to_idx]
        for pos, source in enumerate(items):
            for gap in range(1, max_gap + 1):
                if pos + gap < len(items):
                    pairs.append((source, items[pos + gap]))

    for row in read_jsonl(Path(data_dir) / "rec_interactions_chrono.jsonl"):
        user = row["user_id"]
        if cur_user is None:
            cur_user = user
        if user != cur_user:
            flush(cur_rows)
            cur_rows = []
            cur_user = user
        cur_rows.append(row)
    if cur_rows:
        flush(cur_rows)
    return np.asarray(pairs, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/cjk/Search_Reconmendation/stage3/data/gensar_lite_5k")
    parser.add_argument("--output", required=True)
    parser.add_argument("--item-ids", default="")
    parser.add_argument("--max-items", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--max-gap", type=int, default=1)
    parser.add_argument("--negative-samples", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    item_ids, _, _ = load_item_rows(Path(args.data_dir), args.item_ids, args.max_items)
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    pairs = build_directed_pairs(args.data_dir, item_to_idx, args.max_gap)
    if not len(pairs):
        raise RuntimeError("no directed rec pairs were constructed")

    frequencies = np.bincount(pairs[:, 1], minlength=len(item_ids)).astype(np.float64)
    probabilities = np.power(frequencies + 1.0, 0.75)
    probabilities /= probabilities.sum()
    negative_distribution = torch.from_numpy(probabilities.astype(np.float32)).to(device)

    dataset = TensorDataset(torch.from_numpy(pairs[:, 0]), torch.from_numpy(pairs[:, 1]))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    model = SkipGramNS(len(item_ids), args.dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for source, positive in loader:
            source = source.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = torch.multinomial(
                negative_distribution,
                source.shape[0] * args.negative_samples,
                replacement=True,
            ).reshape(source.shape[0], args.negative_samples)
            loss = model(source, positive, negative)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            steps += 1
        row = {"epoch": epoch, "loss": total / max(steps, 1)}
        logs.append(row)
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            print(json.dumps(row))

    model.eval()
    with torch.no_grad():
        embedding = F.normalize(model.source.weight + model.context.weight, p=2, dim=1).cpu()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "item_ids": item_ids,
            "embedding": embedding,
            "source_embedding": model.source.weight.detach().cpu(),
            "context_embedding": model.context.weight.detach().cpu(),
            "params": vars(args),
        },
        output,
    )
    source_counts = Counter(pairs[:, 0].tolist())
    target_counts = Counter(pairs[:, 1].tolist())
    stats = {
        "output": str(output),
        "num_items": len(item_ids),
        "num_pairs": int(len(pairs)),
        "items_with_source_pairs": len(source_counts),
        "items_with_target_pairs": len(target_counts),
        "source_pair_count_min": min(source_counts.values()),
        "source_pair_count_max": max(source_counts.values()),
        "final_loss": logs[-1]["loss"],
        "params": vars(args),
    }
    write_json(output.with_suffix(".stats.json"), stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
