#!/usr/bin/env python3
"""Learn stable KuaiSAR item representations from anonymized metadata only."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_inputs(data_dir: Path, max_tokens: int, min_token_frequency: int):
    rows = list(read_jsonl(data_dir / "items.jsonl"))
    item_ids = [row["item_id"] for row in rows]
    token_rows = [row.get("description", "").split() for row in rows]
    token_counts = Counter(token for tokens in token_rows for token in tokens)
    token_vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in sorted(token_counts.items()):
        if count >= min_token_frequency:
            token_vocab[token] = len(token_vocab)

    category_vocab = {"<pad>": 0, "<unk>": 1}
    type_vocab = {"<unk>": 0}
    for row in rows:
        for category in row.get("categories", []):
            if category not in category_vocab:
                category_vocab[category] = len(category_vocab)
        item_type = row.get("item_type") or "<unk>"
        if item_type not in type_vocab:
            type_vocab[item_type] = len(type_vocab)

    token_ids = torch.zeros(len(rows), max_tokens, dtype=torch.long)
    token_mask = torch.zeros(len(rows), max_tokens, dtype=torch.bool)
    category_ids = torch.zeros(len(rows), 4, dtype=torch.long)
    category_mask = torch.zeros(len(rows), 4, dtype=torch.bool)
    type_ids = torch.zeros(len(rows), dtype=torch.long)
    first_category = torch.zeros(len(rows), dtype=torch.long)
    valid = torch.zeros(len(rows), dtype=torch.bool)
    for idx, (row, tokens) in enumerate(zip(rows, token_rows)):
        ids = [token_vocab.get(token, 1) for token in tokens[:max_tokens]]
        if ids:
            token_ids[idx, : len(ids)] = torch.tensor(ids)
            token_mask[idx, : len(ids)] = True
        categories = row.get("categories", [])[:4]
        cat_ids = [category_vocab.get(category, 1) for category in categories]
        if cat_ids:
            category_ids[idx, : len(cat_ids)] = torch.tensor(cat_ids)
            category_mask[idx, : len(cat_ids)] = True
            first_category[idx] = cat_ids[0]
        type_ids[idx] = type_vocab.get(row.get("item_type") or "<unk>", 0)
        valid[idx] = bool(ids or cat_ids or type_ids[idx].item())
    return {
        "rows": rows,
        "item_ids": item_ids,
        "token_vocab": token_vocab,
        "category_vocab": category_vocab,
        "type_vocab": type_vocab,
        "token_ids": token_ids,
        "token_mask": token_mask,
        "category_ids": category_ids,
        "category_mask": category_mask,
        "type_ids": type_ids,
        "first_category": first_category,
        "valid": valid,
    }


class StableContentEncoder(nn.Module):
    def __init__(self, token_count: int, category_count: int, type_count: int, dim: int):
        super().__init__()
        self.token_embedding = nn.Embedding(token_count, dim, padding_idx=0)
        self.category_embedding = nn.Embedding(category_count, dim, padding_idx=0)
        self.type_embedding = nn.Embedding(type_count, dim)
        self.encoder = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.projector = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.category_head = nn.Linear(dim, category_count)

    @staticmethod
    def masked_mean(embedding: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (embedding * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)

    def encode(self, token_ids, token_mask, category_ids, category_mask, type_ids):
        token = self.masked_mean(self.token_embedding(token_ids), token_mask)
        category = self.masked_mean(self.category_embedding(category_ids), category_mask)
        item_type = self.type_embedding(type_ids)
        return F.normalize(self.encoder(torch.cat([token, category, item_type], dim=1)), dim=1)

    def forward(self, *inputs):
        representation = self.encode(*inputs)
        return representation, F.normalize(self.projector(representation), dim=1)


def dropout_mask(mask: torch.Tensor, probability: float) -> torch.Tensor:
    dropped = mask & (torch.rand(mask.shape, device=mask.device) >= probability)
    empty = mask.any(dim=1) & ~dropped.any(dim=1)
    if empty.any():
        first = mask.float().argmax(dim=1)
        dropped[empty, first[empty]] = True
    return dropped


def contrastive_loss(left: torch.Tensor, right: torch.Tensor, temperature: float):
    logits = left @ right.t() / temperature
    labels = torch.arange(len(left), device=left.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


@torch.no_grad()
def evaluate(model, loader, device, token_dropout, temperature, category_weight):
    model.eval()
    total = 0.0
    batches = 0
    for batch in loader:
        token_ids, token_mask, category_ids, category_mask, type_ids, first_category = [
            value.to(device) for value in batch
        ]
        left, left_projection = model(
            token_ids,
            dropout_mask(token_mask, token_dropout),
            category_ids,
            category_mask,
            type_ids,
        )
        _, right_projection = model(
            token_ids,
            dropout_mask(token_mask, token_dropout),
            category_ids,
            category_mask,
            type_ids,
        )
        category_valid = first_category > 0
        category_loss = (
            F.cross_entropy(model.category_head(left[category_valid]), first_category[category_valid])
            if category_valid.any()
            else left.new_zeros(())
        )
        total += float(contrastive_loss(left_projection, right_projection, temperature) + category_weight * category_loss)
        batches += 1
    return total / max(batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--min-token-frequency", type=int, default=2)
    parser.add_argument("--token-dropout", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--category-weight", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = build_inputs(args.data_dir, args.max_tokens, args.min_token_frequency)
    valid_indices = data["valid"].nonzero().flatten()
    dataset = TensorDataset(
        data["token_ids"][valid_indices],
        data["token_mask"][valid_indices],
        data["category_ids"][valid_indices],
        data["category_mask"][valid_indices],
        data["type_ids"][valid_indices],
        data["first_category"][valid_indices],
    )
    valid_size = max(1, round(len(dataset) * args.valid_ratio))
    train_dataset, valid_dataset = random_split(
        dataset,
        [len(dataset) - valid_size, valid_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    device = torch.device(args.device)
    model = StableContentEncoder(
        len(data["token_vocab"]), len(data["category_vocab"]), len(data["type_vocab"]), args.dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for batch in train_loader:
            token_ids, token_mask, category_ids, category_mask, type_ids, first_category = [
                value.to(device, non_blocking=True) for value in batch
            ]
            left, left_projection = model(
                token_ids, dropout_mask(token_mask, args.token_dropout),
                category_ids, category_mask, type_ids,
            )
            _, right_projection = model(
                token_ids, dropout_mask(token_mask, args.token_dropout),
                category_ids, category_mask, type_ids,
            )
            category_valid = first_category > 0
            category_loss = (
                F.cross_entropy(model.category_head(left[category_valid]), first_category[category_valid])
                if category_valid.any()
                else left.new_zeros(())
            )
            contrast = contrastive_loss(left_projection, right_projection, args.temperature)
            loss = contrast + args.category_weight * category_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
            steps += 1
        valid_loss = evaluate(
            model, valid_loader, device, args.token_dropout,
            args.temperature, args.category_weight,
        )
        row = {"epoch": epoch, "train_loss": running / max(steps, 1), "valid_loss": valid_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is None:
        raise RuntimeError("common encoder did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    all_dataset = TensorDataset(
        data["token_ids"], data["token_mask"], data["category_ids"],
        data["category_mask"], data["type_ids"],
    )
    embeddings = []
    with torch.no_grad():
        for batch in DataLoader(all_dataset, batch_size=args.batch_size, shuffle=False):
            embeddings.append(model.encode(*[value.to(device) for value in batch]).cpu())
    embedding = torch.cat(embeddings)
    torch.save(embedding, args.output_dir / "common_embedding.pt")
    torch.save(
        {
            "model_state_dict": best_state,
            "params": vars(args),
            "token_vocab": data["token_vocab"],
            "category_vocab": data["category_vocab"],
            "type_vocab": data["type_vocab"],
            "item_ids": data["item_ids"],
        },
        args.output_dir / "common_encoder_best.pt",
    )
    summary = {
        "method": "KuaiSAR stable metadata denoising contrastive encoder",
        "behavior_used": False,
        "items": len(data["item_ids"]),
        "valid_metadata_items": int(data["valid"].sum()),
        "token_vocabulary": len(data["token_vocab"]),
        "category_vocabulary": len(data["category_vocab"]),
        "type_vocabulary": len(data["type_vocab"]),
        "embedding_shape": list(embedding.shape),
        "best_valid_loss": best_loss,
        "history": history,
        "params": vars(args),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, default=str, indent=2))


if __name__ == "__main__":
    main()
