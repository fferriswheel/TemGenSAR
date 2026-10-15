#!/usr/bin/env python3
"""Train a W0-W2 query-item encoder and export Search item representations."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


TOKEN_RE = re.compile(r"[a-z0-9]+")


def parse_tokens(value) -> list[str]:
    text = str(value).strip()
    if text.startswith("["):
        parsed = ast.literal_eval(text)
        return [f"tok{int(token)}" for token in parsed]
    return TOKEN_RE.findall(text.lower())


def build_query_groups(
    source_dir: Path,
    initialization_end_window: int,
    max_query_tokens: int,
    max_positive_items: int,
):
    frame = pd.read_parquet(
        source_dir / "search_positive_events.parquet",
        filters=[("window_id", "<=", initialization_end_window)],
        columns=["query_index", "keyword", "item_index"],
    )
    token_counts = Counter()
    query_tokens = {}
    for query_index, keyword in frame[["query_index", "keyword"]].drop_duplicates("query_index").itertuples(index=False):
        tokens = parse_tokens(keyword)[:max_query_tokens]
        query_tokens[int(query_index)] = tokens
        token_counts.update(tokens)
    token_vocab = {"<pad>": 0, "<unk>": 1}
    for token in sorted(token_counts):
        token_vocab[token] = len(token_vocab)
    support = (
        frame.groupby(["query_index", "item_index"], sort=False)
        .size()
        .rename("support")
        .reset_index()
        .sort_values(["query_index", "support", "item_index"], ascending=[True, False, True])
    )
    groups = []
    for query_index, rows in support.groupby("query_index", sort=True):
        positives = [int(value) - 1 for value in rows["item_index"].head(max_positive_items)]
        token_ids = [token_vocab.get(token, 1) for token in query_tokens[int(query_index)]] or [1]
        groups.append((int(query_index), token_ids, positives))
    return groups, token_vocab, int(len(frame))


class QueryGroupDataset(Dataset):
    def __init__(self, groups):
        self.groups = groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        return self.groups[index]


def collate_groups(rows):
    max_tokens = max(len(row[1]) for row in rows)
    max_positives = max(len(row[2]) for row in rows)
    tokens = torch.zeros(len(rows), max_tokens, dtype=torch.long)
    token_mask = torch.zeros(len(rows), max_tokens, dtype=torch.bool)
    positives = torch.zeros(len(rows), max_positives, dtype=torch.long)
    positive_mask = torch.zeros(len(rows), max_positives, dtype=torch.bool)
    query_ids = []
    for idx, (query_id, token_row, positive_row) in enumerate(rows):
        tokens[idx, : len(token_row)] = torch.tensor(token_row)
        token_mask[idx, : len(token_row)] = True
        positives[idx, : len(positive_row)] = torch.tensor(positive_row)
        positive_mask[idx, : len(positive_row)] = True
        query_ids.append(query_id)
    return torch.tensor(query_ids), tokens, token_mask, positives, positive_mask


class QueryItemEncoder(nn.Module):
    def __init__(self, common_embedding: torch.Tensor, token_count: int, dim: int):
        super().__init__()
        self.register_buffer("common_embedding", common_embedding.float())
        self.token_embedding = nn.Embedding(token_count, dim, padding_idx=0)
        self.query_encoder = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.common_projection = nn.Linear(common_embedding.shape[1], dim)
        self.item_residual = nn.Embedding(len(common_embedding), dim)
        nn.init.zeros_(self.item_residual.weight)

    def query(self, token_ids, token_mask):
        embedded = self.token_embedding(token_ids)
        pooled = (embedded * token_mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / token_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return F.normalize(self.query_encoder(pooled), dim=1)

    def items(self):
        indices = torch.arange(len(self.common_embedding), device=self.common_embedding.device)
        return F.normalize(
            self.common_projection(self.common_embedding) + self.item_residual(indices), dim=1
        )


def group_loss(logits, positives, positive_mask):
    selected = logits.gather(1, positives).masked_fill(~positive_mask, float("-inf"))
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(selected, dim=1)).mean()


@torch.no_grad()
def evaluate(model, loader, device, temperature):
    model.eval()
    item_vectors = model.items()
    loss_sum = 0.0
    hit_sum = 0
    rows = 0
    for _, tokens, token_mask, positives, positive_mask in loader:
        tokens, token_mask, positives, positive_mask = [
            value.to(device) for value in [tokens, token_mask, positives, positive_mask]
        ]
        logits = model.query(tokens, token_mask) @ item_vectors.t() / temperature
        loss_sum += float(group_loss(logits, positives, positive_mask)) * len(tokens)
        top = logits.topk(10, dim=1).indices
        hit = (top.unsqueeze(2) == positives.unsqueeze(1)) & positive_mask.unsqueeze(1)
        hit_sum += int(hit.any(dim=(1, 2)).sum())
        rows += len(tokens)
    return {"loss": loss_sum / max(rows, 1), "hit_at_10": hit_sum / max(rows, 1)}


def train_epochs(model, groups, args, device, epochs, valid_groups=None):
    train_loader = DataLoader(
        QueryGroupDataset(groups), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_groups, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    valid_loader = (
        DataLoader(
            QueryGroupDataset(valid_groups), batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, collate_fn=collate_groups, pin_memory=True,
        )
        if valid_groups
        else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best = None
    best_score = float("inf") if args.selection_metric == "loss" else float("-inf")
    best_epoch = epochs
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        rows = 0
        for _, tokens, token_mask, positives, positive_mask in train_loader:
            tokens, token_mask, positives, positive_mask = [
                value.to(device, non_blocking=True)
                for value in [tokens, token_mask, positives, positive_mask]
            ]
            logits = model.query(tokens, token_mask) @ model.items().t() / args.temperature
            loss = group_loss(logits, positives, positive_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(tokens)
            rows += len(tokens)
        row = {"epoch": epoch, "train_loss": running / max(rows, 1)}
        if valid_loader is not None:
            metrics = evaluate(model, valid_loader, device, args.temperature)
            row.update({f"valid_{key}": value for key, value in metrics.items()})
            score = metrics[args.selection_metric]
            improved = score < best_score if args.selection_metric == "loss" else score > best_score
            if improved:
                best_score = score
                best_epoch = epoch
                best = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        history.append(row)
        print(json.dumps(row), flush=True)
    return history, best_epoch, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--common-embedding", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initialization-end-window", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--max-query-tokens", type=int, default=32)
    parser.add_argument("--max-positive-items", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument(
        "--selection-metric", choices=["loss", "hit_at_10"], default="loss"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = torch.load(args.common_embedding, map_location="cpu", weights_only=False)
    groups, token_vocab, positive_rows = build_query_groups(
        args.source_dir,
        args.initialization_end_window,
        args.max_query_tokens,
        args.max_positive_items,
    )
    shuffled = list(groups)
    random.Random(args.seed).shuffle(shuffled)
    valid_count = max(1, round(len(shuffled) * args.valid_ratio))
    valid_groups = shuffled[:valid_count]
    train_groups = shuffled[valid_count:]
    device = torch.device(args.device)
    selection_model = QueryItemEncoder(common.to(device), len(token_vocab), args.dim).to(device)
    selection_history, best_epoch, _ = train_epochs(
        selection_model, train_groups, args, device, args.epochs, valid_groups
    )

    torch.manual_seed(args.seed)
    final_model = QueryItemEncoder(common.to(device), len(token_vocab), args.dim).to(device)
    final_history, _, _ = train_epochs(
        final_model, groups, args, device, best_epoch, valid_groups=None
    )
    final_model.eval()
    with torch.no_grad():
        embeddings = final_model.items().cpu()
    torch.save(embeddings, args.output_dir / "search_embedding.pt")
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "token_vocab": token_vocab,
            "params": vars(args),
            "best_epoch": best_epoch,
        },
        args.output_dir / "search_encoder_best.pt",
    )
    summary = {
        "method": "multi-positive query-item contrastive encoder",
        "future_behavior_used": False,
        "query_groups": len(groups),
        "positive_rows": positive_rows,
        "query_token_vocabulary": len(token_vocab),
        "embedding_shape": list(embeddings.shape),
        "selected_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "selection_history": selection_history,
        "final_history": final_history,
        "params": vars(args),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if "history" not in key}, default=str, indent=2))


if __name__ == "__main__":
    main()
