#!/usr/bin/env python3
"""Enrich UniSAR item vectors with train-only Rec context states."""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_unisar_faithful_embeddings import (
    EventDataset,
    UniSARFaithful,
    build_query_tokens,
    cap_samples,
    load_sequences,
    make_collate,
    split_samples,
    to_device,
)


@torch.no_grad()
def rec_context_state(model, batch):
    mixed_states, _, mixed_padding = model.encode(batch["mixed"], model.mixed_encoder)
    _, rec_last, _ = model.encode(batch["rec"], model.rec_encoder)
    rec_fused, _ = model.rec_fusion(
        rec_last.unsqueeze(1),
        mixed_states,
        mixed_states,
        key_padding_mask=mixed_padding,
    )
    state = model.norm(rec_fused[:, 0] + model.user_embedding(batch["user"]))
    return F.normalize(model.mmoe(state, batch["task"]), dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-context-samples", type=int, default=1000000)
    parser.add_argument("--context-weight", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not 0.0 <= args.context_weight <= 1.0:
        raise ValueError("--context-weight must be in [0, 1]")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    _, query_tokens, query_lengths = build_query_tokens(data_dir)
    item_ids, users, sequences, samples = load_sequences(
        data_dir,
        int(train_args.get("max_users", 0)),
        int(train_args.get("seed", args.seed)),
    )
    train_samples, _ = split_samples(samples, sequences)
    rec_samples = [sample for sample in train_samples if sample[1] == 1]
    rec_samples = cap_samples(rec_samples, args.max_context_samples, args.seed + 31)

    device = torch.device(args.device)
    model = UniSARFaithful(
        len(item_ids),
        len(users),
        len(checkpoint["model"]["word_embedding.weight"]),
        query_tokens,
        query_lengths,
        int(train_args["dim"]),
        int(train_args["heads"]),
        int(train_args["layers"]),
        float(train_args["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    loader = DataLoader(
        EventDataset(rec_samples, sequences, int(train_args["max_history"])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=make_collate(len(item_ids), 1, args.seed + 47),
        pin_memory=True,
    )
    sums = torch.zeros(len(item_ids) + 1, int(train_args["dim"]), device=device)
    counts = torch.zeros(len(item_ids) + 1, device=device)
    for step, batch in enumerate(loader, 1):
        batch = to_device(batch, device)
        states = rec_context_state(model, batch)
        targets = batch["target_item"]
        sums.index_add_(0, targets, states)
        counts.index_add_(0, targets, torch.ones_like(targets, dtype=counts.dtype))
        if step % 250 == 0:
            print(json.dumps({"step": step, "batches": len(loader)}), flush=True)

    base = F.normalize(model.item_embedding.weight[1:].detach(), dim=-1)
    context = sums[1:] / counts[1:].clamp_min(1).unsqueeze(1)
    seen = counts[1:] > 0
    context[~seen] = base[~seen]
    context = F.normalize(context, dim=-1)
    enriched = F.normalize(
        (1.0 - args.context_weight) * base + args.context_weight * context,
        dim=-1,
    ).cpu()
    torch.save(enriched, output_dir / "UniSAR_contextual_rec.pt")
    with (output_dir / "item_ids.json").open("w", encoding="utf-8") as handle:
        json.dump(item_ids, handle)
    cosine = F.cosine_similarity(base, context, dim=-1)
    summary = {
        "method": "UniSAR static item + aggregated strict-earlier Rec context",
        "future_behavior_used": False,
        "items": len(item_ids),
        "users": len(users),
        "rec_context_samples": len(rec_samples),
        "context_covered_items": int(seen.sum()),
        "context_coverage": float(seen.float().mean()),
        "context_weight": args.context_weight,
        "base_context_cosine_mean": float(cosine.mean()),
        "base_context_cosine_p10": float(torch.quantile(cosine, 0.1)),
        "embedding_shape": list(enriched.shape),
        "source_checkpoint": args.checkpoint,
        "params": vars(args),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
