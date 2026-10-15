#!/usr/bin/env python3
"""Train a practical UniSAR-faithful joint Search/Rec item representation.

The implementation keeps UniSAR's core inductive biases while producing the
static item table expected by GenSAR's indexer. Events at the target timestamp
are excluded from history so the matched Search/Rec pair cannot leak labels.
"""

import argparse
import bisect
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


TOKEN_RE = re.compile(r"[a-z0-9]+")


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_query_tokens(data_dir, min_word_freq=2, max_query_words=16):
    rows = list(read_jsonl(data_dir / "queries.jsonl"))
    tokenized = {int(row["query_id"]): TOKEN_RE.findall(row["query"].lower()) for row in rows}
    counts = Counter(token for tokens in tokenized.values() for token in tokens)
    word_to_id = {"<pad>": 0, "<unk>": 1}
    for word, count in sorted(counts.items()):
        if count >= min_word_freq:
            word_to_id[word] = len(word_to_id)
    max_qid = max(tokenized, default=0)
    query_tokens = torch.zeros(max_qid + 1, max_query_words, dtype=torch.long)
    query_lengths = torch.ones(max_qid + 1, dtype=torch.long)
    for qid, tokens in tokenized.items():
        ids = [word_to_id.get(token, 1) for token in tokens[:max_query_words]] or [1]
        query_tokens[qid, : len(ids)] = torch.tensor(ids)
        query_lengths[qid] = len(ids)
    return word_to_id, query_tokens, query_lengths


def load_sequences(data_dir, max_users=0, seed=2026):
    item_ids = [row["item_id"] for row in read_jsonl(data_dir / "item_ids.jsonl")]
    item_to_id = {item: idx + 1 for idx, item in enumerate(item_ids)}
    rec_by_user = defaultdict(list)
    search_by_user = defaultdict(list)
    for row in read_jsonl(data_dir / "rec_interactions_chrono.jsonl"):
        if row["item_id"] in item_to_id:
            rec_by_user[row["user_id"]].append(
                (int(row["timestamp"]), item_to_id[row["item_id"]], -1, 1)
            )
    for row in read_jsonl(data_dir / "search_interactions.jsonl"):
        if row["item_id"] in item_to_id:
            search_by_user[row["user_id"]].append(
                (int(row["timestamp"]), item_to_id[row["item_id"]], int(row["query_id"]), 0)
            )
    users = sorted(set(rec_by_user) & set(search_by_user))
    if max_users and len(users) > max_users:
        users = sorted(random.Random(seed).sample(users, max_users))
    user_to_id = {user: idx + 1 for idx, user in enumerate(users)}
    sequences = []
    samples = []
    for user in users:
        rec = sorted(rec_by_user[user])
        search = sorted(search_by_user[user])
        mixed = sorted(rec + search, key=lambda x: (x[0], x[3], x[1], x[2]))
        sequence_id = len(sequences)
        sequences.append(
            {
                "mixed": mixed,
                "mixed_ts": [event[0] for event in mixed],
                "search": search,
                "search_ts": [event[0] for event in search],
                "rec": rec,
                "rec_ts": [event[0] for event in rec],
                "user": user_to_id[user],
            }
        )
        # Search retrieval: query and strict-earlier mixed behavior predict item.
        for event in search:
            samples.append((sequence_id, 0, event[0], event[1], event[2]))
        # Recommendation: strict-earlier behavior predicts the next Rec item.
        for event in rec[1:]:
            samples.append((sequence_id, 1, event[0], event[1], -1))
    return item_ids, user_to_id, sequences, samples


def split_samples(samples, sequences):
    by_user_task = defaultdict(list)
    for sample in samples:
        by_user_task[(sample[0], sample[1])].append(sample)
    train, valid = [], []
    for rows in by_user_task.values():
        rows.sort(key=lambda row: row[2])
        if len(rows) >= 2:
            train.extend(rows[:-1])
            valid.append(rows[-1])
        else:
            train.extend(rows)
    return train, valid


class EventDataset(Dataset):
    def __init__(self, samples, sequences, max_history):
        self.samples = samples
        self.sequences = sequences
        self.max_history = max_history

    def __len__(self):
        return len(self.samples)

    def _history(self, sequence, key, timestamp):
        end = bisect.bisect_left(sequence[f"{key}_ts"], timestamp)
        return sequence[key][max(0, end - self.max_history) : end]

    def __getitem__(self, index):
        sequence_id, task, timestamp, target_item, target_query = self.samples[index]
        sequence = self.sequences[sequence_id]
        return {
            "user": sequence["user"],
            "task": task,
            "target_item": target_item,
            "target_query": target_query,
            "mixed": self._history(sequence, "mixed", timestamp),
            "search": self._history(sequence, "search", timestamp),
            "rec": self._history(sequence, "rec", timestamp),
        }


def make_collate(num_items, num_negatives, seed):
    rng = random.Random(seed)

    def pad_events(rows, key):
        length = max(1, max(len(row[key]) for row in rows))
        items = torch.zeros(len(rows), length, dtype=torch.long)
        queries = torch.full((len(rows), length), -1, dtype=torch.long)
        scenarios = torch.zeros(len(rows), length, dtype=torch.long)
        mask = torch.ones(len(rows), length, dtype=torch.bool)
        for batch_idx, row in enumerate(rows):
            for pos, (_ts, item, query, scenario) in enumerate(row[key]):
                items[batch_idx, pos] = item
                queries[batch_idx, pos] = query
                scenarios[batch_idx, pos] = scenario
                mask[batch_idx, pos] = False
            if not row[key]:
                # Keep one learned padding state visible; all-masked attention yields NaNs.
                mask[batch_idx, 0] = False
        return items, queries, scenarios, mask

    def collate(rows):
        target = torch.tensor([row["target_item"] for row in rows], dtype=torch.long)
        negatives = torch.empty(len(rows), num_negatives, dtype=torch.long)
        for batch_idx, positive in enumerate(target.tolist()):
            for neg_idx in range(num_negatives):
                candidate = rng.randint(1, num_items)
                while candidate == positive:
                    candidate = rng.randint(1, num_items)
                negatives[batch_idx, neg_idx] = candidate
        return {
            "user": torch.tensor([row["user"] for row in rows], dtype=torch.long),
            "task": torch.tensor([row["task"] for row in rows], dtype=torch.long),
            "target_item": target,
            "target_query": torch.tensor([row["target_query"] for row in rows], dtype=torch.long),
            "negatives": negatives,
            "mixed": pad_events(rows, "mixed"),
            "search": pad_events(rows, "search"),
            "rec": pad_events(rows, "rec"),
        }

    return collate


class MMoE(nn.Module):
    def __init__(self, dim, num_experts=4):
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)) for _ in range(num_experts)]
        )
        self.gates = nn.ModuleList([nn.Linear(dim, num_experts) for _ in range(2)])

    def forward(self, states, tasks):
        expert_states = torch.stack([expert(states) for expert in self.experts], dim=1)
        all_gates = torch.stack([F.softmax(gate(states), dim=-1) for gate in self.gates], dim=1)
        gates = all_gates[torch.arange(states.size(0), device=states.device), tasks]
        return torch.sum(expert_states * gates.unsqueeze(-1), dim=1)


class UniSARFaithful(nn.Module):
    def __init__(self, num_items, num_users, num_words, query_tokens, query_lengths, dim, heads, layers, dropout):
        super().__init__()
        self.dim = dim
        self.item_embedding = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users + 1, dim, padding_idx=0)
        self.word_embedding = nn.Embedding(num_words, dim, padding_idx=0)
        self.scenario_embedding = nn.Embedding(2, dim)
        self.position_embedding = nn.Embedding(128, dim)
        self.register_buffer("query_tokens", query_tokens)
        self.register_buffer("query_lengths", query_lengths)
        layer = lambda: nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.search_encoder = nn.TransformerEncoder(layer(), layers)
        self.rec_encoder = nn.TransformerEncoder(layer(), layers)
        self.mixed_encoder = nn.TransformerEncoder(layer(), layers)
        self.search_fusion = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.rec_fusion = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.mmoe = MMoE(dim)
        self.norm = nn.LayerNorm(dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
            self.user_embedding.weight[0].zero_()

    def query_repr(self, query_ids):
        safe_ids = query_ids.clamp_min(0)
        tokens = self.query_tokens[safe_ids]
        lengths = self.query_lengths[safe_ids].clamp_min(1).unsqueeze(-1)
        result = self.word_embedding(tokens).sum(dim=-2) / lengths
        return result * (query_ids >= 0).unsqueeze(-1)

    def encode(self, packed, encoder):
        items, queries, scenarios, padding = packed
        length = items.size(1)
        positions = torch.arange(length, device=items.device).unsqueeze(0)
        states = self.item_embedding(items) + self.query_repr(queries)
        states = states + self.scenario_embedding(scenarios) + self.position_embedding(positions)
        states = encoder(states, src_key_padding_mask=padding)
        valid = (~padding).sum(dim=1).clamp_min(1)
        last = states[torch.arange(states.size(0), device=states.device), valid - 1]
        return states, last, padding

    def forward(self, batch, alpha=0.1, beta=0.1, temperature=0.1):
        mixed_states, mixed_last, mixed_padding = self.encode(batch["mixed"], self.mixed_encoder)
        _, search_last, _ = self.encode(batch["search"], self.search_encoder)
        _, rec_last, _ = self.encode(batch["rec"], self.rec_encoder)
        search_fused, _ = self.search_fusion(
            search_last.unsqueeze(1), mixed_states, mixed_states, key_padding_mask=mixed_padding
        )
        rec_fused, _ = self.rec_fusion(
            rec_last.unsqueeze(1), mixed_states, mixed_states, key_padding_mask=mixed_padding
        )
        task_states = torch.stack([search_fused[:, 0], rec_fused[:, 0]], dim=1)
        state = task_states[torch.arange(task_states.size(0), device=task_states.device), batch["task"]]
        state = self.norm(state + self.user_embedding(batch["user"]))
        state = F.normalize(self.mmoe(state, batch["task"]), dim=-1)
        positive = F.normalize(self.item_embedding(batch["target_item"]), dim=-1)
        negative = F.normalize(self.item_embedding(batch["negatives"]), dim=-1)
        pos_score = torch.sum(state * positive, dim=-1, keepdim=True)
        neg_score = torch.einsum("bd,bnd->bn", state, negative)
        retrieval = F.binary_cross_entropy_with_logits(
            torch.cat([pos_score, neg_score], dim=1),
            torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=1),
        )
        valid_search = (batch["task"] == 0) & (batch["target_query"] >= 0)
        relevance = state.new_zeros(())
        if valid_search.sum() > 1:
            query = F.normalize(self.query_repr(batch["target_query"][valid_search]), dim=-1)
            items = positive[valid_search]
            labels = torch.arange(query.size(0), device=query.device)
            relevance = F.cross_entropy(query @ items.T / temperature, labels)
        same = torch.stack([search_last, rec_last], dim=1)
        task_same = same[torch.arange(same.size(0), device=same.device), batch["task"]]
        align_logits = F.normalize(task_same, dim=-1) @ F.normalize(mixed_last, dim=-1).T / temperature
        align_labels = torch.arange(state.size(0), device=state.device)
        alignment = F.cross_entropy(align_logits, align_labels)
        loss = retrieval + alpha * relevance + beta * alignment
        return loss, {"retrieval": retrieval.detach(), "relevance": relevance.detach(), "alignment": alignment.detach()}


def to_device(batch, device):
    out = {}
    for key, value in batch.items():
        if isinstance(value, (tuple, list)):
            out[key] = tuple(part.to(device, non_blocking=True) for part in value)
        else:
            out[key] = value.to(device, non_blocking=True)
    return out


def cap_samples(samples, limit, seed):
    if limit <= 0 or len(samples) <= limit:
        return samples
    return random.Random(seed).sample(samples, limit)


@torch.no_grad()
def evaluate(model, loader, device, alpha, beta, max_batches):
    model.eval()
    totals = defaultdict(float)
    count = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        loss, parts = model(to_device(batch, device), alpha=alpha, beta=beta)
        totals["loss"] += loss.item()
        for key, value in parts.items():
            totals[key] += value.item()
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="stage3/data/gensar_faithful_20k")
    parser.add_argument("--output-dir", default="stage3/artifacts/unisar_faithful_20k")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-history", type=int, default=30)
    parser.add_argument("--num-negatives", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=20000)
    parser.add_argument("--max-valid-batches", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--init-data-dir", default="")
    parser.add_argument("--update-task", choices=["all", "rec", "search"], default="all")
    parser.add_argument("--update-start-timestamp", type=int, default=0)
    parser.add_argument("--freeze-for-rec-update", action="store_true")
    parser.add_argument(
        "--zero-unseen-items",
        action="store_true",
        help="Export exact zero vectors for vocabulary items with no revealed event.",
    )
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    words, query_tokens, query_lengths = build_query_tokens(data_dir)
    item_ids, users, sequences, samples = load_sequences(data_dir, args.max_users, args.seed)
    if args.update_task != "all" or args.update_start_timestamp:
        task_id = {"search": 0, "rec": 1}.get(args.update_task)
        samples = [
            row for row in samples
            if (task_id is None or row[1] == task_id)
            and row[2] >= args.update_start_timestamp
        ]
        shuffled = list(samples)
        random.Random(args.seed).shuffle(shuffled)
        valid_count = min(max(len(shuffled) // 10, 1), args.max_valid_samples)
        valid_samples, train_samples = shuffled[:valid_count], shuffled[valid_count:]
    else:
        train_samples, valid_samples = split_samples(samples, sequences)
    train_samples = cap_samples(train_samples, args.max_train_samples, args.seed)
    valid_samples = cap_samples(valid_samples, args.max_valid_samples, args.seed + 1)
    collate = make_collate(len(item_ids), args.num_negatives, args.seed)
    train_loader = DataLoader(
        EventDataset(train_samples, sequences, args.max_history), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, collate_fn=collate, pin_memory=True,
    )
    valid_loader = DataLoader(
        EventDataset(valid_samples, sequences, args.max_history), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, collate_fn=collate, pin_memory=True,
    )
    device = torch.device(args.device)
    model = UniSARFaithful(
        len(item_ids), len(users), len(words), query_tokens, query_lengths,
        args.dim, args.heads, args.layers, args.dropout,
    ).to(device)
    warmstart = None
    if args.init_checkpoint:
        if not args.init_data_dir:
            raise ValueError("--init-data-dir is required with --init-checkpoint")
        warmstart = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        old_args = warmstart.get("args", {})
        old_max_users = int(old_args.get("max_users", 0))
        old_seed = int(old_args.get("seed", args.seed))
        old_items, old_users, _, _ = load_sequences(
            Path(args.init_data_dir), old_max_users, old_seed
        )
        if old_items != item_ids:
            raise ValueError("Warm-start item vocabulary does not match the update snapshot")
        state = model.state_dict()
        for key, value in warmstart["model"].items():
            if key == "user_embedding.weight":
                continue
            if key in state and state[key].shape == value.shape:
                state[key] = value
        old_user_weight = warmstart["model"]["user_embedding.weight"]
        new_user_weight = state["user_embedding.weight"]
        for user, old_id in old_users.items():
            new_id = users.get(user)
            if new_id is not None:
                new_user_weight[new_id] = old_user_weight[old_id]
        state["user_embedding.weight"] = new_user_weight
        model.load_state_dict(state)
    if args.freeze_for_rec_update:
        if not args.init_checkpoint or args.update_task != "rec":
            raise ValueError("Rec update freezing requires a Rec warm-start run")
        for parameter in model.parameters():
            parameter.requires_grad = False
        train_prefixes = (
            "item_embedding.", "rec_encoder.", "rec_fusion.", "mmoe.experts.",
            "mmoe.gates.1.", "norm.",
        )
        for name, parameter in model.named_parameters():
            if name.startswith(train_prefixes):
                parameter.requires_grad = True
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, _ = model(to_device(batch, device), alpha=args.alpha, beta=args.beta)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            if step % 200 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "train_loss": running / step}), flush=True)
        metrics = evaluate(model, valid_loader, device, args.alpha, args.beta, args.max_valid_batches)
        row = {"epoch": epoch, "train_loss": running / max(len(train_loader), 1), **{f"valid_{k}": v for k, v in metrics.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["loss"] < best:
            best = metrics["loss"]
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "valid": metrics}, output_dir / "best.pt")
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    embeddings = F.normalize(model.item_embedding.weight[1:].detach().float().cpu(), dim=-1)
    seen_items = set()
    if args.zero_unseen_items:
        for filename in ("rec_interactions_chrono.jsonl", "search_interactions.jsonl"):
            seen_items.update(
                row["item_id"] for row in read_jsonl(data_dir / filename)
            )
        unseen_mask = torch.tensor(
            [item_id not in seen_items for item_id in item_ids], dtype=torch.bool
        )
        embeddings[unseen_mask] = 0.0
    torch.save(embeddings, output_dir / "UniSAR_faithful.pt")
    with (output_dir / "item_ids.json").open("w", encoding="utf-8") as handle:
        json.dump(item_ids, handle)
    summary = {
        "items": len(item_ids), "users": len(users), "words": len(words),
        "train_samples": len(train_samples), "valid_samples": len(valid_samples),
        "best_epoch": checkpoint["epoch"], "best_valid": checkpoint["valid"],
        "embedding_shape": list(embeddings.shape), "history": history,
        "seen_items": len(seen_items) if args.zero_unseen_items else len(item_ids),
        "zero_unseen_items": bool(args.zero_unseen_items),
        "warmstart_checkpoint": args.init_checkpoint or None,
        "update_task": args.update_task,
        "update_start_timestamp": args.update_start_timestamp,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
