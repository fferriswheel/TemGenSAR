#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


SPECIAL = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
PAD, BOS, EOS, UNK = range(4)


def read_jsonl(path, limit=0):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_vocab(rows):
    tokens = set(SPECIAL)
    for row in rows:
        tokens.update(row["input_text"].split())
        tokens.update(row["target_sid"])
    ordered = SPECIAL + sorted(tokens - set(SPECIAL))
    stoi = {token: idx for idx, token in enumerate(ordered)}
    return ordered, stoi


def encode(tokens, stoi, add_bos=False, add_eos=False):
    ids = []
    if add_bos:
        ids.append(BOS)
    ids.extend(stoi.get(token, UNK) for token in tokens)
    if add_eos:
        ids.append(EOS)
    return ids


class SidDataset(Dataset):
    def __init__(self, path, stoi, limit=0):
        self.rows = read_jsonl(path, limit)
        self.stoi = stoi

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "task": row["task"],
            "src": encode(row["input_text"].split(), self.stoi, add_eos=True),
            "tgt": encode(row["target_sid"], self.stoi, add_bos=True, add_eos=True),
            "target_sid": row["target_sid"],
            "target_item_id": row["target_item_id"],
        }


def pad_2d(seqs, pad_value=PAD):
    max_len = max(len(seq) for seq in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, seq in enumerate(seqs):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def collate(batch):
    return {
        "task": [row["task"] for row in batch],
        "src": pad_2d([row["src"] for row in batch]),
        "tgt": pad_2d([row["tgt"] for row in batch]),
        "target_sid": [row["target_sid"] for row in batch],
        "target_item_id": [row["target_item_id"] for row in batch],
    }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TinySeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt_in):
        src_key_padding_mask = src.eq(PAD)
        tgt_key_padding_mask = tgt_in.eq(PAD)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1), device=tgt_in.device)
        src_emb = self.pos(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos(self.embedding(tgt_in) * math.sqrt(self.d_model))
        hidden = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.out(hidden)

    def encode(self, src):
        src_key_padding_mask = src.eq(PAD)
        src_emb = self.pos(self.embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    def decode_step(self, memory, src_key_padding_mask, ys):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1), device=ys.device)
        tgt_emb = self.pos(self.embedding(ys) * math.sqrt(self.d_model))
        hidden = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.out(hidden[:, -1])


def build_sid_index(rows):
    sid_to_items = defaultdict(set)
    for row in rows:
        sid_to_items[tuple(row["target_sid"])].add(row["target_item_id"])
    return sid_to_items


def build_trie(sids, stoi):
    root = {}
    for sid in sids:
        node = root
        for token in sid:
            node = node.setdefault(stoi[token], {})
        node[EOS] = {}
    return root


def trie_allowed(trie, prefix):
    node = trie
    for token_id in prefix:
        if token_id == BOS:
            continue
        if token_id not in node:
            return [EOS]
        node = node[token_id]
    return list(node.keys()) or [EOS]


def beam_search(model, src, trie, beam_size, max_new_tokens):
    memory, src_mask = model.encode(src)
    beams = [([BOS], 0.0)]
    for _ in range(max_new_tokens):
        candidates = []
        all_done = True
        for seq, score in beams:
            if seq[-1] == EOS:
                candidates.append((seq, score))
                continue
            all_done = False
            allowed = trie_allowed(trie, seq[1:])
            ys = torch.tensor([seq], dtype=torch.long, device=src.device)
            log_probs = torch.log_softmax(model.decode_step(memory, src_mask, ys)[0], dim=-1)
            allowed_scores = log_probs[allowed]
            top_scores, top_pos = torch.topk(allowed_scores, k=min(beam_size, len(allowed)))
            for token_score, pos in zip(top_scores.tolist(), top_pos.tolist()):
                token_id = allowed[pos]
                candidates.append((seq + [token_id], score + token_score))
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        if all_done:
            break
    return beams


def ids_to_sid(ids, itos):
    sid = []
    for token_id in ids:
        if token_id in (BOS, PAD):
            continue
        if token_id == EOS:
            break
        sid.append(itos[token_id])
    return sid


def reciprocal_discount(rank):
    return 1.0 / math.log2(rank + 1)


@torch.no_grad()
def evaluate(model, dataset, sid_to_items, trie, itos, args, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    ks = sorted(set(int(x) for x in args.eval_ks.split(",") if x))
    counts = Counter(total=0)
    task_counts = defaultdict(Counter)
    examples = []
    for batch in tqdm(loader, desc="eval", leave=False):
        src = batch["src"].to(device)
        beams = beam_search(model, src, trie, args.num_beams, args.max_new_tokens)
        pred_sids = []
        seen = set()
        for seq, score in beams:
            sid = ids_to_sid(seq, itos)
            key = tuple(sid)
            if key and key not in seen:
                seen.add(key)
                pred_sids.append((sid, score))
        target_sid = batch["target_sid"][0]
        target_item = batch["target_item_id"][0]
        task = batch["task"][0]
        counts["total"] += 1
        task_counts[task]["total"] += 1
        for k in ks:
            sid_hit_rank = None
            item_hit_rank = None
            for rank, (sid, _) in enumerate(pred_sids[:k], start=1):
                if sid_hit_rank is None and sid == target_sid:
                    sid_hit_rank = rank
                if item_hit_rank is None and target_item in sid_to_items.get(tuple(sid), set()):
                    item_hit_rank = rank
            if sid_hit_rank is not None:
                counts[f"sid_recall_at_{k}"] += 1
                task_counts[task][f"sid_recall_at_{k}"] += 1
            if item_hit_rank is not None:
                counts[f"item_hit_at_{k}"] += 1
                counts[f"ndcg_at_{k}"] += reciprocal_discount(item_hit_rank)
                task_counts[task][f"item_hit_at_{k}"] += 1
                task_counts[task][f"ndcg_at_{k}"] += reciprocal_discount(item_hit_rank)
        if len(examples) < args.num_saved_predictions:
            examples.append(
                {
                    "target_sid": target_sid,
                    "target_item_id": target_item,
                    "predictions": [{"sid": sid, "score": score} for sid, score in pred_sids[: max(ks)]],
                }
            )
    total = counts["total"]
    metrics = {"total": total}
    for k in ks:
        metrics[f"sid_recall_at_{k}"] = counts[f"sid_recall_at_{k}"] / total if total else 0.0
        metrics[f"item_hit_at_{k}"] = counts[f"item_hit_at_{k}"] / total if total else 0.0
        metrics[f"ndcg_at_{k}"] = counts[f"ndcg_at_{k}"] / total if total else 0.0
    metrics["by_task"] = {}
    for task, task_counter in sorted(task_counts.items()):
        task_total = task_counter["total"]
        task_metrics = {"total": task_total}
        for k in ks:
            task_metrics[f"sid_recall_at_{k}"] = (
                task_counter[f"sid_recall_at_{k}"] / task_total if task_total else 0.0
            )
            task_metrics[f"item_hit_at_{k}"] = (
                task_counter[f"item_hit_at_{k}"] / task_total if task_total else 0.0
            )
            task_metrics[f"ndcg_at_{k}"] = task_counter[f"ndcg_at_{k}"] / task_total if task_total else 0.0
        metrics["by_task"][task] = task_metrics
    return metrics, examples


def metric_value(metrics, name):
    if name not in metrics:
        raise KeyError(f"Metric not found: {name}")
    return metrics[name]


def save_checkpoint(path, model, args, itos, stoi):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    write_json(path / "vocab.json", {"itos": itos, "stoi": stoi})
    write_json(path / "model_args.json", vars(args))


def load_checkpoint(path, args, device):
    vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
    model_args = json.loads((path / "model_args.json").read_text(encoding="utf-8"))
    model = TinySeq2Seq(
        len(vocab["itos"]),
        model_args["d_model"],
        model_args["nhead"],
        model_args["num_layers"],
        model_args["dim_feedforward"],
        model_args["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(path / "model.pt", map_location=device))
    return model, vocab["itos"], vocab["stoi"]


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.train_file, args.max_train_examples)
    valid_rows = read_jsonl(args.valid_file, args.max_valid_examples)
    test_rows = read_jsonl(args.test_file, args.max_test_examples)
    itos, stoi = build_vocab(train_rows + valid_rows + test_rows)

    train_data = SidDataset(args.train_file, stoi, args.max_train_examples)
    valid_data = SidDataset(args.valid_file, stoi, args.max_valid_examples)
    test_data = SidDataset(args.test_file, stoi, args.max_test_examples)
    sid_to_items = build_sid_index(train_rows + valid_rows + test_rows)
    trie = build_trie(sid_to_items.keys(), stoi)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TinySeq2Seq(
        len(itos),
        args.d_model,
        args.nhead,
        args.num_layers,
        args.dim_feedforward,
        args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate)

    best_score = -1.0
    best_epoch = 0
    log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for batch in tqdm(loader, desc=f"epoch {epoch}"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            logits = model(src, tgt[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            running += loss.item()
            steps += 1
        valid_metrics, valid_examples = evaluate(model, valid_data, sid_to_items, trie, itos, args, device)
        score = metric_value(valid_metrics, args.checkpoint_metric)
        log_row = {"epoch": epoch, "train_loss": running / max(steps, 1), "valid": valid_metrics}
        log.append(log_row)
        write_json(output_dir / "train_log.json", log)
        write_json(output_dir / f"valid_predictions_epoch{epoch}.json", {"metrics": valid_metrics, "examples": valid_examples})
        save_checkpoint(output_dir / "last", model, args, itos, stoi)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(output_dir / "best", model, args, itos, stoi)

    model, itos, stoi = load_checkpoint(output_dir / "best", args, device)
    sid_to_items = build_sid_index(train_rows + valid_rows + test_rows)
    trie = build_trie(sid_to_items.keys(), stoi)
    test_metrics, test_examples = evaluate(model, test_data, sid_to_items, trie, itos, args, device)
    summary = {
        "vocab_size": len(itos),
        "train_examples": len(train_data),
        "valid_examples": len(valid_data),
        "test_examples": len(test_data),
        "checkpoint_metric": args.checkpoint_metric,
        "best_valid_score": best_score,
        "best_epoch": best_epoch,
        "test": test_metrics,
        "args": vars(args),
    }
    write_json(output_dir / "test_predictions.json", {"metrics": test_metrics, "examples": test_examples})
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def eval_only(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, itos, stoi = load_checkpoint(Path(args.model_path), args, device)
    test_rows = read_jsonl(args.test_file, args.max_test_examples)
    train_rows = read_jsonl(args.train_file, args.max_train_examples) if args.train_file else []
    valid_rows = read_jsonl(args.valid_file, args.max_valid_examples) if args.valid_file else []
    test_data = SidDataset(args.test_file, stoi, args.max_test_examples)
    sid_to_items = build_sid_index(train_rows + valid_rows + test_rows)
    trie = build_trie(sid_to_items.keys(), stoi)
    metrics, examples = evaluate(model, test_data, sid_to_items, trie, itos, args, device)
    summary = {"mode": "eval_only", "test_examples": len(test_data), "test": metrics, "args": vars(args)}
    write_json(output_dir / "eval_only_predictions.json", {"metrics": metrics, "examples": examples})
    write_json(output_dir / "eval_only_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--valid-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--eval-ks", default="1,5,10")
    parser.add_argument("--checkpoint-metric", default="item_hit_at_10")
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-valid-examples", type=int, default=0)
    parser.add_argument("--max-test-examples", type=int, default=0)
    parser.add_argument("--num-saved-predictions", type=int, default=50)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if args.eval_only:
        eval_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
