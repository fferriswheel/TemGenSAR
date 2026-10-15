#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


SID_RE = re.compile(r"<[PRS][12]_\d+>")


def read_jsonl(path, limit=0):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class SidDataset(Dataset):
    def __init__(self, path, limit=0):
        self.rows = read_jsonl(path, limit)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "task": row["task"],
            "input_text": row["input_text"],
            "target_text": " ".join(row["target_sid"]),
            "target_sid": row["target_sid"],
            "target_item_id": row["target_item_id"],
        }


def collate_rows(batch):
    return {
        "task": [row["task"] for row in batch],
        "input_text": [row["input_text"] for row in batch],
        "target_text": [row["target_text"] for row in batch],
        "target_sid": [row["target_sid"] for row in batch],
        "target_item_id": [row["target_item_id"] for row in batch],
    }


def make_train_sampler(dataset, balance_by):
    if balance_by == "none":
        return None

    keys = []
    for row in dataset.rows:
        if balance_by == "target_item":
            key = row["target_item_id"]
        elif balance_by == "target_sid":
            key = tuple(row["target_sid"])
        elif balance_by == "target_suffix":
            key = tuple(row["target_sid"][2:])
        else:
            raise ValueError(f"Unknown balance key: {balance_by}")
        keys.append(key)

    counts = Counter(keys)
    weights = torch.tensor([1.0 / counts[key] for key in keys], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def collect_special_tokens(paths, mapping_path):
    tokens = {"<SEARCH>", "<REC>", "<HIST>"}
    for path in paths:
        for row in read_jsonl(path):
            tokens.update(row.get("target_sid", []))

    if mapping_path and Path(mapping_path).exists():
        mapping = json.load(open(mapping_path, "r", encoding="utf-8"))
        for item in mapping.values():
            tokens.update(item["search_sid"])
            tokens.update(item["rec_sid"])
    return sorted(tokens)


def build_sid_index(mapping_path):
    sid_to_items = {"search": defaultdict(set), "rec": defaultdict(set)}
    if not mapping_path or not Path(mapping_path).exists():
        return sid_to_items
    mapping = json.load(open(mapping_path, "r", encoding="utf-8"))
    for item_id, item in mapping.items():
        sid_to_items["search"][tuple(item["search_sid"])].add(item_id)
        sid_to_items["rec"][tuple(item["rec_sid"])].add(item_id)
    return sid_to_items


def build_sid_rows_index(rows):
    sid_to_items = {"search": defaultdict(set), "rec": defaultdict(set)}
    for row in rows:
        sid_to_items[row["task"]][tuple(row["target_sid"])].add(row["target_item_id"])
    return sid_to_items


def merge_sid_indexes(*indexes):
    merged = {"search": defaultdict(set), "rec": defaultdict(set)}
    for index in indexes:
        for task, task_index in index.items():
            for sid, items in task_index.items():
                merged[task][sid].update(items)
    return merged


def build_sid_trie(sid_to_items, tokenizer):
    tries = {"search": {}, "rec": {}}
    for task, task_index in sid_to_items.items():
        root = tries[task]
        for sid in task_index:
            node = root
            for token in sid:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id == tokenizer.unk_token_id:
                    raise ValueError(f"SID token is not in tokenizer vocabulary: {token}")
                node = node.setdefault(token_id, {})
            node[tokenizer.eos_token_id] = {}
    return tries


def build_allowed_token_ids(tokenizer):
    allowed = {
        "p1": [],
        "p2": [],
        "s1": [],
        "s2": [],
        "r1": [],
        "r2": [],
    }
    vocab = tokenizer.get_vocab()
    for token, token_id in vocab.items():
        if re.fullmatch(r"<P1_\d+>", token):
            allowed["p1"].append(token_id)
        elif re.fullmatch(r"<P2_\d+>", token):
            allowed["p2"].append(token_id)
        elif re.fullmatch(r"<S1_\d+>", token):
            allowed["s1"].append(token_id)
        elif re.fullmatch(r"<S2_\d+>", token):
            allowed["s2"].append(token_id)
        elif re.fullmatch(r"<R1_\d+>", token):
            allowed["r1"].append(token_id)
        elif re.fullmatch(r"<R2_\d+>", token):
            allowed["r2"].append(token_id)
    return allowed


def make_prefix_allowed_tokens_fn(tokenizer, tasks, allowed):
    eos = tokenizer.eos_token_id
    decoder_start = tokenizer.pad_token_id

    def prefix_allowed_tokens_fn(batch_id, sentence):
        tokens = sentence.tolist()
        if tokens and tokens[0] == decoder_start:
            tokens = tokens[1:]
        pos = len(tokens)
        task = tasks[batch_id]
        if pos == 0:
            return allowed["p1"]
        if pos == 1:
            return allowed["p2"]
        if pos == 2:
            return allowed["s1"] if task == "search" else allowed["r1"]
        if pos == 3:
            return allowed["s2"] if task == "search" else allowed["r2"]
        return [eos]

    return prefix_allowed_tokens_fn


def make_trie_prefix_allowed_tokens_fn(tokenizer, tasks, tries):
    eos = tokenizer.eos_token_id
    decoder_start = tokenizer.pad_token_id

    def prefix_allowed_tokens_fn(batch_id, sentence):
        tokens = sentence.tolist()
        if tokens and tokens[0] == decoder_start:
            tokens = tokens[1:]
        task = tasks[batch_id]
        node = tries[task]
        for token_id in tokens:
            if token_id == eos:
                return [eos]
            if token_id not in node:
                return [eos]
            node = node[token_id]
        return list(node.keys()) or [eos]

    return prefix_allowed_tokens_fn


def tokenize_batch(tokenizer, batch, device, max_input_len, max_target_len):
    inputs = tokenizer(
        batch["input_text"],
        max_length=max_input_len,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    targets = tokenizer(
        text_target=batch["target_text"],
        max_length=max_target_len,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    labels = targets["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["labels"] = labels.to(device)
    return inputs


def parse_sid(text):
    return SID_RE.findall(text)


def parse_eval_ks(value):
    ks = []
    for part in value.split(","):
        part = part.strip()
        if part:
            ks.append(int(part))
    if not ks:
        raise ValueError("--eval-ks must contain at least one K")
    return sorted(set(ks))


def dedupe_sids(sids):
    seen = set()
    deduped = []
    for sid in sids:
        key = tuple(sid)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sid)
    return deduped


def reciprocal_discount(rank):
    return 1.0 / math.log2(rank + 1)


def get_metric(metrics, metric_name):
    if "." in metric_name:
        task, name = metric_name.split(".", 1)
        if task not in metrics["by_task"]:
            raise KeyError(f"Task metric is unavailable: {metric_name}")
        return metrics["by_task"][task][name]
    return metrics["overall"][metric_name]


@torch.no_grad()
def evaluate(model, tokenizer, dataset, sid_to_items, args, device, output_examples_path=None, sid_trie=None):
    model.eval()
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_rows)
    allowed_token_ids = build_allowed_token_ids(tokenizer) if args.constrained_decode else None
    eval_ks = parse_eval_ks(args.eval_ks)
    max_k = max(eval_ks)
    counts = Counter()
    by_task = defaultdict(Counter)
    examples = []

    for batch in tqdm(loader, desc="eval", leave=False):
        encoded = tokenizer(
            batch["input_text"],
            max_length=args.max_input_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "num_beams": max(args.num_beams, max_k),
            "num_return_sequences": max_k,
        }
        if args.trie_constrained_decode:
            if sid_trie is None:
                raise ValueError("--trie-constrained-decode requires a SID trie")
            generate_kwargs["prefix_allowed_tokens_fn"] = make_trie_prefix_allowed_tokens_fn(
                tokenizer,
                batch["task"],
                sid_trie,
            )
        elif args.constrained_decode:
            generate_kwargs["prefix_allowed_tokens_fn"] = make_prefix_allowed_tokens_fn(
                tokenizer,
                batch["task"],
                allowed_token_ids,
            )
        generated = model.generate(**encoded, **generate_kwargs)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)

        for idx in range(len(batch["task"])):
            task = batch["task"][idx]
            target_sid = batch["target_sid"][idx]
            target_item = batch["target_item_id"][idx]
            start = idx * max_k
            pred_texts = decoded[start : start + max_k]
            pred_sids = dedupe_sids([parse_sid(text)[:4] for text in pred_texts])
            pred_sid4 = pred_sids[0] if pred_sids else []

            valid = len(pred_sid4) == 4
            exact = pred_sid4 == target_sid
            prefix = valid and pred_sid4[:2] == target_sid[:2]
            suffix = valid and pred_sid4[2:] == target_sid[2:]
            hit = valid and target_item in sid_to_items[task].get(tuple(pred_sid4), set())
            sid_rank = None
            item_rank = None
            for rank, pred_sid_k in enumerate(pred_sids, start=1):
                if len(pred_sid_k) != 4:
                    continue
                sid_key = tuple(pred_sid_k)
                if sid_rank is None and pred_sid_k == target_sid:
                    sid_rank = rank
                if item_rank is None and target_item in sid_to_items[task].get(sid_key, set()):
                    item_rank = rank
                if sid_rank is not None and item_rank is not None:
                    break

            for counter in [counts, by_task[task]]:
                counter["total"] += 1
                counter["valid_sid"] += int(valid)
                counter["exact_sid"] += int(exact)
                counter["prefix_exact"] += int(prefix)
                counter["suffix_exact"] += int(suffix)
                counter["item_hit_at_1"] += int(hit)
                for k in eval_ks:
                    counter[f"sid_recall_at_{k}"] += int(sid_rank is not None and sid_rank <= k)
                    counter[f"item_hit_at_{k}"] += int(item_rank is not None and item_rank <= k)
                    counter[f"ndcg_at_{k}"] += reciprocal_discount(item_rank) if item_rank is not None and item_rank <= k else 0.0

            if len(examples) < args.num_saved_predictions:
                examples.append(
                    {
                        "task": task,
                        "input_text": batch["input_text"][idx],
                        "target_sid": target_sid,
                        "target_item_id": target_item,
                        "pred_text": pred_texts[0] if pred_texts else "",
                        "pred_sid": pred_sid4,
                        "top_pred_sids": pred_sids[:max_k],
                        "valid_sid": valid,
                        "exact_sid": exact,
                        "item_hit_at_1": hit,
                        "sid_rank": sid_rank,
                        "item_rank": item_rank,
                    }
                )

    def rates(counter):
        total = max(counter["total"], 1)
        result = {
            "total": counter["total"],
            "valid_sid_rate": counter["valid_sid"] / total,
            "exact_sid_match": counter["exact_sid"] / total,
            "prefix_accuracy": counter["prefix_exact"] / total,
            "suffix_accuracy": counter["suffix_exact"] / total,
            "item_hit_at_1": counter["item_hit_at_1"] / total,
        }
        for k in eval_ks:
            result[f"sid_recall_at_{k}"] = counter[f"sid_recall_at_{k}"] / total
            result[f"item_hit_at_{k}"] = counter[f"item_hit_at_{k}"] / total
            result[f"ndcg_at_{k}"] = counter[f"ndcg_at_{k}"] / total
        return result

    metrics = {"overall": rates(counts), "by_task": {task: rates(counter) for task, counter in by_task.items()}}
    if output_examples_path:
        write_json(output_examples_path, {"metrics": metrics, "examples": examples})
    return metrics


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = SidDataset(args.train_file, args.max_train_examples)
    valid_dataset = SidDataset(args.valid_file, args.max_valid_examples)
    test_dataset = SidDataset(args.test_file, args.max_test_examples)

    special_tokens = collect_special_tokens(
        [args.train_file, args.valid_file, args.test_file],
        args.sid_mapping,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if added:
        model.resize_token_embeddings(len(tokenizer))

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_sampler = make_train_sampler(train_dataset, args.balance_train_by)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_rows,
    )
    mapping_sid_to_items = build_sid_index(args.sid_mapping)
    data_sid_to_items = merge_sid_indexes(
        build_sid_rows_index(train_dataset.rows),
        build_sid_rows_index(valid_dataset.rows),
        build_sid_rows_index(test_dataset.rows),
    )
    if args.trie_source == "mapping":
        sid_to_items = mapping_sid_to_items
    elif args.trie_source == "data":
        sid_to_items = data_sid_to_items
    else:
        sid_to_items = merge_sid_indexes(mapping_sid_to_items, data_sid_to_items)
    sid_trie = build_sid_trie(sid_to_items, tokenizer) if args.trie_constrained_decode else None

    if args.eval_only:
        test_metrics = evaluate(
            model,
            tokenizer,
            test_dataset,
            sid_to_items,
            args,
            device,
            output_dir / "eval_only_predictions.json",
            sid_trie,
        )
        summary = {
            "mode": "eval_only",
            "model_name": args.model_name,
            "special_tokens": len(special_tokens),
            "test_examples": len(test_dataset),
            "test": test_metrics,
            "args": vars(args),
        }
        write_json(output_dir / "eval_only_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    train_log = []
    best_valid = -1.0
    best_dir = output_dir / "best"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(progress, start=1):
            inputs = tokenize_batch(tokenizer, batch, device, args.max_input_len, args.max_target_len)
            outputs = model(**inputs)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            running += loss.item() * args.grad_accum_steps
            steps += 1
            if step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=running / max(steps, 1))

        if steps % args.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        valid_metrics = evaluate(
            model,
            tokenizer,
            valid_dataset,
            sid_to_items,
            args,
            device,
            output_dir / f"valid_predictions_epoch{epoch}.json",
            sid_trie,
        )
        valid_score = get_metric(valid_metrics, args.checkpoint_metric)
        row = {"epoch": epoch, "train_loss": running / max(steps, 1), "valid": valid_metrics}
        train_log.append(row)
        write_json(output_dir / "train_log.json", train_log)

        if valid_score > best_valid:
            best_valid = valid_score
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

    tokenizer = AutoTokenizer.from_pretrained(best_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(best_dir).to(device)
    test_metrics = evaluate(
        model,
        tokenizer,
        test_dataset,
        sid_to_items,
        args,
        device,
        output_dir / "test_predictions.json",
        sid_trie,
    )
    summary = {
        "model_name": args.model_name,
        "special_tokens": len(special_tokens),
        "added_tokens": added,
        "train_examples": len(train_dataset),
        "valid_examples": len(valid_dataset),
        "test_examples": len(test_dataset),
        "checkpoint_metric": args.checkpoint_metric,
        "best_valid_score": best_valid,
        "test": test_metrics,
        "args": vars(args),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-file",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/v1_split/train.jsonl",
    )
    parser.add_argument(
        "--valid-file",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/v1_split/valid.jsonl",
    )
    parser.add_argument(
        "--test-file",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke/v1_split/test.jsonl",
    )
    parser.add_argument(
        "--sid-mapping",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/sid_v1_smoke_p128_32/sid_mapping.v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/cjk/Search_Reconmendation/stage3/runs/t5_v1_smoke",
    )
    parser.add_argument("--model-name", default="t5-small")
    parser.add_argument("--device", default="")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-input-len", type=int, default=96)
    parser.add_argument("--max-target-len", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--constrained-decode", action="store_true")
    parser.add_argument("--trie-constrained-decode", action="store_true")
    parser.add_argument("--trie-source", choices=["mapping", "data", "both"], default="mapping")
    parser.add_argument("--eval-ks", default="1,5,10")
    parser.add_argument("--checkpoint-metric", default="item_hit_at_10")
    parser.add_argument(
        "--balance-train-by",
        choices=["none", "target_item", "target_sid", "target_suffix"],
        default="none",
    )
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-valid-examples", type=int, default=0)
    parser.add_argument("--max-test-examples", type=int, default=0)
    parser.add_argument("--num-saved-predictions", type=int, default=50)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
