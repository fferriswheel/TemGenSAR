#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-vocab", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    with Path(args.item_vocab).open("rb") as handle:
        vocab = pickle.load(handle)
    texts = [f"{row['title']} {row['description']}".strip() for row in vocab[1:]]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device).eval()
    outputs = []
    for batch in DataLoader(texts, batch_size=args.batch_size, shuffle=False):
        encoded = tokenizer(
            list(batch), padding=True, truncation=True, max_length=args.max_length, return_tensors="pt"
        ).to(args.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            # The BGE sentence-transformers configuration uses CLS pooling.
            embedding = model(**encoded).last_hidden_state[:, 0]
        outputs.append(F.normalize(embedding.float(), dim=-1).cpu())
    result = torch.cat(outputs)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print({"shape": list(result.shape), "model": args.model, "output": args.output})


if __name__ == "__main__":
    main()
