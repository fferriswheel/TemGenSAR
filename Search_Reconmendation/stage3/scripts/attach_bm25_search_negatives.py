#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


def load(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump(path, value):
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--num-negatives", type=int, default=999)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()
    item_vocab = load(args.data_dir / "vocab/item_vocab.pkl")
    query_vocab = load(args.data_dir / "vocab/query_vocab.pkl")
    documents = [f"{row['title']} {row['description']}" for row in item_vocab[1:]]
    vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b[a-zA-Z0-9]+\b")
    counts = vectorizer.fit_transform(documents).astype(np.float32)
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    avg_length = max(float(lengths.mean()), 1.0)
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log1p((counts.shape[0] - df + 0.5) / (df + 0.5)).astype(np.float32)
    rows = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
    denom = counts.data + args.k1 * (1.0 - args.b + args.b * lengths[rows] / avg_length)
    weighted = counts.copy()
    weighted.data = counts.data * (args.k1 + 1.0) / denom
    weighted = weighted @ sparse.diags(idf)
    frames = {}
    query_ids = set()
    for split in ("src_val", "src_test"):
        frame = load(args.data_dir / "dataset" / f"{split}.pkl")
        frames[split] = frame
        query_ids.update(int(value) for value in frame["query_id"])
    rankings = {}
    for index, query_id in enumerate(sorted(query_ids), 1):
        query = query_vocab[query_id]["query"]
        query_vec = vectorizer.transform([query])
        scores = (query_vec @ weighted.T).toarray()[0]
        top_n = min(len(scores), args.num_negatives + 100)
        top = np.argpartition(scores, -top_n)[-top_n:]
        top = top[np.argsort(scores[top])[::-1]]
        rankings[query_id] = (top + 1).tolist()
        if index % 100 == 0:
            print(f"ranked {index}/{len(query_ids)} queries", flush=True)
    for split, frame in frames.items():
        negatives = []
        for row in frame.itertuples(index=False):
            positive = int(row.item_id)
            ranked = [item for item in rankings[int(row.query_id)] if item != positive]
            if len(ranked) < args.num_negatives:
                used = set(ranked) | {positive}
                ranked.extend(item for item in range(1, len(item_vocab)) if item not in used)
            negatives.append(ranked[: args.num_negatives])
        frame = frame.copy()
        frame["neg_items"] = negatives
        dump(args.data_dir / "dataset" / f"{split}.pkl", frame)
        print(f"updated {split}: rows={len(frame)}, negatives={args.num_negatives}")


if __name__ == "__main__":
    main()
