#!/usr/bin/env python
"""Materialize a lightweight GenSAR dataset view for one temporal window."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path


def link(source: Path, target: Path) -> None:
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source)


def frame_size(path: Path) -> int:
    with path.open("rb") as handle:
        return len(pickle.load(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarterly-dir", type=Path, required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--semantic-sid", type=Path, required=True)
    parser.add_argument("--collaborate-sid", type=Path, required=True)
    parser.add_argument("--doc2query", type=Path, required=True)
    parser.add_argument(
        "--eval-dir",
        type=Path,
        help="Optional directory containing rec_eval.pkl and src_eval.pkl.",
    )
    args = parser.parse_args()

    source = args.quarterly_dir.resolve()
    window = source / "windows" / args.window
    eval_dir = args.eval_dir.resolve() if args.eval_dir else window
    output = args.output_dir.resolve()
    if not window.is_dir():
        raise FileNotFoundError(window)

    dataset_links = {
        "rec_train.pkl": source / "dataset" / "rec_train.pkl",
        "rec_val.pkl": source / "dataset" / "rec_val.pkl",
        "src_train.pkl": source / "dataset" / "src_train.pkl",
        "src_val.pkl": source / "dataset" / "src_val.pkl",
        "rec_test.pkl": eval_dir / "rec_eval.pkl",
        "src_test.pkl": eval_dir / "src_eval.pkl",
    }
    for name, path in dataset_links.items():
        link(path, output / "dataset" / name)

    for path in sorted((source / "vocab").glob("*.pkl")):
        link(path, output / "vocab" / path.name)
    link(args.doc2query, output / "vocab" / args.doc2query.name)
    link(args.semantic_sid, output / "emb" / args.semantic_sid.name)
    link(args.collaborate_sid, output / "emb" / args.collaborate_sid.name)

    manifest = {
        "quarterly_dir": str(source),
        "window": args.window,
        "eval_dir": str(eval_dir),
        "semantic_sid": str(args.semantic_sid.resolve()),
        "collaborate_sid": str(args.collaborate_sid.resolve()),
        "doc2query": str(args.doc2query.resolve()),
        "splits": {name: frame_size(path) for name, path in dataset_links.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "temporal_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
