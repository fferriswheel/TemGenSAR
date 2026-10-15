#!/usr/bin/env python3
"""Create the embedding-only dataset view expected by GenSAR index training."""

import argparse
import json
from pathlib import Path


def link(source, target):
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--semantic-embedding", type=Path, required=True)
    p.add_argument("--collaborative-embedding", type=Path, required=True)
    p.add_argument("--semantic-name", default="bge-base-en-v1.5.pt")
    p.add_argument("--collaborative-name", default="UniSAR_rec_w1.pt")
    args = p.parse_args()
    link(args.semantic_embedding, args.output_dir / "emb" / args.semantic_name)
    link(args.collaborative_embedding, args.output_dir / "emb" / args.collaborative_name)
    print(json.dumps({
        "semantic": str((args.output_dir / 'emb' / args.semantic_name).resolve()),
        "collaborative": str((args.output_dir / 'emb' / args.collaborative_name).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
