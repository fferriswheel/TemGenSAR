#!/usr/bin/env python3
"""Create a W3-update/W4-eval GenSAR dataset for a utility-gated Rec SID."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--semantic-sid", type=Path, required=True)
    parser.add_argument("--collaborate-sid", type=Path, required=True)
    parser.add_argument("--selected-internal", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_dataset.resolve()
    output = args.output_dir.resolve()
    selected = [int(value) for value in json.loads(args.selected_internal.read_text())]
    if not selected:
        raise ValueError("Utility gate selected no SID changes")

    for path in (source / "dataset").glob("*.pkl"):
        link(path, output / "dataset" / path.name)
    excluded = {"rec_sid_migrations.pkl"}
    for path in (source / "vocab").glob("*.pkl"):
        if path.name not in excluded:
            link(path, output / "vocab" / path.name)
    link(args.semantic_sid, output / "emb" / args.semantic_sid.name)
    link(args.collaborate_sid, output / "emb" / args.collaborate_sid.name)

    manifest = {
        "source_dataset": str(source),
        "selection": str(args.selected_internal),
        "changed_items": len(selected),
        "alignment_items": "preserved_from_source",
        "semantic_sid": args.semantic_sid.name,
        "collaborate_sid": args.collaborate_sid.name,
    }
    (output / "utility_gated_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
