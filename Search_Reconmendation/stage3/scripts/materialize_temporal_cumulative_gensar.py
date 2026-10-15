#!/usr/bin/env python3
"""Build the cumulative t0+W1 training view used by the full-retrain oracle."""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


def load(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=4)


def link(source, target):
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quarterly-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--semantic-sid", type=Path, required=True)
    p.add_argument("--collaborate-sid", type=Path, required=True)
    p.add_argument("--update-window", default="W1")
    p.add_argument("--eval-window", default="W2")
    args = p.parse_args()

    root = args.quarterly_dir.resolve()
    update = root / "windows" / args.update_window
    evaluation = root / "windows" / args.eval_window
    output = args.output_dir.resolve()
    rec = pd.concat(
        [load(root / "dataset/rec_train.pkl"), load(update / "rec_update.pkl")],
        ignore_index=True,
    ).sort_values(["ts", "user_id"]).reset_index(drop=True)
    src = pd.concat(
        [load(root / "dataset/src_train.pkl"), load(update / "src_update.pkl")],
        ignore_index=True,
    ).sort_values(["ts", "user_id"]).reset_index(drop=True)
    dump(output / "dataset/rec_train.pkl", rec)
    dump(output / "dataset/src_train.pkl", src)
    link(update / "rec_eval.pkl", output / "dataset/rec_val.pkl")
    link(update / "src_eval.pkl", output / "dataset/src_val.pkl")
    link(evaluation / "rec_eval.pkl", output / "dataset/rec_test.pkl")
    link(evaluation / "src_eval.pkl", output / "dataset/src_test.pkl")
    for path in (root / "vocab").glob("*.pkl"):
        link(path, output / "vocab" / path.name)
    link(args.semantic_sid, output / "emb" / args.semantic_sid.name)
    link(args.collaborate_sid, output / "emb" / args.collaborate_sid.name)
    manifest = {
        "rec_train_rows": len(rec), "search_train_rows": len(src),
        "rec_val_rows": len(load(update / "rec_eval.pkl")),
        "search_val_rows": len(load(update / "src_eval.pkl")),
        "rec_test_rows": len(load(evaluation / "rec_eval.pkl")),
        "search_test_rows": len(load(evaluation / "src_eval.pkl")),
    }
    (output / "cumulative_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
