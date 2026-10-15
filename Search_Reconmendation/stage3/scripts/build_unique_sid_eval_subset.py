#!/usr/bin/env python3
import argparse
import json
import pickle
from collections import Counter
from pathlib import Path


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def collision_items(mapping):
    counts = Counter(tuple(tokens) for tokens in mapping.values())
    return {
        int(item_id)
        for item_id, tokens in mapping.items()
        if counts[tuple(tokens)] > 1
    }


def latest_rows(frame, limit):
    timestamp = "timestamp" if "timestamp" in frame.columns else "ts"
    return frame.sort_values(timestamp).iloc[-limit:].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--semantic-index", action="append", required=True)
    parser.add_argument("--collaborate-index", action="append", required=True)
    parser.add_argument("--test-limit", type=int, default=1000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["vocab", "emb"]:
        target = args.output_dir / name
        if not target.exists():
            target.symlink_to((args.source_dir / name).resolve(), target_is_directory=True)

    semantic_collisions = set()
    collaborate_collisions = set()
    for name in args.semantic_index:
        semantic_collisions |= collision_items(load_json(args.source_dir / "emb" / name))
    for name in args.collaborate_index:
        collaborate_collisions |= collision_items(load_json(args.source_dir / "emb" / name))

    rows = {}
    for task, filename, excluded in [
        ("rec", "rec_test.pkl", collaborate_collisions),
        ("search", "src_test.pkl", semantic_collisions),
    ]:
        original = latest_rows(
            load_pickle(args.source_dir / "dataset" / filename), args.test_limit
        )
        keep = ~original["item_id"].astype(int).isin(excluded)
        filtered = original.loc[keep].reset_index(drop=True)
        dump_pickle(args.output_dir / "dataset" / filename, filtered)
        rows[task] = {
            "original": len(original),
            "filtered": len(filtered),
            "removed": int((~keep).sum()),
            "removed_fraction": float((~keep).mean()),
        }

    stats = {
        "policy": "common target-unique subset across every supplied SID mapping",
        "semantic_indices": args.semantic_index,
        "collaborate_indices": args.collaborate_index,
        "rows": rows,
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
