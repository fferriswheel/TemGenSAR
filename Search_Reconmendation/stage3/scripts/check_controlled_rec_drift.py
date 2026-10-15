#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from run_temporal_drift_detection import (
    load_windows,
    original_aggregate,
    quarterly_protocol,
    rec_metrics,
    write_json,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--injected-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-relation-support", type=int, default=5)
    parser.add_argument("--neighbor-top-k", type=int, default=20)
    args = parser.parse_args()

    windows, _, pairs = quarterly_protocol()
    original, original_coverage, _ = load_windows(
        args.original_dir / "rec_interactions_chrono.jsonl", "rec", windows
    )
    injected, injected_coverage, _ = load_windows(
        args.injected_dir / "rec_interactions_chrono.jsonl", "rec", windows
    )
    rows = []
    for left, right in pairs:
        base = rec_metrics(
            original_aggregate(original[left], "rec"),
            original_aggregate(original[right], "rec"),
            args.minimum_relation_support,
            args.neighbor_top_k,
        )
        drift = rec_metrics(
            original_aggregate(injected[left], "rec"),
            original_aggregate(injected[right], "rec"),
            args.minimum_relation_support,
            args.neighbor_top_k,
        )
        rows.append({
            "left": left,
            "right": right,
            "original": base,
            "injected": drift,
            "delta": {
                key: drift[key] - base[key]
                for key in ("popularity_js", "transition_js", "neighbor_jaccard_drift")
            },
        })

    coverage_equal = all(
        original_coverage[name][key] == injected_coverage[name][key]
        for name in windows for key in ("events", "users")
    )
    result = {
        "search_sha256_equal": sha256(args.original_dir / "search_interactions.jsonl")
        == sha256(args.injected_dir / "search_interactions.jsonl"),
        "rec_window_event_and_user_counts_equal": coverage_equal,
        "pairs": rows,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
