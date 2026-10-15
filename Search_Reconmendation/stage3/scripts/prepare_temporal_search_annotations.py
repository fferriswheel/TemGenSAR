#!/usr/bin/env python3
"""Build item- and example-level annotations for temporal Search evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarterly-dir", type=Path, required=True)
    parser.add_argument("--base-mapping", type=Path, required=True)
    parser.add_argument("--updated-mapping", type=Path, required=True)
    parser.add_argument("--drift-csv", type=Path, required=True)
    parser.add_argument("--injection-manifest", type=Path, required=True)
    parser.add_argument("--window", default="W2")
    parser.add_argument("--drift-left", default="C8")
    parser.add_argument("--drift-right", default="W1")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def search_codes(row: dict) -> tuple[int, ...]:
    return tuple(int(value) for value in row["search_suffix_codes"])


def popularity_bins(item_ids: list[str], counts: Counter[str]) -> dict[str, str]:
    ordered = sorted(item_ids, key=lambda item: (-counts[item], item))
    size = len(ordered)
    return {
        item: "head" if rank / size < 0.2 else "middle" if rank / size < 0.5 else "tail"
        for rank, item in enumerate(ordered)
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    id_maps = json.loads((args.quarterly_dir / "id_maps.json").read_text(encoding="utf-8"))
    int_to_asin = {int(key): str(value) for key, value in id_maps["int_to_asin"].items()}
    item_ids = [int_to_asin[index] for index in sorted(int_to_asin)]

    base = json.loads(args.base_mapping.read_text(encoding="utf-8"))
    updated = json.loads(args.updated_mapping.read_text(encoding="utf-8"))
    if set(base) != set(item_ids) or set(updated) != set(item_ids):
        raise ValueError("SID mapping item universe does not match quarterly item universe")

    injection = json.loads(args.injection_manifest.read_text(encoding="utf-8"))
    injected = set(map(str, injection["selected_items"]))
    changed = {item for item in item_ids if search_codes(base[item]) != search_codes(updated[item])}

    drift = pd.read_csv(args.drift_csv)
    drift = drift[(drift["left"] == args.drift_left) & (drift["right"] == args.drift_right)].copy()
    drift["item_id"] = drift["item_id"].astype(str)
    drift_by_item = drift.set_index("item_id").to_dict("index")
    selected = set(drift.loc[drift["selected_for_update"].astype(bool), "item_id"])

    train = pd.read_pickle(args.quarterly_dir / "dataset" / "src_train.pkl")
    train_counts_int = Counter(int(value) for value in train["item_id"])
    train_counts = Counter({int_to_asin[index]: count for index, count in train_counts_int.items()})
    popularity = popularity_bins(item_ids, train_counts)

    item_rows: list[dict[str, object]] = []
    for item_int in sorted(int_to_asin):
        item = int_to_asin[item_int]
        row = drift_by_item.get(item, {})
        item_rows.append({
            "item_int": item_int,
            "item_id": item,
            "injected": item in injected,
            "drift_eligible": bool(row.get("eligible", False)),
            "drift_selected": item in selected,
            "sid_changed": item in changed,
            "selected_and_injected": item in selected and item in injected,
            "changed_and_injected": item in changed and item in injected,
            "update_group": "changed" if item in changed else "selected_unchanged" if item in selected else "not_selected",
            "popularity_group": popularity[item],
            "t0_search_events": train_counts[item],
            "query_js": float(row.get("query_js", 0.0)),
            "query_centroid_shift": float(row.get("query_centroid_shift", 0.0)),
            "drift_rank_score": float(row.get("drift_rank_score", 0.0)),
        })
    item_frame = pd.DataFrame(item_rows).sort_values("item_int")
    item_frame.to_csv(args.output_dir / "item_annotations.csv", index=False)

    examples = pd.read_pickle(
        args.quarterly_dir / "windows" / args.window / "src_eval.pkl"
    ).reset_index(drop=True)
    example_frame = examples[["user_id", "item_id", "query_id", "ts"]].copy()
    example_frame.insert(0, "row_index", np.arange(len(example_frame)))
    example_frame = example_frame.rename(columns={"item_id": "item_int"})
    example_frame = example_frame.merge(item_frame, on="item_int", how="left", validate="many_to_one")
    if example_frame["item_id"].isna().any():
        raise ValueError("Failed to annotate every Search example")
    example_frame.to_csv(args.output_dir / f"{args.window.lower()}_search_example_annotations.csv", index=False)

    summary = {
        "items": len(item_frame),
        "injected_items": int(item_frame["injected"].sum()),
        "drift_eligible_items": int(item_frame["drift_eligible"].sum()),
        "drift_selected_items": int(item_frame["drift_selected"].sum()),
        "changed_items": int(item_frame["sid_changed"].sum()),
        "selected_and_injected_items": int(item_frame["selected_and_injected"].sum()),
        "changed_and_injected_items": int(item_frame["changed_and_injected"].sum()),
        "examples": len(example_frame),
        "example_counts": {
            column: int(example_frame[column].sum())
            for column in [
                "injected", "drift_eligible", "drift_selected", "sid_changed",
                "selected_and_injected", "changed_and_injected",
            ]
        },
    }
    (args.output_dir / "annotation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
