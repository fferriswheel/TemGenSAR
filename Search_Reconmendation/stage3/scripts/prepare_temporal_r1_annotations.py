#!/usr/bin/env python3
"""Build immutable item-level and W2-example annotations for temporal R1."""

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
    parser.add_argument("--rec-w1-dir", type=Path, required=True)
    parser.add_argument("--base-mapping", type=Path, required=True)
    parser.add_argument("--updated-mapping", type=Path, required=True)
    parser.add_argument("--sid-stats", type=Path, required=True)
    parser.add_argument("--drift-csv", type=Path, required=True)
    parser.add_argument("--drift-left", default="C8")
    parser.add_argument("--drift-right", default="W1")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_active_items(path: Path, start: int, end: int) -> set[str]:
    active: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if start <= int(row["timestamp"]) < end:
                active.add(str(row["item_id"]))
    return active


def rec_codes(row: dict) -> tuple[int, ...]:
    return tuple(int(x) for x in row["rec_suffix_codes"])


def percentile_bins(item_ids: list[str], values: dict[str, int]) -> dict[str, str]:
    ordered = sorted(item_ids, key=lambda item: (-values.get(item, 0), item))
    n = len(ordered)
    labels: dict[str, str] = {}
    for rank, item in enumerate(ordered):
        fraction = rank / max(n, 1)
        labels[item] = "head" if fraction < 0.2 else "middle" if fraction < 0.5 else "tail"
    return labels


def rank_quintiles(items: pd.DataFrame) -> dict[str, str]:
    ordered = items.sort_values(["drift_rank_score", "item_id"], ascending=[True, True])
    n = len(ordered)
    labels: dict[str, str] = {}
    for rank, item in enumerate(ordered["item_id"].astype(str)):
        labels[item] = f"D{min(4, int(5 * rank / max(n, 1))) + 1}"
    return labels


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    id_maps = json.loads((args.quarterly_dir / "id_maps.json").read_text(encoding="utf-8"))
    int_to_asin = {int(key): str(value) for key, value in id_maps["int_to_asin"].items()}
    integer_ids = sorted(int_to_asin)
    item_ids = [int_to_asin[index] for index in integer_ids]

    base = json.loads(args.base_mapping.read_text(encoding="utf-8"))
    updated = json.loads(args.updated_mapping.read_text(encoding="utf-8"))
    if set(item_ids) != set(base) or set(item_ids) != set(updated):
        raise ValueError("SID mapping item universe does not match quarterly id map")

    sid_stats = json.loads(args.sid_stats.read_text(encoding="utf-8"))
    active = read_active_items(
        args.rec_w1_dir / "rec_interactions_chrono.jsonl",
        int(sid_stats["update_window"]["start"]),
        int(sid_stats["update_window"]["end"]),
    )
    changed = {item for item in item_ids if rec_codes(base[item]) != rec_codes(updated[item])}
    if not changed <= active:
        raise ValueError("Found changed Rec SID outside the active-item set")

    train = pd.read_pickle(args.quarterly_dir / "dataset" / "rec_train.pkl")
    counts_by_int = Counter(int(x) for x in train["item_id"])
    counts = {asin: counts_by_int[index] for index, asin in int_to_asin.items()}
    popularity = percentile_bins(item_ids, counts)

    drift = pd.read_csv(args.drift_csv)
    drift = drift[(drift["left"] == args.drift_left) & (drift["right"] == args.drift_right)].copy()
    drift["item_id"] = drift["item_id"].astype(str)
    drift_by_item = drift.set_index("item_id").to_dict("index")
    changed_drift = drift[drift["item_id"].isin(changed)][["item_id", "drift_rank_score"]]
    drift_quintile = rank_quintiles(changed_drift)

    item_rows = []
    for index in integer_ids:
        asin = int_to_asin[index]
        old = rec_codes(base[asin])
        new = rec_codes(updated[asin])
        drift_row = drift_by_item.get(asin, {})
        is_active = asin in active
        is_changed = asin in changed
        item_rows.append({
            "item_int": index,
            "item_id": asin,
            "active": is_active,
            "changed": is_changed,
            "update_group": "changed" if is_changed else "active_unchanged" if is_active else "inactive",
            "popularity_group": popularity[asin],
            "t0_rec_events": counts[asin],
            "old_r1": old[0],
            "old_r2": old[1],
            "new_r1": new[0],
            "new_r2": new[1],
            "r1_changed": old[0] != new[0],
            "r2_changed": old[1] != new[1],
            "drift_rank_score": float(drift_row.get("drift_rank_score", 0.0)),
            "popularity_log_ratio": float(drift_row.get("popularity_log_ratio", 0.0)),
            "transition_js": float(drift_row.get("transition_js", 0.0)),
            "event_support": int(drift_row.get("event_support", 0)),
            "drift_eligible": bool(drift_row.get("eligible", False)),
            "drift_selected": bool(drift_row.get("selected_for_update", False)),
            "drift_quintile_changed": drift_quintile.get(asin, "not_changed"),
        })
    item_frame = pd.DataFrame(item_rows).sort_values("item_int")
    item_frame.to_csv(args.output_dir / "item_annotations.csv", index=False)

    w2 = pd.read_pickle(args.quarterly_dir / "windows" / "W2" / "rec_eval.pkl").reset_index(drop=True)
    example_frame = w2[["user_id", "item_id", "ts"]].copy()
    example_frame.insert(0, "row_index", np.arange(len(example_frame)))
    example_frame = example_frame.rename(columns={"item_id": "item_int"})
    example_frame = example_frame.merge(item_frame, on="item_int", how="left", validate="many_to_one")
    if example_frame["item_id"].isna().any():
        raise ValueError("Failed to annotate every W2 Rec example")
    example_frame.to_csv(args.output_dir / "w2_rec_example_annotations.csv", index=False)

    summary = {
        "num_items": len(item_frame),
        "active_items": int(item_frame["active"].sum()),
        "changed_items": int(item_frame["changed"].sum()),
        "r1_changed_items": int(item_frame["r1_changed"].sum()),
        "r2_changed_items": int(item_frame["r2_changed"].sum()),
        "w2_examples": len(example_frame),
        "w2_update_groups": example_frame["update_group"].value_counts().to_dict(),
        "w2_popularity_groups": example_frame["popularity_group"].value_counts().to_dict(),
        "w2_changed_drift_quintiles": example_frame.loc[
            example_frame["changed"], "drift_quintile_changed"
        ].value_counts().sort_index().to_dict(),
        "drift_eligible_items": int(item_frame["drift_eligible"].sum()),
        "drift_selected_items": int(item_frame["drift_selected"].sum()),
        "changed_and_drift_eligible_items": int((item_frame["changed"] & item_frame["drift_eligible"]).sum()),
        "changed_and_drift_selected_items": int((item_frame["changed"] & item_frame["drift_selected"]).sum()),
        "w2_drift_eligible_examples": int(example_frame["drift_eligible"].sum()),
        "w2_drift_selected_examples": int(example_frame["drift_selected"].sum()),
        "definitions": {
            "active": "item has at least one Rec event in the W1 update interval",
            "changed": "operational Rec suffix differs between t0 and W1 mappings",
            "popularity": "rank all 19,999 items by t0 Rec target frequency; top 20%, next 30%, remaining 50%",
            "drift_quintile": (
                f"rank changed items by {args.drift_left}-to-{args.drift_right} drift_rank_score "
                "into five equal-item-count bins"
            ),
            "drift_selected": "top 30% drift_rank_score among support-eligible C8-to-W1 items",
        },
    }
    (args.output_dir / "annotation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
