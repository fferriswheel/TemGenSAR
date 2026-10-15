#!/usr/bin/env python3
"""Build item and test-example annotations for the R3 rolling trajectory."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarterly-dir", type=Path, required=True)
    parser.add_argument("--fixed-mapping", type=Path, required=True)
    parser.add_argument("--w1-mapping", type=Path, required=True)
    parser.add_argument("--rolling-dir", type=Path, required=True)
    parser.add_argument("--search-drift", type=Path, required=True)
    parser.add_argument("--rec-drift", type=Path, required=True)
    parser.add_argument("--injection-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping dictionary in {path}")
    return payload


def code(row: dict, task: str) -> tuple[int, ...]:
    return tuple(int(value) for value in row[f"{task}_suffix_codes"])


def drift_pair(window: int) -> tuple[str, str]:
    return ("C8", "W1") if window == 1 else (f"W{window - 1}", f"W{window}")


def mapping_path(args: argparse.Namespace, window: int) -> Path:
    if window == 0:
        return args.fixed_mapping
    if window == 1:
        return args.w1_mapping
    return args.rolling_dir / f"W{window}" / "mappings" / "selective_joint.json"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    id_maps = json.loads((args.quarterly_dir / "id_maps.json").read_text(encoding="utf-8"))
    int_to_asin = {int(key): str(value) for key, value in id_maps["int_to_asin"].items()}
    all_items = set(int_to_asin.values())
    mappings = {window: load_mapping(mapping_path(args, window)) for window in range(10)}
    for window, mapping in mappings.items():
        if set(mapping) != all_items:
            raise ValueError(f"W{window} mapping item universe mismatch")

    drift_frames = {
        "search": pd.read_csv(args.search_drift, dtype={"item_id": str}),
        "rec": pd.read_csv(args.rec_drift, dtype={"item_id": str}),
    }
    injected: set[str] = set()
    if args.injection_manifest:
        manifest = json.loads(args.injection_manifest.read_text(encoding="utf-8"))
        selected = manifest.get("selected_items", [])
        if isinstance(selected, list):
            injected = set(map(str, selected))

    change_history: dict[str, Counter[str]] = {
        "search": Counter(),
        "rec": Counter(),
    }
    summaries: list[dict[str, object]] = []

    for window in range(1, 10):
        update_window = f"W{window}"
        eval_window = f"W{window + 1}"
        previous = mappings[window - 1]
        current = mappings[window]
        left, right = drift_pair(window)

        for task in ("search", "rec"):
            drift = drift_frames[task]
            pair = drift[(drift["left"] == left) & (drift["right"] == right)].copy()
            drift_by_item = pair.set_index("item_id").to_dict("index")
            selected = set(pair.loc[pair["selected_for_update"].astype(bool), "item_id"])
            changed = {
                item for item in all_items
                if code(previous[item], task) != code(current[item], task)
            }

            data_name = "src" if task == "search" else "rec"
            update = pd.read_pickle(
                args.quarterly_dir / "windows" / update_window / f"{data_name}_update.pkl"
            ).reset_index(drop=True)
            evaluation = pd.read_pickle(
                args.quarterly_dir / "windows" / eval_window / f"{data_name}_eval.pkl"
            ).reset_index(drop=True)
            update_counts = Counter(int(value) for value in update["item_id"])
            eval_counts = Counter(int(value) for value in evaluation["item_id"])

            item_rows: list[dict[str, object]] = []
            for item_int in sorted(int_to_asin):
                item = int_to_asin[item_int]
                drift_row = drift_by_item.get(item, {})
                was_changed = change_history[task][item] > 0
                is_changed = item in changed
                old_code = code(previous[item], task)
                new_code = code(current[item], task)
                item_rows.append({
                    "item_int": item_int,
                    "item_id": item,
                    "task": task,
                    "update_window": update_window,
                    "eval_window": eval_window,
                    "drift_left": left,
                    "drift_right": right,
                    "drift_eligible": bool(drift_row.get("eligible", False)),
                    "drift_selected": item in selected,
                    "sid_changed": is_changed,
                    "selected_unchanged": item in selected and not is_changed,
                    "first_change": is_changed and not was_changed,
                    "repeat_change": is_changed and was_changed,
                    "prior_change_count": int(change_history[task][item]),
                    "changed_layers": sum(a != b for a, b in zip(old_code, new_code)),
                    "update_exposure": int(update_counts[item_int]),
                    "eval_exposure": int(eval_counts[item_int]),
                    "injected": task == "search" and item in injected,
                    "event_support": int(drift_row.get("event_support", 0)),
                    "drift_rank_score": float(drift_row.get("drift_rank_score", 0.0)),
                    "query_js": float(drift_row.get("query_js", 0.0)),
                    "query_centroid_shift": float(drift_row.get("query_centroid_shift", 0.0)),
                    "popularity_log_ratio": float(drift_row.get("popularity_log_ratio", 0.0)),
                    "transition_js": float(drift_row.get("transition_js", 0.0)),
                    "old_code": "-".join(map(str, old_code)),
                    "new_code": "-".join(map(str, new_code)),
                })
            item_frame = pd.DataFrame(item_rows).sort_values("item_int")

            example = evaluation[[column for column in [
                "user_id", "item_id", "query_id", "search_session_id", "ts"
            ] if column in evaluation.columns]].copy()
            example.insert(0, "row_index", np.arange(len(example)))
            example = example.rename(columns={"item_id": "item_int"})
            example = example.merge(item_frame, on="item_int", how="left", validate="many_to_one")
            if example["item_id"].isna().any():
                raise ValueError(f"Failed to annotate all {eval_window} {task} examples")

            destination = args.output_dir / update_window / task
            destination.mkdir(parents=True, exist_ok=True)
            item_frame.to_csv(destination / "item_annotations.csv", index=False)
            example.to_csv(destination / "example_annotations.csv", index=False)

            summaries.append({
                "update_window": update_window,
                "eval_window": eval_window,
                "task": task,
                "selected_items": len(selected),
                "changed_items": len(changed),
                "first_change_items": int(item_frame["first_change"].sum()),
                "repeat_change_items": int(item_frame["repeat_change"].sum()),
                "selected_eval_examples": int(example["drift_selected"].sum()),
                "changed_eval_examples": int(example["sid_changed"].sum()),
                "first_change_eval_examples": int(example["first_change"].sum()),
                "repeat_change_eval_examples": int(example["repeat_change"].sum()),
                "injected_eval_examples": int(example["injected"].sum()),
                "total_eval_examples": len(example),
            })
            for item in changed:
                change_history[task][item] += 1

    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "exposure_summary.csv", index=False)
    (args.output_dir / "exposure_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
