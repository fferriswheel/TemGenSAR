#!/usr/bin/env python3
"""Summarize KuaiSAR temporal results globally and on item subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, raw_path = value.split("=", 1)
        if name in result:
            raise ValueError(f"Duplicate name: {name}")
        result[name] = Path(raw_path)
    return result


def load_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("all_prompt_instance_results")
    if prompts is None or len(prompts) != 1:
        raise ValueError(f"Expected one prompt with instance results in {path}")
    hits = np.asarray(prompts[0], dtype=np.float64)
    if hits.ndim != 2 or hits.shape[1] < 10:
        raise ValueError(f"Expected [N, >=10] hit vectors in {path}, got {hits.shape}")
    hit10 = (hits[:, :10].sum(axis=1) > 0).astype(np.float64)
    ndcg10 = (hits[:, :10] / np.log2(np.arange(2, 12))[None, :]).sum(axis=1)
    return hit10, ndcg10


def load_external_item_ids(path: Path) -> list[str]:
    return [str(json.loads(line)["item_id"]) for line in path.read_text().splitlines()]


def changed_item_indices(base_path: Path, current_path: Path, item_ids: list[str]) -> set[int]:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    current = json.loads(current_path.read_text(encoding="utf-8"))
    return {
        index
        for index, item_id in enumerate(item_ids, start=1)
        if base[item_id]["rec_sid"] != current[item_id]["rec_sid"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-pickle", type=Path, required=True)
    parser.add_argument("--item-ids", type=Path, required=True)
    parser.add_argument("--base-mapping", type=Path, required=True)
    parser.add_argument("--result", action="append", default=[], help="NAME=result.json")
    parser.add_argument("--mapping", action="append", default=[], help="NAME=sid_mapping.json")
    parser.add_argument("--focus-csv", action="append", default=[], help="NAME=selection.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = {name: load_metrics(path) for name, path in parse_named_paths(args.result).items()}
    mappings = parse_named_paths(args.mapping)
    focuses = parse_named_paths(args.focus_csv)
    frame = pd.read_pickle(args.eval_pickle).reset_index(drop=True)
    item_ids = load_external_item_ids(args.item_ids)
    index_by_external_id = {item_id: index for index, item_id in enumerate(item_ids, start=1)}

    for name, metrics in results.items():
        if len(metrics[0]) != len(frame):
            raise ValueError(f"{name} has {len(metrics[0])} rows; eval set has {len(frame)}")

    groups: dict[str, np.ndarray] = {"all": np.ones(len(frame), dtype=bool)}
    target_indices = frame["item_id"].to_numpy()
    for name, path in mappings.items():
        changed = changed_item_indices(args.base_mapping, path, item_ids)
        groups[f"changed_by_{name}"] = np.isin(target_indices, list(changed))
    for name, path in focuses.items():
        selection = pd.read_csv(path, dtype={"item_id": str})
        if "selected_for_update" in selection:
            selection = selection[selection["selected_for_update"]]
        indices = {
            index_by_external_id[item_id]
            for item_id in selection["item_id"].astype(str)
            if item_id in index_by_external_id
        }
        groups[f"selected_by_{name}"] = np.isin(target_indices, list(indices))

    rows = []
    for group, mask in groups.items():
        row: dict[str, object] = {"group": group, "samples": int(mask.sum())}
        for name, (hit10, ndcg10) in results.items():
            row[f"{name}_hit10"] = float(hit10[mask].mean()) if mask.any() else None
            row[f"{name}_ndcg10"] = float(ndcg10[mask].mean()) if mask.any() else None
        rows.append(row)

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".csv":
        output.to_csv(args.output, index=False)
    else:
        args.output.write_text(
            json.dumps(output.to_dict("records"), indent=2), encoding="utf-8"
        )
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
