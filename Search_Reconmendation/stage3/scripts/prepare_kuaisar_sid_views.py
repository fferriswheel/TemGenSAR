#!/usr/bin/env python3
"""Validate and package aligned KuaiSAR item representations for SID models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--common-embedding", type=Path, required=True)
    parser.add_argument("--search-embedding", type=Path, required=True)
    parser.add_argument("--collaborative-embedding", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"Expected a rank-2 tensor in {path}, got {type(value)}")
    value = value.float().contiguous()
    if not torch.isfinite(value).all():
        raise ValueError(f"Non-finite values found in {path}")
    return value


def main() -> None:
    args = parse_args()
    catalog = pd.read_parquet(args.catalog)
    if "item_id" not in catalog.columns:
        raise ValueError("Catalog must contain item_id")
    item_ids = catalog["item_id"].astype(str).tolist()
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Catalog item IDs are not unique")

    common = load_tensor(args.common_embedding)
    search = load_tensor(args.search_embedding)
    collaborative = load_tensor(args.collaborative_embedding)
    expected_rows = len(item_ids)
    for name, tensor in {
        "common": common,
        "search": search,
        "collaborative": collaborative,
    }.items():
        if tensor.shape[0] != expected_rows:
            raise ValueError(
                f"{name} has {tensor.shape[0]} rows, expected {expected_rows}"
            )

    common = F.normalize(common, dim=-1)
    search = F.normalize(search, dim=-1)
    collaborative = F.normalize(collaborative, dim=-1)
    signal_matched_semantic = F.normalize(torch.cat([common, search], dim=-1), dim=-1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "common": args.output_dir / "common.pt",
        "search": args.output_dir / "search.pt",
        "collaborative": args.output_dir / "collaborative.pt",
        "signal_matched_semantic": args.output_dir / "common_search.pt",
        "item_ids": args.output_dir / "item_ids.json",
    }
    torch.save(common, outputs["common"])
    torch.save(search, outputs["search"])
    torch.save(collaborative, outputs["collaborative"])
    torch.save(signal_matched_semantic, outputs["signal_matched_semantic"])
    outputs["item_ids"].write_text(
        json.dumps(item_ids, ensure_ascii=False), encoding="utf-8"
    )

    summary = {
        "num_items": expected_rows,
        "catalog": str(args.catalog.resolve()),
        "behavior_window": "W0-W2 only",
        "future_behavior_used": False,
        "item_order": "catalog row order",
        "dimensions": {
            "common": common.shape[1],
            "search": search.shape[1],
            "collaborative": collaborative.shape[1],
            "signal_matched_semantic": signal_matched_semantic.shape[1],
        },
        "outputs": {key: str(path.resolve()) for key, path in outputs.items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
