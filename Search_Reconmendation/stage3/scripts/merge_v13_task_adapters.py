#!/usr/bin/env python3
"""Merge independently updated Search and Rec adapters into a full v13 checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--search-adapter", type=Path)
    parser.add_argument("--rec-adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", required=True)
    return parser.parse_args()


def merge_task(
    merged: dict[str, torch.Tensor],
    source_path: Path,
    prefix: str,
) -> dict[str, object]:
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    source = payload["model_state_dict"]
    if set(source) != set(merged):
        missing = sorted(set(merged) - set(source))[:5]
        extra = sorted(set(source) - set(merged))[:5]
        raise ValueError(f"State dict mismatch: missing={missing}, extra={extra}")
    keys = [key for key in merged if key.startswith(prefix)]
    if not keys:
        raise ValueError(f"No parameters found with prefix {prefix}")
    for key in keys:
        merged[key] = source[key].detach().cpu().clone()
    changed = payload.get("summary", {}).get("changed_parameter_tensors", [])
    illegal = [key for key in changed if not key.startswith(prefix)]
    if illegal:
        raise ValueError(f"Adapter checkpoint reports non-task changes: {illegal[:5]}")
    return {
        "path": str(source_path),
        "parameter_tensors": len(keys),
        "reported_changed_tensors": changed,
        "summary": payload.get("summary", {}),
    }


def main() -> None:
    args = parse_args()
    if not args.search_adapter and not args.rec_adapter:
        raise ValueError("At least one task adapter is required")
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    original = base["model_state_dict"]
    merged = {key: value.detach().cpu().clone() for key, value in original.items()}
    provenance: dict[str, object] = {
        "window": args.window,
        "base_checkpoint": str(args.base_checkpoint),
    }
    allowed_prefixes = []
    if args.search_adapter:
        provenance["search"] = merge_task(merged, args.search_adapter, "search_adapter.")
        allowed_prefixes.append("search_adapter.")
    if args.rec_adapter:
        provenance["rec"] = merge_task(merged, args.rec_adapter, "rec_adapter.")
        allowed_prefixes.append("rec_adapter.")

    changed = [key for key in merged if not torch.equal(merged[key], original[key].cpu())]
    illegal = [key for key in changed if not key.startswith(tuple(allowed_prefixes))]
    if illegal:
        raise RuntimeError(f"Non-adapter tensors changed: {illegal[:5]}")
    provenance["changed_parameter_tensors"] = changed
    provenance["non_adapter_exactly_preserved"] = True

    output = copy.copy(base)
    output["model_state_dict"] = merged
    history = list(base.get("temporal_adapter_history", []))
    history.append(provenance)
    output["temporal_adapter_history"] = history
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
