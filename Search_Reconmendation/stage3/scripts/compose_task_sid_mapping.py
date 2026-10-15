#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from build_neural_rq_sid_v2 import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-mapping", required=True)
    parser.add_argument("--rec-mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    search_mapping = json.loads(Path(args.search_mapping).read_text(encoding="utf-8"))
    rec_mapping = json.loads(Path(args.rec_mapping).read_text(encoding="utf-8"))
    if set(search_mapping) != set(rec_mapping):
        raise ValueError("search and rec mappings contain different item sets")

    output = {}
    prefix_mismatches = 0
    for item_id, search_row in search_mapping.items():
        rec_row = rec_mapping[item_id]
        if search_row["prefix_codes"] != rec_row["prefix_codes"]:
            prefix_mismatches += 1
            continue
        output[item_id] = {
            "prefix": search_row["prefix"],
            "search_sid": search_row["search_sid"],
            "rec_sid": rec_row["rec_sid"],
            "search_suffix": search_row["search_suffix"],
            "rec_suffix": rec_row["rec_suffix"],
            "prefix_codes": search_row["prefix_codes"],
            "search_suffix_codes": search_row["search_suffix_codes"],
            "rec_suffix_codes": rec_row["rec_suffix_codes"],
        }
    if prefix_mismatches:
        raise ValueError(f"{prefix_mismatches} items have inconsistent shared prefixes")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / "sid_mapping.composed.json"
    write_json(mapping_path, output)
    write_json(output_dir / "sid_mapping.v2.json", output)
    write_json(output_dir / "sid_mapping.v1.json", output)
    write_json(
        output_dir / "compose_stats.json",
        {
            "num_items": len(output),
            "prefix_mismatches": prefix_mismatches,
            "search_mapping": str(Path(args.search_mapping).resolve()),
            "rec_mapping": str(Path(args.rec_mapping).resolve()),
            "output": str(mapping_path.resolve()),
        },
    )
    print(json.dumps({"num_items": len(output), "output": str(mapping_path)}, indent=2))


if __name__ == "__main__":
    main()
