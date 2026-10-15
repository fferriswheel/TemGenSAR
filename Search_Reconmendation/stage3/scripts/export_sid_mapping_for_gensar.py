#!/usr/bin/env python3
"""Export an ASIN-keyed shared-prefix SID mapping to GenSAR item-index files."""

import argparse
import json
from pathlib import Path

from prepare_gensar_lite import normalize_sid_tokens, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--item-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    item_ids = []
    with args.item_ids.open("r", encoding="utf-8") as handle:
        for line in handle:
            item_ids.append(str(json.loads(line)["item_id"]))
    semantic, collaborate = {}, {}
    for item_int, item_id in enumerate(item_ids, start=1):
        row = mapping[item_id]
        semantic[str(item_int)] = normalize_sid_tokens(row["search_sid"], "semantic")
        collaborate[str(item_int)] = normalize_sid_tokens(row["rec_sid"], "collaborate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    semantic_path = args.output_dir / f"{args.name}_semantic.json"
    collaborate_path = args.output_dir / f"{args.name}_collaborate.json"
    write_json(semantic_path, semantic)
    write_json(collaborate_path, collaborate)
    print(json.dumps({
        "items": len(item_ids),
        "semantic": str(semantic_path),
        "collaborate": str(collaborate_path),
    }, indent=2))


if __name__ == "__main__":
    main()
