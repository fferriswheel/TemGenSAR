#!/usr/bin/env python3
"""Convert GenSAR token-index JSON files to the shared temporal SID schema."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TOKEN_RE = re.compile(r"^<(shared|semantic|collaborate)_([a-z])_(\d+)>$")


def parse_codes(tokens: list[str], expected_private: str) -> tuple[list[int], list[int]]:
    shared: list[int] = []
    private: list[int] = []
    private_started = False
    for token in tokens:
        match = TOKEN_RE.fullmatch(token)
        if not match:
            raise ValueError(f"Invalid GenSAR SID token: {token}")
        family, _, code = match.groups()
        if family == "shared":
            if private_started:
                raise ValueError(f"Shared code after private code: {tokens}")
            shared.append(int(code))
        elif family == expected_private:
            private_started = True
            private.append(int(code))
        else:
            raise ValueError(
                f"Expected {expected_private} private token, found {family}: {token}"
            )
    if not shared or not private:
        raise ValueError(f"SID must contain shared and private codes: {tokens}")
    return shared, private


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--collaborate", type=Path, required=True)
    parser.add_argument("--item-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    semantic = json.loads(args.semantic.read_text(encoding="utf-8"))
    collaborate = json.loads(args.collaborate.read_text(encoding="utf-8"))
    item_ids = [
        str(json.loads(line)["item_id"])
        for line in args.item_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_keys = {str(index) for index in range(1, len(item_ids) + 1)}
    if set(semantic) != expected_keys or set(collaborate) != expected_keys:
        raise ValueError("GenSAR index files do not match the item catalog")

    mapping: dict[str, dict[str, list[int]]] = {}
    for index, item_id in enumerate(item_ids, start=1):
        search_prefix, search_suffix = parse_codes(semantic[str(index)], "semantic")
        rec_prefix, rec_suffix = parse_codes(collaborate[str(index)], "collaborate")
        if search_prefix != rec_prefix:
            raise ValueError(f"Shared prefix mismatch for item {item_id}")
        mapping[item_id] = {
            "prefix_codes": search_prefix,
            "search_suffix_codes": search_suffix,
            "rec_suffix_codes": rec_suffix,
            "search_sid": search_prefix + search_suffix,
            "rec_sid": rec_prefix + rec_suffix,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(json.dumps({"items": len(mapping), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
