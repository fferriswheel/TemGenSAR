#!/usr/bin/env python3
"""Materialize exact old-to-new Rec SID migration pairs for GenSAR training."""

import argparse
import json
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-sid", type=Path, required=True)
    parser.add_argument("--new-sid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    old = json.loads(args.old_sid.read_text(encoding="utf-8"))
    new = json.loads(args.new_sid.read_text(encoding="utf-8"))
    if old.keys() != new.keys():
        raise ValueError("Old and new SID files have different item keys")

    migrations = [
        {
            "item_id": int(item_id),
            "old_sid": old[item_id],
            "new_sid": new[item_id],
        }
        for item_id in old
        if old[item_id] != new[item_id]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(migrations, handle, protocol=4)
    print(json.dumps({"migrations": len(migrations), "output": str(args.output)}))


if __name__ == "__main__":
    main()
