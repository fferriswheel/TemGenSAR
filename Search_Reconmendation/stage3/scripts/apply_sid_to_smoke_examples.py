#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def convert_file(input_path, output_path, mapping):
    total = 0
    kept = 0
    missing = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", encoding="utf-8") as out:
        for row in read_jsonl(input_path):
            total += 1
            item_id = row["target_item_id"]
            sid = mapping.get(item_id)
            if sid is None:
                missing += 1
                continue
            row = dict(row)
            if row["task"] == "search":
                row["target_sid"] = sid["search_sid"]
            elif row["task"] == "rec":
                row["target_sid"] = sid["rec_sid"]
            else:
                raise ValueError(f"Unknown task: {row['task']}")
            row["sid_source"] = "baseline_modular_sid"
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    return {"input": str(input_path), "output": str(output_path), "total": total, "kept": kept, "missing": missing}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/sid_smoke/sid_mapping.baseline.json",
    )
    parser.add_argument(
        "--smoke-dir",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/smoke",
    )
    args = parser.parse_args()

    mapping = json.load(open(args.mapping, "r", encoding="utf-8"))
    smoke_dir = Path(args.smoke_dir)
    outputs = []
    outputs.append(
        convert_file(
            smoke_dir / "search_smoke_examples.jsonl",
            smoke_dir / "search_smoke_examples.sid.jsonl",
            mapping,
        )
    )
    outputs.append(
        convert_file(
            smoke_dir / "rec_chrono_smoke_examples.jsonl",
            smoke_dir / "rec_chrono_smoke_examples.sid.jsonl",
            mapping,
        )
    )

    mixed_path = smoke_dir / "mixed_smoke_examples.sid.jsonl"
    with mixed_path.open("w", encoding="utf-8") as out:
        for name in ["search_smoke_examples.sid.jsonl", "rec_chrono_smoke_examples.sid.jsonl"]:
            with (smoke_dir / name).open("r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)

    stats = {"files": outputs, "mixed_output": str(mixed_path)}
    with (smoke_dir / "sid_apply_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
