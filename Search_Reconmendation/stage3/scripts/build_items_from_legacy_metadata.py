#!/usr/bin/env python3
import argparse
import ast
import gzip
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_meta_line(line):
    # The legacy Amazon metadata file is Python-literal-like, not strict JSON.
    return ast.literal_eval(line)


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v)
    return str(value)


def normalize_categories(value):
    if not value:
        return []
    if isinstance(value, list):
        if value and all(isinstance(v, list) for v in value):
            return [[str(x) for x in path] for path in value]
        return [[str(x) for x in value]]
    return [[str(value)]]


def build(meta_path, item_ids_path, output_dir):
    meta_path = Path(meta_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    needed_items = {row["item_id"] for row in read_jsonl(item_ids_path)}
    found = {}
    total_meta_rows = 0

    with gzip.open(meta_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            total_meta_rows += 1
            obj = parse_meta_line(line)
            asin = obj.get("asin")
            if asin not in needed_items or asin in found:
                continue

            title = normalize_text(obj.get("title"))
            description = normalize_text(obj.get("description"))
            categories = normalize_categories(obj.get("categories"))
            brand = normalize_text(obj.get("brand"))
            price = normalize_text(obj.get("price"))
            sales_rank = obj.get("salesRank", {})
            related = obj.get("related", {})

            text_parts = [title, description, brand]
            for path in categories[:2]:
                text_parts.append(" ".join(path))
            item_text = " ".join(part for part in text_parts if part).strip()

            found[asin] = {
                "item_id": asin,
                "title": title,
                "description": description,
                "categories": categories,
                "brand": brand,
                "price": price,
                "sales_rank": sales_rank,
                "related": related,
                "item_text": item_text,
            }
            if len(found) == len(needed_items):
                break

    missing = sorted(needed_items - set(found))
    output_path = output_dir / "items.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for item_id in sorted(found):
            f.write(json.dumps(found[item_id], ensure_ascii=False) + "\n")

    missing_path = output_dir / "missing_item_metadata.jsonl"
    with missing_path.open("w", encoding="utf-8") as f:
        for item_id in missing:
            f.write(json.dumps({"item_id": item_id}, ensure_ascii=False) + "\n")

    stats = {
        "meta_path": str(meta_path),
        "item_ids_path": str(item_ids_path),
        "total_needed_items": len(needed_items),
        "total_meta_rows_scanned": total_meta_rows,
        "items_found": len(found),
        "items_missing": len(missing),
        "missing_ratio": len(missing) / max(len(needed_items), 1),
        "output_path": str(output_path),
        "missing_path": str(missing_path),
    }
    with (output_dir / "items_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta-path",
        default="/data/cjk/Search_Reconmendation/stage3/raw/amazon_legacy_5core/meta_Electronics.json.gz",
    )
    parser.add_argument(
        "--item-ids",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/item_ids.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics",
    )
    args = parser.parse_args()

    stats = build(args.meta_path, args.item_ids, args.output_dir)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
