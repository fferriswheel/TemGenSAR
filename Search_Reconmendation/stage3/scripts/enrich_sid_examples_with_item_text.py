#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path


SPACE_RE = re.compile(r"\s+")


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(x) for x in value if x)
    return SPACE_RE.sub(" ", str(value)).strip()


def category_text(categories):
    if not categories:
        return ""
    if isinstance(categories, list):
        paths = []
        for path in categories:
            if isinstance(path, list):
                paths.append(" > ".join(normalize_text(x) for x in path if x))
            else:
                paths.append(normalize_text(path))
        return " ; ".join(x for x in paths if x)
    return normalize_text(categories)


def trim_words(text, max_words):
    words = normalize_text(text).split()
    return " ".join(words[:max_words])


def load_items(path):
    items = {}
    for row in read_jsonl(path):
        item_id = row.get("item_id")
        if not item_id:
            continue
        title = trim_words(row.get("title", ""), 16)
        brand = trim_words(row.get("brand", ""), 6)
        cats = trim_words(category_text(row.get("categories")), 20)
        parts = []
        if title:
            parts.append(f"title: {title}")
        if brand:
            parts.append(f"brand: {brand}")
        if cats:
            parts.append(f"category: {cats}")
        items[item_id] = " ; ".join(parts)
    return items


def enrich_rec_input(row, items):
    history = row.get("source", {}).get("history_item_ids", [])
    sid_tokens = row["input_text"].split()
    if not history or not sid_tokens:
        return row

    prefix = sid_tokens[0]
    sid_chunks = sid_tokens[1:]
    item_chunks = []
    for idx, item_id in enumerate(history):
        sid = " ".join(sid_chunks[idx * 4 : (idx + 1) * 4])
        meta = items.get(item_id, "")
        if sid and meta:
            item_chunks.append(f"{sid} {meta}")
        elif sid:
            item_chunks.append(sid)
    new_row = dict(row)
    new_row["input_text"] = f"{prefix} " + " ; ".join(item_chunks)
    new_row["input_enrichment"] = "history_sid_title_brand_category"
    return new_row


def process_file(input_path, output_path, items):
    rows = []
    counts = {"total": 0, "rec": 0, "enriched_rec": 0}
    for row in read_jsonl(input_path):
        counts["total"] += 1
        if row.get("task") == "rec":
            counts["rec"] += 1
            new_row = enrich_rec_input(row, items)
            if new_row.get("input_enrichment"):
                counts["enriched_rec"] += 1
            row = new_row
        rows.append(row)
    write_jsonl(output_path, rows)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--items",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/items.jsonl",
    )
    parser.add_argument(
        "--files",
        default="train.jsonl,valid.jsonl,test.jsonl,train.rec.jsonl,valid.rec.jsonl,test.rec.jsonl,train.search.jsonl,valid.search.jsonl,test.search.jsonl,mixed.jsonl",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    items = load_items(args.items)
    summary = {"input_dir": str(input_dir), "output_dir": str(output_dir), "items": len(items), "files": {}}

    for name in [x.strip() for x in args.files.split(",") if x.strip()]:
        src = input_dir / name
        if not src.exists():
            continue
        dst = output_dir / name
        summary["files"][name] = process_file(src, dst, items)

    stats_path = input_dir / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats["input_enrichment"] = "history_sid_title_brand_category"
        (output_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    (output_dir / "enrichment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
