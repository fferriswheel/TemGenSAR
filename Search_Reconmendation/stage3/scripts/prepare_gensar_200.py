#!/usr/bin/env python3
import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_pickle(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def item_text(row):
    title = row.get("title") or ""
    desc = row.get("description") or ""
    return {
        "title": str(title),
        "description": str(desc),
    }


def normalize_sid_tokens(tokens, task):
    """Convert our SID tokens to GenSAR-style semantic/collaborative tokens."""
    out = []
    for token in tokens:
        if token.startswith("<P"):
            out.append(token.replace("<P", "<shared_", 1))
        elif token.startswith("<S"):
            out.append(token.replace("<S", "<semantic_", 1))
        elif token.startswith("<R"):
            out.append(token.replace("<R", "<collaborate_", 1))
        else:
            prefix = "semantic" if task == "semantic" else "collaborate"
            out.append(f"<{prefix}_{token.strip('<>')}>")
    return out


def split_train_valid(rows, valid_ratio, seed):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n_valid = int(round(len(rows) * valid_ratio))
    return rows[n_valid:], rows[:n_valid]


def sample_negs(pos_item, all_items, rng, max_negs):
    candidates = [item for item in all_items if item != pos_item]
    rng.shuffle(candidates)
    return candidates[:max_negs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics",
    )
    parser.add_argument(
        "--sid-mapping",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/sid_v2_m7b_text_encoder_fusion_plc_lora_collision005_balanced_unique_200_p8_4_suf32_8/sid_mapping.v2.json",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/cjk/external/GenSAR/data/SR200_M7b_collision005_unique",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--max-negs", type=int, default=199)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    rng = random.Random(args.seed)

    sid_mapping = json.loads(Path(args.sid_mapping).read_text(encoding="utf-8"))
    asin_list = sorted(sid_mapping)
    asin_to_int = {asin: idx + 1 for idx, asin in enumerate(asin_list)}
    int_to_asin = {idx: asin for asin, idx in asin_to_int.items()}
    all_int_items = sorted(int_to_asin)

    item_vocab = [{"title": "", "description": ""}]
    item_rows = {row["item_id"]: row for row in read_jsonl(data_dir / "items.jsonl")}
    for asin in asin_list:
        item_vocab.append(item_text(item_rows.get(asin, {"title": asin, "description": ""})))

    query_rows = list(read_jsonl(data_dir / "queries.jsonl"))
    max_qid = max(int(row["query_id"]) for row in query_rows)
    query_vocab = [{"query": ""} for _ in range(max_qid + 1)]
    for row in query_rows:
        query_vocab[int(row["query_id"])] = {"query": row["query"]}

    semantic_indices = {}
    collaborate_indices = {}
    for asin, item_int in asin_to_int.items():
        semantic_indices[str(item_int)] = normalize_sid_tokens(
            sid_mapping[asin]["search_sid"], task="semantic"
        )
        collaborate_indices[str(item_int)] = normalize_sid_tokens(
            sid_mapping[asin]["rec_sid"], task="collaborate"
        )

    raw_rec_by_user = defaultdict(list)
    for row in read_jsonl(data_dir / "rec_interactions_chrono.jsonl"):
        asin = row["item_id"]
        if asin not in asin_to_int:
            continue
        raw_rec_by_user[row["user_id"]].append(
            {
                "asin": asin,
                "item": asin_to_int[asin],
                "ts": int(row.get("timestamp", 0)),
                "position": int(row.get("position", 0)),
            }
        )
    for rows in raw_rec_by_user.values():
        rows.sort(key=lambda x: (x["ts"], x["position"], x["item"]))

    users = sorted(
        set(raw_rec_by_user)
        | {row["user_id"] for row in read_jsonl(data_dir / "search_interactions.jsonl") if row["item_id"] in asin_to_int}
    )
    user_to_int = {user: idx + 1 for idx, user in enumerate(users)}

    rec_pos_by_user = {}
    for user, rows in raw_rec_by_user.items():
        by_asin = defaultdict(list)
        for idx, row in enumerate(rows):
            by_asin[row["asin"]].append(idx)
        rec_pos_by_user[user] = by_asin

    src_session_vocab = {}
    src_sessions_by_user = defaultdict(list)
    src_train_raw = []
    src_test = []
    session_id = 1
    for order, row in enumerate(read_jsonl(data_dir / "search_interactions.jsonl")):
        asin = row["item_id"]
        if asin not in asin_to_int:
            continue
        user = row["user_id"]
        user_int = user_to_int[user]
        item_int = asin_to_int[asin]
        query_id = int(row["query_id"])
        ts = 10**12 + order
        rec_positions = rec_pos_by_user.get(user, {}).get(asin, [])
        rec_his_count = rec_positions[0] if rec_positions else len(raw_rec_by_user.get(user, []))
        src_his_count = len(src_sessions_by_user[user])
        src_session_vocab[session_id] = {
            "query_id": query_id,
            "query": row.get("query", query_vocab[query_id]["query"]),
            "pos_items": [item_int],
            "time_list": [ts],
        }
        src_sessions_by_user[user].append(session_id)
        out = {
            "user_id": user_int,
            "item_id": item_int,
            "search_session_id": session_id,
            "ts": ts,
            "query_id": query_id,
            "rec_his": rec_his_count,
            "src_session_his": src_his_count,
            "src_his": src_his_count,
            "all_his": rec_his_count + src_his_count,
            "neg_items": sample_negs(item_int, all_int_items, rng, args.max_negs),
        }
        if row.get("split") == "test":
            src_test.append(out)
        else:
            src_train_raw.append(out)
        session_id += 1

    rec_rows_raw = []
    for user, rows in raw_rec_by_user.items():
        user_int = user_to_int[user]
        for pos in range(1, len(rows)):
            item_int = rows[pos]["item"]
            rec_rows_raw.append(
                {
                    "user_id": user_int,
                    "item_id": item_int,
                    "ts": rows[pos]["ts"],
                    "rec_his": pos,
                    "src_session_his": 0,
                    "src_his": 0,
                    "all_his": pos,
                    "neg_items": sample_negs(item_int, all_int_items, rng, args.max_negs),
                }
            )

    rec_train, rec_val = split_train_valid(rec_rows_raw, args.valid_ratio, args.seed)
    src_train, src_val = split_train_valid(src_train_raw, args.valid_ratio, args.seed)

    user_vocab = {}
    for user, user_int in user_to_int.items():
        rec_rows = raw_rec_by_user.get(user, [])
        user_vocab[user_int] = {
            "rec_his": [row["item"] for row in rec_rows],
            "rec_his_ts": [row["ts"] for row in rec_rows],
            "src_session_his": src_sessions_by_user.get(user, []),
        }

    dump_pickle(out_dir / "vocab/user_vocab.pkl", user_vocab)
    dump_pickle(out_dir / "vocab/item_vocab.pkl", item_vocab)
    dump_pickle(out_dir / "vocab/query_vocab.pkl", query_vocab)
    dump_pickle(out_dir / "vocab/src_session_vocab.pkl", src_session_vocab)
    dump_pickle(out_dir / "vocab/src_item_set.pkl", all_int_items)
    dump_pickle(out_dir / "vocab/rec_item_set.pkl", all_int_items)

    for name, rows in [
        ("rec_train", rec_train),
        ("rec_val", rec_val),
        ("rec_test", rec_val if not rec_val else rec_val),
        ("src_train", src_train),
        ("src_val", src_val),
        ("src_test", src_test if src_test else src_val),
    ]:
        (out_dir / "dataset").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_pickle(out_dir / "dataset" / f"{name}.pkl")

    write_json(out_dir / "emb/ours_search_semantic.json", semantic_indices)
    write_json(out_dir / "emb/ours_rec_collaborate.json", collaborate_indices)
    write_json(
        out_dir / "id_maps.json",
        {
            "asin_to_int": asin_to_int,
            "int_to_asin": int_to_asin,
            "user_to_int": user_to_int,
        },
    )
    write_json(
        out_dir / "stats.json",
        {
            "items": len(all_int_items),
            "users": len(user_vocab),
            "queries": len(query_vocab),
            "src_sessions": len(src_session_vocab),
            "rec_rows": {
                "train": len(rec_train),
                "valid": len(rec_val),
                "test": len(rec_val),
            },
            "src_rows": {
                "train": len(src_train),
                "valid": len(src_val),
                "test": len(src_test if src_test else src_val),
            },
            "sid_mapping": str(Path(args.sid_mapping)),
            "notes": [
                "Search timestamps are synthetic because the public qrels do not include timestamps.",
                "Rec task uses rec history only by setting src_session_his=0 for rec rows.",
            ],
        },
    )
    print(f"Wrote GenSAR dataset to {out_dir}")
    print(json.dumps(json.loads((out_dir / "stats.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
