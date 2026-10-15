#!/usr/bin/env python3
import argparse
import bisect
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
        pickle.dump(obj, f, protocol=4)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def item_text(row):
    return {
        "title": str(row.get("title") or ""),
        "description": str(row.get("description") or ""),
    }


def normalize_sid_tokens(tokens, task):
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


def load_item_ids(data_dir):
    item_id_path = data_dir / "item_ids.jsonl"
    if item_id_path.exists():
        return [row["item_id"] for row in read_jsonl(item_id_path)]
    return [row["item_id"] for row in read_jsonl(data_dir / "items.jsonl")]


def split_search_train_val(rows, valid_ratio, seed):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n_valid = int(round(len(rows) * valid_ratio))
    return rows[n_valid:], rows[:n_valid]


def split_search_by_user(raw_search_by_user):
    train = []
    val = []
    test = []
    for rows in raw_search_by_user.values():
        if len(rows) >= 3:
            train.extend(rows[:-2])
            val.append(rows[-2])
            test.append(rows[-1])
        elif len(rows) == 2:
            train.append(rows[0])
            test.append(rows[1])
        elif len(rows) == 1:
            train.append(rows[0])
    return train, val, test


def split_rec_by_user(raw_rec_by_user):
    train = []
    val = []
    test = []
    for rows in raw_rec_by_user.values():
        # Candidate rows start from position 1 because position 0 has no history.
        cand = []
        for pos in range(1, len(rows)):
            cand.append(
                {
                    "user_id": rows[pos]["user_int"],
                    "item_id": rows[pos]["item"],
                    "ts": rows[pos]["ts"],
                    "rec_his": pos,
                    "src_session_his": 0,
                    "src_his": 0,
                    "all_his": pos,
                }
            )
        if len(cand) >= 3:
            train.extend(cand[:-2])
            val.append(cand[-2])
            test.append(cand[-1])
        elif len(cand) == 2:
            train.append(cand[0])
            test.append(cand[1])
        elif len(cand) == 1:
            train.append(cand[0])
    return train, val, test


def split_strict_by_user(rows_by_user):
    """GenSAR leave-one-out after dual-history filtering."""
    train = []
    val = []
    test = []
    for rows in rows_by_user.values():
        valid = [
            row
            for row in rows
            if int(row["rec_his"]) > 0 and int(row["src_session_his"]) > 0
        ]
        if len(valid) < 3:
            continue
        train.extend(valid[:-2])
        val.append(valid[-2])
        test.append(valid[-1])
    return train, val, test


def pair_cap_rows(val_rows, test_rows, max_users, seed):
    val_by_user = {int(row["user_id"]): row for row in val_rows}
    test_by_user = {int(row["user_id"]): row for row in test_rows}
    users = sorted(set(val_by_user) & set(test_by_user))
    if max_users > 0 and len(users) > max_users:
        rng = random.Random(seed)
        users = sorted(rng.sample(users, max_users))
    return [val_by_user[user] for user in users], [test_by_user[user] for user in users]


def sample_negs(pos_item, all_items, rng, max_negs):
    pos_idx = bisect.bisect_left(all_items, pos_item)
    if pos_idx == len(all_items) or all_items[pos_idx] != pos_item:
        raise ValueError(f"positive item {pos_item} is missing from the item vocabulary")
    candidate_count = len(all_items) - 1
    if max_negs <= 0 or max_negs >= candidate_count:
        candidates = [item for item in all_items if item != pos_item]
        rng.shuffle(candidates)
        return candidates
    # Sample indices from the compact range and skip the positive's position.
    sampled = rng.sample(range(candidate_count), max_negs)
    return [all_items[idx if idx < pos_idx else idx + 1] for idx in sampled]


def attach_negs(rows, all_items, rng, max_negs):
    out = []
    for row in rows:
        row = dict(row)
        row["neg_items"] = sample_negs(row["item_id"], all_items, rng, max_negs)
        out.append(row)
    return out


def cap_rows(rows, max_rows, seed):
    rows = list(rows)
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:max_rows]


def parse_sid_specs(values):
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError("--sid-spec must be NAME=/path/to/sid_mapping.json")
        name, path = value.split("=", 1)
        specs.append((name, Path(path)))
    return specs


def write_sid_files(out_dir, sid_specs, asin_to_int):
    written = {}
    for name, path in sid_specs:
        sid_mapping = json.loads(path.read_text(encoding="utf-8"))
        semantic_indices = {}
        collaborate_indices = {}
        for asin, item_int in asin_to_int.items():
            if asin not in sid_mapping:
                raise KeyError(f"{asin} missing in {path}")
            semantic_indices[str(item_int)] = normalize_sid_tokens(
                sid_mapping[asin]["search_sid"], task="semantic"
            )
            collaborate_indices[str(item_int)] = normalize_sid_tokens(
                sid_mapping[asin]["rec_sid"], task="collaborate"
            )
        semantic_file = f"{name}_semantic.json"
        collaborate_file = f"{name}_collaborate.json"
        write_json(out_dir / "emb" / semantic_file, semantic_indices)
        write_json(out_dir / "emb" / collaborate_file, collaborate_indices)
        written[name] = {
            "source": str(path),
            "semantic_index_file": semantic_file,
            "collaborate_index_file": collaborate_file,
        }
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/cjk/Search_Reconmendation/stage3/data/gensar_lite_5k")
    parser.add_argument("--output-dir", default="/data/cjk/external/GenSAR/data/GenSAR_Lite_5K")
    parser.add_argument("--sid-spec", action="append", default=[])
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument(
        "--search-timestamp-policy",
        choices=["synthetic", "source"],
        default="synthetic",
    )
    parser.add_argument(
        "--search-split-policy",
        choices=["source", "per_user_chrono"],
        default="source",
    )
    parser.add_argument("--max-negs", type=int, default=999)
    parser.add_argument("--max-val-per-task", type=int, default=5000)
    parser.add_argument("--max-test-per-task", type=int, default=5000)
    parser.add_argument(
        "--strict-gensar-split",
        action="store_true",
        help="Require both histories and >=3 valid rows before per-user leave-one-out.",
    )
    parser.add_argument(
        "--paired-eval-users",
        type=int,
        default=0,
        help="In strict mode, retain the same sampled users in validation and test.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    rng = random.Random(args.seed)

    asin_list = load_item_ids(data_dir)
    asin_to_int = {asin: idx + 1 for idx, asin in enumerate(asin_list)}
    int_to_asin = {idx: asin for asin, idx in asin_to_int.items()}
    all_int_items = sorted(int_to_asin)

    item_rows = {row["item_id"]: row for row in read_jsonl(data_dir / "items.jsonl")}
    item_vocab = [{"title": "", "description": ""}]
    for asin in asin_list:
        item_vocab.append(item_text(item_rows.get(asin, {"title": asin, "description": ""})))

    query_rows = list(read_jsonl(data_dir / "queries.jsonl"))
    max_qid = max(int(row["query_id"]) for row in query_rows)
    query_vocab = [{"query": ""} for _ in range(max_qid + 1)]
    for row in query_rows:
        query_vocab[int(row["query_id"])] = {"query": row["query"]}

    raw_rec_by_user = defaultdict(list)
    for row in read_jsonl(data_dir / "rec_interactions_chrono.jsonl"):
        asin = row["item_id"]
        if asin not in asin_to_int:
            continue
        raw_rec_by_user[row["user_id"]].append(
            {
                "item": asin_to_int[asin],
                "asin": asin,
                "ts": int(row.get("timestamp", 0)),
                "position": int(row.get("position", 0)),
            }
        )
    for rows in raw_rec_by_user.values():
        rows.sort(key=lambda x: (x["ts"], x["position"], x["item"]))

    search_source_rows = [row for row in read_jsonl(data_dir / "search_interactions.jsonl") if row["item_id"] in asin_to_int]
    users = sorted(set(raw_rec_by_user) | {row["user_id"] for row in search_source_rows})
    user_to_int = {user: idx + 1 for idx, user in enumerate(users)}
    for user, rows in raw_rec_by_user.items():
        for row in rows:
            row["user_int"] = user_to_int[user]

    rec_ts_by_user = {}
    for user, rows in raw_rec_by_user.items():
        rec_ts_by_user[user] = [row["ts"] for row in rows]

    src_session_vocab = {}
    src_sessions_by_user = defaultdict(list)
    src_train_raw = []
    src_test = []
    src_by_user = defaultdict(list)
    session_id = 1
    if args.search_timestamp_policy == "source":
        missing_timestamps = [row for row in search_source_rows if "timestamp" not in row]
        if missing_timestamps:
            raise ValueError(
                f"source search timestamps requested but {len(missing_timestamps)} rows lack timestamp"
            )
        search_source_rows.sort(
            key=lambda row: (row["user_id"], int(row["timestamp"]), row["interaction_id"])
        )
    for order, row in enumerate(search_source_rows):
        user = row["user_id"]
        asin = row["item_id"]
        user_int = user_to_int[user]
        item_int = asin_to_int[asin]
        query_id = int(row["query_id"])
        ts = int(row["timestamp"]) if args.search_timestamp_policy == "source" else 10**12 + order
        rec_his_count = bisect.bisect_left(rec_ts_by_user.get(user, []), ts)
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
        }
        if row.get("split") == "test":
            src_test.append(out)
        else:
            src_train_raw.append(out)
        src_by_user[user].append(out)
        session_id += 1

    if args.strict_gensar_split:
        search_ts_by_user = {
            user: [int(row["ts"]) for row in rows]
            for user, rows in src_by_user.items()
        }
        rec_candidates_by_user = defaultdict(list)
        for user, rows in raw_rec_by_user.items():
            src_timestamps = search_ts_by_user.get(user, [])
            for pos in range(1, len(rows)):
                src_his = bisect.bisect_left(src_timestamps, int(rows[pos]["ts"]))
                rec_candidates_by_user[user].append(
                    {
                        "user_id": rows[pos]["user_int"],
                        "item_id": rows[pos]["item"],
                        "ts": rows[pos]["ts"],
                        "rec_his": pos,
                        "src_session_his": src_his,
                        "src_his": src_his,
                        "all_his": pos + src_his,
                    }
                )
        rec_train, rec_val, rec_test = split_strict_by_user(rec_candidates_by_user)
        src_train, src_val, src_test = split_strict_by_user(src_by_user)
        rec_val, rec_test = pair_cap_rows(
            rec_val, rec_test, args.paired_eval_users, args.seed
        )
        src_val, src_test = pair_cap_rows(
            src_val, src_test, args.paired_eval_users, args.seed + 1
        )
    else:
        rec_train, rec_val, rec_test = split_rec_by_user(raw_rec_by_user)
        if args.search_split_policy == "per_user_chrono":
            src_train, src_val, src_test = split_search_by_user(src_by_user)
        else:
            src_train, src_val = split_search_train_val(src_train_raw, args.valid_ratio, args.seed)
        rec_val = cap_rows(rec_val, args.max_val_per_task, args.seed)
        rec_test = cap_rows(rec_test, args.max_test_per_task, args.seed)
        src_val = cap_rows(src_val, args.max_val_per_task, args.seed)
        src_test = cap_rows(src_test, args.max_test_per_task, args.seed)
    rec_val = attach_negs(rec_val, all_int_items, rng, args.max_negs)
    rec_test = attach_negs(rec_test, all_int_items, rng, args.max_negs)
    src_val = attach_negs(src_val, all_int_items, rng, args.max_negs)
    src_test = attach_negs(src_test, all_int_items, rng, args.max_negs)

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
        ("rec_test", rec_test),
        ("src_train", src_train),
        ("src_val", src_val),
        ("src_test", src_test),
    ]:
        (out_dir / "dataset").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_pickle(out_dir / "dataset" / f"{name}.pkl")

    sid_files = write_sid_files(out_dir, parse_sid_specs(args.sid_spec), asin_to_int) if args.sid_spec else {}
    write_json(
        out_dir / "id_maps.json",
        {"asin_to_int": asin_to_int, "int_to_asin": int_to_asin, "user_to_int": user_to_int},
    )
    stats = {
        "items": len(all_int_items),
        "users": len(user_vocab),
        "queries": len(query_vocab),
        "src_sessions": len(src_session_vocab),
        "rec_rows": {"train": len(rec_train), "valid": len(rec_val), "test": len(rec_test)},
        "src_rows": {"train": len(src_train), "valid": len(src_val), "test": len(src_test)},
        "max_negs": args.max_negs,
        "sid_files": sid_files,
        "notes": [
            "Train rows do not store neg_items; GenSAR only uses neg_items in valid/test.",
            f"Search timestamp policy: {args.search_timestamp_policy}.",
            f"Search split policy: {args.search_split_policy}.",
            f"Strict GenSAR split: {args.strict_gensar_split}.",
            f"Paired evaluation users per task: {args.paired_eval_users}.",
            "Rec split is per-user chronological when at least three next-item rows exist.",
        ],
    }
    write_json(out_dir / "stats.json", stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
