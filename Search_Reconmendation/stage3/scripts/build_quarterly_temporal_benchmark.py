#!/usr/bin/env python3
"""Build a leakage-auditable quarterly Search/Rec temporal benchmark."""

import argparse
import bisect
import json
import pickle
import random
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd


CALIBRATION = {f"C{idx + 1}": str(period) for idx, period in enumerate(pd.period_range("2010Q1", "2011Q4", freq="Q"))}
MONITOR = {f"W{idx + 1}": str(period) for idx, period in enumerate(pd.period_range("2012Q1", "2014Q2", freq="Q"))}


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dump_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=4)


def sample_negs(positive, all_items, rng, count):
    position = bisect.bisect_left(all_items, positive)
    compact = rng.sample(range(len(all_items) - 1), count)
    return [idx + 1 if idx < position else idx + 2 for idx in compact]


def cap_one_per_user(rows, max_users, seed):
    latest = {}
    for row in rows:
        key = int(row["user_id"])
        if key not in latest or (row["ts"], row["item_id"]) > (latest[key]["ts"], latest[key]["item_id"]):
            latest[key] = row
    users = sorted(latest)
    if max_users > 0 and len(users) > max_users:
        users = sorted(random.Random(seed).sample(users, max_users))
    return [dict(latest[user]) for user in users]


def leave_last_per_user(rows, max_val_users, seed):
    by_user = defaultdict(list)
    for row in rows:
        by_user[int(row["user_id"])].append(row)
    eligible = sorted(user for user, values in by_user.items() if len(values) >= 2)
    if max_val_users > 0 and len(eligible) > max_val_users:
        eligible = sorted(random.Random(seed).sample(eligible, max_val_users))
    val_users = set(eligible)
    train, val = [], []
    for user, values in by_user.items():
        values = sorted(values, key=lambda row: (row["ts"], row["item_id"]))
        if user in val_users:
            train.extend(dict(row) for row in values[:-1])
            val.append(dict(values[-1]))
        else:
            train.extend(dict(row) for row in values)
    return train, val


def period_bounds(period):
    value = pd.Period(period, freq="Q")
    return int(value.start_time.timestamp()), int((value + 1).start_time.timestamp())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-eval-users", type=int, default=5000)
    parser.add_argument("--max-negs", type=int, default=999)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in ("items.jsonl", "item_ids.jsonl", "queries.jsonl", "users.jsonl"):
        shutil.copy2(args.source_dir / name, args.output_dir / name)
    item_ids = [row["item_id"] for row in read_jsonl(args.source_dir / "item_ids.jsonl")]
    asin_to_int = {asin: idx + 1 for idx, asin in enumerate(item_ids)}
    int_to_asin = {idx: asin for asin, idx in asin_to_int.items()}
    query_rows = list(read_jsonl(args.source_dir / "queries.jsonl"))
    query_text = {int(row["query_id"]): str(row["query"]) for row in query_rows}

    rec_source = [row for row in read_jsonl(args.source_dir / "rec_interactions_chrono.jsonl") if row["item_id"] in asin_to_int]
    src_source = [row for row in read_jsonl(args.source_dir / "search_interactions.jsonl") if row["item_id"] in asin_to_int]
    users = sorted({row["user_id"] for row in rec_source} | {row["user_id"] for row in src_source})
    user_to_int = {user: idx + 1 for idx, user in enumerate(users)}

    rec_by_user = defaultdict(list)
    for row in rec_source:
        rec_by_user[row["user_id"]].append({
            "item": asin_to_int[row["item_id"]], "ts": int(row["timestamp"]),
            "position": int(row.get("position", 0)),
        })
    for rows in rec_by_user.values():
        rows.sort(key=lambda row: (row["ts"], row["position"], row["item"]))

    src_by_user = defaultdict(list)
    for row in src_source:
        src_by_user[row["user_id"]].append({
            "item": asin_to_int[row["item_id"]], "ts": int(row["timestamp"]),
            "query_id": int(row["query_id"]), "query": str(row.get("query") or query_text[int(row["query_id"])]),
            "interaction_id": str(row["interaction_id"]),
        })
    for rows in src_by_user.values():
        rows.sort(key=lambda row: (row["ts"], row["interaction_id"], row["item"]))

    # Session ids and histories are global and immutable. Counts on examples use
    # bisect_left, so an event never sees another event with the same timestamp.
    src_session_vocab = {}
    src_session_ids = defaultdict(list)
    src_session_ts = defaultdict(list)
    session_id = 1
    for user in users:
        for row in src_by_user.get(user, []):
            src_session_vocab[session_id] = {
                "query_id": row["query_id"], "query": row["query"],
                "pos_items": [row["item"]], "time_list": [row["ts"]],
            }
            row["session_id"] = session_id
            src_session_ids[user].append(session_id)
            src_session_ts[user].append(row["ts"])
            session_id += 1

    rec_ts = {user: [row["ts"] for row in rows] for user, rows in rec_by_user.items()}
    rec_candidates = []
    for user, rows in rec_by_user.items():
        user_int = user_to_int[user]
        for row in rows:
            rec_his = bisect.bisect_left(rec_ts[user], row["ts"])
            src_his = bisect.bisect_left(src_session_ts.get(user, []), row["ts"])
            if rec_his <= 0 or src_his <= 0:
                continue
            rec_candidates.append({
                "user_id": user_int, "item_id": row["item"], "ts": row["ts"],
                "rec_his": rec_his, "src_session_his": src_his,
                "src_his": src_his, "all_his": rec_his + src_his,
            })
    src_candidates = []
    for user, rows in src_by_user.items():
        user_int = user_to_int[user]
        for row in rows:
            rec_his = bisect.bisect_left(rec_ts.get(user, []), row["ts"])
            src_his = bisect.bisect_left(src_session_ts[user], row["ts"])
            if rec_his <= 0 or src_his <= 0:
                continue
            src_candidates.append({
                "user_id": user_int, "item_id": row["item"], "search_session_id": row["session_id"],
                "query_id": row["query_id"], "ts": row["ts"], "rec_his": rec_his,
                "src_session_his": src_his, "src_his": src_his, "all_his": rec_his + src_his,
            })

    item_rows = {row["item_id"]: row for row in read_jsonl(args.source_dir / "items.jsonl")}
    item_vocab = [{"title": "", "description": ""}] + [
        {"title": str(item_rows.get(asin, {}).get("title") or ""),
         "description": str(item_rows.get(asin, {}).get("description") or "")}
        for asin in item_ids
    ]
    max_query = max(query_text)
    query_vocab = [{"query": ""} for _ in range(max_query + 1)]
    for query_id, text in query_text.items():
        query_vocab[query_id] = {"query": text}
    user_vocab = {}
    for user, user_int in user_to_int.items():
        user_vocab[user_int] = {
            "rec_his": [row["item"] for row in rec_by_user.get(user, [])],
            "rec_his_ts": rec_ts.get(user, []),
            "src_session_his": src_session_ids.get(user, []),
        }
    for name, value in (
        ("user_vocab", user_vocab), ("item_vocab", item_vocab),
        ("query_vocab", query_vocab), ("src_session_vocab", src_session_vocab),
        ("src_item_set", list(range(1, len(item_ids) + 1))),
        ("rec_item_set", list(range(1, len(item_ids) + 1))),
    ):
        dump_pickle(args.output_dir / "vocab" / f"{name}.pkl", value)
    write_json(args.output_dir / "id_maps.json", {
        "asin_to_int": asin_to_int, "int_to_asin": int_to_asin, "user_to_int": user_to_int,
    })

    rng = random.Random(args.seed)
    all_items = list(range(1, len(item_ids) + 1))
    stats = {"calibration": CALIBRATION, "monitor": MONITOR, "windows": {}}
    for window, period in {**CALIBRATION, **MONITOR}.items():
        start, end = period_bounds(period)
        rec_update = [dict(row) for row in rec_candidates if start <= row["ts"] < end]
        src_update = [dict(row) for row in src_candidates if start <= row["ts"] < end]
        rec_eval = cap_one_per_user(rec_update, args.max_eval_users, args.seed + len(stats["windows"]) * 2)
        src_eval = cap_one_per_user(src_update, args.max_eval_users, args.seed + len(stats["windows"]) * 2 + 1)
        for row in rec_eval:
            row["neg_items"] = sample_negs(row["item_id"], all_items, rng, args.max_negs)
        for row in src_eval:
            row["neg_items"] = sample_negs(row["item_id"], all_items, rng, args.max_negs)
        for task, kind, rows in (
            ("rec", "update", rec_update), ("src", "update", src_update),
            ("rec", "eval", rec_eval), ("src", "eval", src_eval),
        ):
            dump_pickle(args.output_dir / "windows" / window / f"{task}_{kind}.pkl", pd.DataFrame(rows))
        stats["windows"][window] = {
            "period": period, "start": start, "end_exclusive": end,
            "rec_update": len(rec_update), "src_update": len(src_update),
            "rec_eval": len(rec_eval), "src_eval": len(src_eval),
            "rec_update_users": len({row["user_id"] for row in rec_update}),
            "src_update_users": len({row["user_id"] for row in src_update}),
        }
    t0_end = period_bounds("2011Q4")[1]
    t0_rec_all = [dict(row) for row in rec_candidates if row["ts"] < t0_end]
    t0_src_all = [dict(row) for row in src_candidates if row["ts"] < t0_end]
    t0_rec, t0_rec_val = leave_last_per_user(t0_rec_all, args.max_eval_users, args.seed + 1000)
    t0_src, t0_src_val = leave_last_per_user(t0_src_all, args.max_eval_users, args.seed + 1001)
    for row in t0_rec_val:
        row["neg_items"] = sample_negs(row["item_id"], all_items, rng, args.max_negs)
    for row in t0_src_val:
        row["neg_items"] = sample_negs(row["item_id"], all_items, rng, args.max_negs)
    for task, kind, rows in (
        ("rec", "train", t0_rec), ("src", "train", t0_src),
        ("rec", "val", t0_rec_val), ("src", "val", t0_src_val),
    ):
        dump_pickle(args.output_dir / "dataset" / f"{task}_{kind}.pkl", pd.DataFrame(rows))
    stats["t0"] = {
        "end_exclusive": t0_end, "rec_train": len(t0_rec), "src_train": len(t0_src),
        "rec_val": len(t0_rec_val), "src_val": len(t0_src_val),
        "split": "per-user latest eligible target held out for validation",
    }
    stats["items"] = len(item_ids)
    stats["users"] = len(users)
    stats["max_negs"] = args.max_negs
    write_json(args.output_dir / "stats.json", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
