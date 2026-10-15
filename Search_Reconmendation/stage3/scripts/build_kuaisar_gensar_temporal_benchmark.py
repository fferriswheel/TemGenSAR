#!/usr/bin/env python3
"""Materialize a leakage-auditable GenSAR benchmark from strict KuaiSAR Parquet."""

from __future__ import annotations

import argparse
import ast
import bisect
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initialization-end-window", type=int, default=2)
    parser.add_argument("--max-val-users", type=int, default=5000)
    parser.add_argument("--num-negatives", type=int, default=99)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def dump_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=4)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def token_text(value, max_tokens: int | None = None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            value = [value]
    if not isinstance(value, (list, tuple)):
        value = [value]
    if max_tokens is not None:
        value = value[:max_tokens]
    return " ".join(f"tok{token}" for token in value)


def category_text(row) -> str:
    values = []
    for level in ("first", "second", "third", "fourth"):
        value = row.get(f"{level}_level_category_name_en")
        if value is not None and not pd.isna(value) and str(value).lower() != "empty":
            values.append(str(value))
    item_type = row.get("item_type")
    if item_type is not None and not pd.isna(item_type):
        values.append(str(item_type))
    return " ".join(values)


def hold_out_latest(rows: list[dict], max_users: int, seed: int) -> tuple[list[dict], list[dict]]:
    by_user = defaultdict(list)
    for row in rows:
        by_user[int(row["user_id"])].append(row)
    eligible = sorted(user for user, values in by_user.items() if len(values) >= 2)
    if max_users > 0 and len(eligible) > max_users:
        eligible = sorted(random.Random(seed).sample(eligible, max_users))
    held_users = set(eligible)
    train, valid = [], []
    for user, values in by_user.items():
        values.sort(key=lambda row: (int(row["ts"]), int(row["item_id"])))
        if user in held_users:
            train.extend(dict(row) for row in values[:-1])
            valid.append(dict(values[-1]))
        else:
            train.extend(dict(row) for row in values)
    return train, valid


def sample_prior_unseen_negatives(
    row: dict,
    user_vocab: dict,
    src_session_vocab: dict,
    num_items: int,
    count: int,
    rng: random.Random,
) -> list[int]:
    user = int(row["user_id"])
    history = user_vocab[user]
    excluded = set(history["rec_his"][: int(row["rec_his"])])
    for session_id in history["src_session_his"][: int(row["src_session_his"])]:
        excluded.update(src_session_vocab[session_id]["pos_items"])
    excluded.add(int(row["item_id"]))
    available = num_items - len(excluded)
    if available < count:
        raise ValueError(f"Only {available} negatives available for user {user}")
    negatives = set()
    while len(negatives) < count:
        candidate = rng.randint(1, num_items)
        if candidate not in excluded:
            negatives.add(candidate)
    result = list(negatives)
    rng.shuffle(result)
    return result


def attach_negatives(
    rows: list[dict],
    user_vocab: dict,
    src_session_vocab: dict,
    num_items: int,
    count: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    output = []
    for row in rows:
        value = dict(row)
        value["neg_items"] = sample_prior_unseen_negatives(
            value, user_vocab, src_session_vocab, num_items, count, rng
        )
        output.append(value)
    return output


def main() -> None:
    args = parse_args()
    source = args.source_dir
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    catalog = pd.read_parquet(source / "catalog.parquet").sort_values("catalog_index")
    items = pd.read_parquet(source / "items.parquet").sort_values("catalog_index")
    queries = pd.read_parquet(source / "queries.parquet").sort_values("query_index")
    windows = pd.read_csv(source / "windows.csv")
    rec = pd.read_parquet(source / "rec_positive_events.parquet")
    search = pd.read_parquet(source / "search_positive_events.parquet")
    rec_targets = pd.read_parquet(source / "rec_eval_targets.parquet")
    search_targets = pd.read_parquet(source / "search_eval_targets.parquet")

    item_ids = catalog["item_id"].astype(str).tolist()
    item_to_int = {item_id: idx + 1 for idx, item_id in enumerate(item_ids)}
    num_items = len(item_ids)
    src_item_set = [
        int(row.catalog_index)
        for row in catalog.itertuples()
        if int(row.search_support_w0_w2) > 0
    ]
    rec_item_set = [
        int(row.catalog_index)
        for row in catalog.itertuples()
        if int(row.rec_support_w0_w2) > 0
    ]
    catalog_index_to_int = {
        int(row.catalog_index): idx + 1 for idx, row in enumerate(catalog.itertuples())
    }

    users = sorted(
        set(rec["user_id"].astype(int)) | set(search["user_id"].astype(int))
    )
    user_to_int = {user: idx + 1 for idx, user in enumerate(users)}
    query_text_by_id = {
        int(row.query_index): token_text(row.keyword, max_tokens=32)
        for row in queries.itertuples()
    }

    rec_by_user = defaultdict(list)
    for row in rec.sort_values(["user_id", "timestamp", "item_index"]).itertuples():
        rec_by_user[int(row.user_id)].append(
            {
                "item": catalog_index_to_int[int(row.item_index)],
                "ts": int(row.timestamp),
                "window": int(row.window_id),
            }
        )

    session_key_columns = ["user_id", "search_session_id", "timestamp", "query_index", "window_id"]
    grouped_search = (
        search.groupby(session_key_columns, sort=True)["item_index"]
        .apply(lambda values: sorted(set(int(value) for value in values)))
        .reset_index(name="item_indices")
        .sort_values(["user_id", "timestamp", "search_session_id"])
    )
    src_session_vocab = {}
    src_sessions_by_user = defaultdict(list)
    src_session_ts_by_user = defaultdict(list)
    session_rows = []
    for session_id, row in enumerate(grouped_search.itertuples(), start=1):
        raw_user = int(row.user_id)
        positive_items = [catalog_index_to_int[value] for value in row.item_indices]
        session = {
            "session_id": session_id,
            "raw_session_id": int(row.search_session_id),
            "raw_user": raw_user,
            "ts": int(row.timestamp),
            "window": int(row.window_id),
            "query_id": int(row.query_index),
            "pos_items": positive_items,
        }
        session_rows.append(session)
        src_sessions_by_user[raw_user].append(session_id)
        src_session_ts_by_user[raw_user].append(session["ts"])
        src_session_vocab[session_id] = {
            "query_id": session["query_id"],
            "query": query_text_by_id[session["query_id"]],
            "pos_items": positive_items,
            "time_list": [session["ts"]] * len(positive_items),
        }

    rec_ts_by_user = {
        user: [row["ts"] for row in values] for user, values in rec_by_user.items()
    }
    rec_candidates = []
    for raw_user, values in rec_by_user.items():
        user_int = user_to_int[raw_user]
        search_ts = src_session_ts_by_user.get(raw_user, [])
        for row in values:
            rec_his = bisect.bisect_left(rec_ts_by_user[raw_user], row["ts"])
            src_his = bisect.bisect_left(search_ts, row["ts"])
            if rec_his > 0 and src_his > 0:
                rec_candidates.append(
                    {
                        "user_id": user_int,
                        "item_id": row["item"],
                        "ts": row["ts"],
                        "window_id": row["window"],
                        "rec_his": rec_his,
                        "src_session_his": src_his,
                        "src_his": src_his,
                        "all_his": rec_his + src_his,
                    }
                )

    src_candidates = []
    for session in session_rows:
        raw_user = session["raw_user"]
        user_int = user_to_int[raw_user]
        rec_his = bisect.bisect_left(rec_ts_by_user.get(raw_user, []), session["ts"])
        src_his = bisect.bisect_left(src_session_ts_by_user[raw_user], session["ts"])
        if rec_his <= 0 or src_his <= 0:
            continue
        for item in session["pos_items"]:
            src_candidates.append(
                {
                    "user_id": user_int,
                    "item_id": item,
                    "search_session_id": session["session_id"],
                    "raw_search_session_id": session["raw_session_id"],
                    "query_id": session["query_id"],
                    "ts": session["ts"],
                    "window_id": session["window"],
                    "rec_his": rec_his,
                    "src_session_his": src_his,
                    "src_his": src_his,
                    "all_his": rec_his + src_his,
                }
            )

    user_vocab = {}
    for raw_user, user_int in user_to_int.items():
        user_vocab[user_int] = {
            "rec_his": [row["item"] for row in rec_by_user.get(raw_user, [])],
            "rec_his_ts": rec_ts_by_user.get(raw_user, []),
            "src_session_his": src_sessions_by_user.get(raw_user, []),
        }

    item_by_index = {int(row.catalog_index): row for row in items.itertuples()}
    item_vocab = [{"title": "", "description": ""}]
    for index in catalog["catalog_index"].astype(int):
        row = item_by_index.get(index)
        if row is None:
            item_vocab.append({"title": "", "description": ""})
        else:
            values = row._asdict()
            item_vocab.append(
                {
                    "title": token_text(values.get("caption_tokens"), max_tokens=64),
                    "description": category_text(values),
                }
            )

    max_query_id = int(queries["query_index"].max())
    query_vocab = [{"query": ""} for _ in range(max_query_id + 1)]
    for row in queries.itertuples():
        query_vocab[int(row.query_index)] = {"query": token_text(row.keyword, max_tokens=32)}

    for name, value in (
        ("user_vocab", user_vocab),
        ("item_vocab", item_vocab),
        ("query_vocab", query_vocab),
        ("src_session_vocab", src_session_vocab),
        ("src_item_set", src_item_set),
        ("rec_item_set", rec_item_set),
    ):
        dump_pickle(output / "vocab" / f"{name}.pkl", value)

    init_rec = [row for row in rec_candidates if row["window_id"] <= args.initialization_end_window]
    init_src = [row for row in src_candidates if row["window_id"] <= args.initialization_end_window]
    rec_train, rec_val = hold_out_latest(init_rec, args.max_val_users, args.seed)
    src_train, src_val = hold_out_latest(init_src, args.max_val_users, args.seed + 1)
    rec_val = attach_negatives(
        rec_val, user_vocab, src_session_vocab, num_items, args.num_negatives, args.seed + 10
    )
    src_val = attach_negatives(
        src_val, user_vocab, src_session_vocab, num_items, args.num_negatives, args.seed + 11
    )
    for name, rows in (
        ("rec_train", rec_train),
        ("src_train", src_train),
        ("rec_val", rec_val),
        ("src_val", src_val),
    ):
        dump_pickle(output / "dataset" / f"{name}.pkl", pd.DataFrame(rows))

    eval_stats = {}
    for window in (4, 5):
        window_start = int(windows.loc[windows["window_id"] == window, "start_ms"].iloc[0])
        rec_target_keys = {
            (user_to_int[int(target.user_id)], int(target.item_index), int(target.timestamp))
            for target in rec_targets[rec_targets["window_id"] == window].itertuples()
        }
        rec_eval = []
        for row in rec_candidates:
            key = (int(row["user_id"]), int(row["item_id"]), int(row["ts"]))
            if key not in rec_target_keys:
                continue
            raw_user = users[int(row["user_id"]) - 1]
            value = dict(row)
            value["rec_his"] = bisect.bisect_left(rec_ts_by_user.get(raw_user, []), window_start)
            value["src_session_his"] = bisect.bisect_left(
                src_session_ts_by_user.get(raw_user, []), window_start
            )
            value["src_his"] = value["src_session_his"]
            value["all_his"] = value["rec_his"] + value["src_session_his"]
            if value["rec_his"] > 0 and value["src_session_his"] > 0:
                rec_eval.append(value)
        src_target_keys = {
            (
                user_to_int[int(target.user_id)],
                int(target.item_index),
                int(target.timestamp),
                int(target.search_session_id),
            )
            for target in search_targets[search_targets["window_id"] == window].itertuples()
        }
        src_eval = []
        for row in src_candidates:
            key = (
                int(row["user_id"]),
                int(row["item_id"]),
                int(row["ts"]),
                int(row["raw_search_session_id"]),
            )
            if key not in src_target_keys:
                continue
            raw_user = users[int(row["user_id"]) - 1]
            value = dict(row)
            value["rec_his"] = bisect.bisect_left(rec_ts_by_user.get(raw_user, []), window_start)
            value["src_session_his"] = bisect.bisect_left(
                src_session_ts_by_user.get(raw_user, []), window_start
            )
            value["src_his"] = value["src_session_his"]
            value["all_his"] = value["rec_his"] + value["src_session_his"]
            if value["rec_his"] > 0 and value["src_session_his"] > 0:
                src_eval.append(value)
        rec_eval = attach_negatives(
            rec_eval, user_vocab, src_session_vocab, num_items, args.num_negatives,
            args.seed + 100 + window * 2,
        )
        src_eval = attach_negatives(
            src_eval, user_vocab, src_session_vocab, num_items, args.num_negatives,
            args.seed + 101 + window * 2,
        )
        dump_pickle(output / "windows" / f"W{window}" / "rec_eval.pkl", pd.DataFrame(rec_eval))
        dump_pickle(output / "windows" / f"W{window}" / "src_eval.pkl", pd.DataFrame(src_eval))
        eval_stats[f"W{window}"] = {"rec_eval": len(rec_eval), "src_eval": len(src_eval)}

    update_stats = {}
    for window in (3, 4):
        rec_update = [row for row in rec_candidates if row["window_id"] == window]
        src_update = [row for row in src_candidates if row["window_id"] == window]
        dump_pickle(output / "windows" / f"W{window}" / "rec_update.pkl", pd.DataFrame(rec_update))
        dump_pickle(output / "windows" / f"W{window}" / "src_update.pkl", pd.DataFrame(src_update))
        update_stats[f"W{window}"] = {
            "rec_update": len(rec_update), "src_update": len(src_update)
        }

    item_id_path = output / "item_ids.jsonl"
    item_id_path.write_text(
        "".join(json.dumps({"item_id": item_id}) + "\n" for item_id in item_ids),
        encoding="utf-8",
    )
    write_json(
        output / "id_maps.json",
        {
            "item_to_int": item_to_int,
            "int_to_item": {str(value): key for key, value in item_to_int.items()},
            "user_to_int": {str(key): value for key, value in user_to_int.items()},
        },
    )
    summary = {
        "name": "KuaiSAR_Strict_Balanced_30K_GenSAR_Temporal",
        "catalog_items": num_items,
        "src_task_items_w0_w2": len(src_item_set),
        "rec_task_items_w0_w2": len(rec_item_set),
        "task_item_union": len(set(src_item_set) | set(rec_item_set)),
        "catalog_policy": "W0-W2 Top-15K Rec union Top-15K Search, frozen",
        "initialization_windows": [0, args.initialization_end_window],
        "history_policy": "strictly before target timestamp; W4/W5 first target per user",
        "negative_policy": f"{args.num_negatives} fixed random items unseen in prior cross-task history",
        "search_session_policy": "group positive items by original search_session_id",
        "t0": {
            "rec_train": len(rec_train),
            "src_train": len(src_train),
            "rec_val": len(rec_val),
            "src_val": len(src_val),
        },
        "updates": update_stats,
        "evaluation": eval_stats,
        "future_behavior_used_for_representation": False,
    }
    write_json(output / "stats.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
