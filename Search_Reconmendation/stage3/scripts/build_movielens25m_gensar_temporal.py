#!/usr/bin/env python3
"""Convert the MovieLens temporal pilot into GenSAR-compatible pickles."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--max-train-rows-per-task", type=int, default=1_000_000)
    parser.add_argument("--max-val-rows", type=int, default=5_000)
    parser.add_argument("--max-eval-rows", type=int, default=5_000)
    parser.add_argument("--num-negatives", type=int, default=99)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def dump_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=4)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def stable_sample(rows: list[dict], count: int, seed: int) -> list[dict]:
    if count <= 0 or len(rows) <= count:
        return rows
    indices = sorted(random.Random(seed).sample(range(len(rows)), count))
    return [rows[index] for index in indices]


def latest_per_user_split(rows: list[dict], max_valid: int, seed: int):
    by_user = defaultdict(list)
    for row in rows:
        by_user[int(row["user_id"])].append(row)
    eligible = sorted(user for user, values in by_user.items() if len(values) >= 2)
    if max_valid > 0 and len(eligible) > max_valid:
        eligible = sorted(random.Random(seed).sample(eligible, max_valid))
    held = set(eligible)
    train, valid = [], []
    for user, values in by_user.items():
        values.sort(key=lambda row: (int(row["ts"]), int(row["item_id"])))
        if user in held:
            train.extend(values[:-1])
            valid.append(dict(values[-1]))
        else:
            train.extend(values)
    return train, valid


def sample_negatives(
    row: dict,
    user_vocab: dict,
    src_session_vocab: dict,
    num_items: int,
    count: int,
    rng: random.Random,
) -> list[int]:
    history = user_vocab[int(row["user_id"])]
    excluded = set(history["rec_his"][: int(row["rec_his"])])
    for session_id in history["src_session_his"][: int(row["src_session_his"])]:
        excluded.update(src_session_vocab[session_id]["pos_items"])
    excluded.add(int(row["item_id"]))
    if num_items - len(excluded) < count:
        raise ValueError("not enough unseen items for fixed negative sampling")
    values = set()
    while len(values) < count:
        candidate = rng.randint(1, num_items)
        if candidate not in excluded:
            values.add(candidate)
    result = list(values)
    rng.shuffle(result)
    return result


def attach_negatives(rows, user_vocab, src_session_vocab, num_items, count, seed):
    rng = random.Random(seed)
    output = []
    for row in rows:
        value = dict(row)
        value["neg_items"] = sample_negatives(
            value, user_vocab, src_session_vocab, num_items, count, rng
        )
        output.append(value)
    return output


def description(row) -> str:
    def values(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        try:
            return [str(item) for item in value]
        except TypeError:
            return []

    fields = []
    genres = values(row.genres)
    user_tags = values(row.user_tags)
    genome_tags = values(row.genome_tags)
    if genres:
        fields.append("Genres: " + ", ".join(genres))
    if user_tags:
        fields.append("User tags: " + ", ".join(user_tags[:20]))
    if genome_tags:
        fields.append("Attributes: " + ", ".join(genome_tags[:20]))
    if pd.notna(row.release_year):
        fields.append(f"Release year: {int(row.release_year)}")
    return ". ".join(fields)


def main() -> None:
    args = parse_args()
    source = args.source_dir
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty")
    output.mkdir(parents=True, exist_ok=True)

    catalog = pd.read_parquet(source / "catalog.parquet").sort_values("catalog_index")
    items = pd.read_parquet(source / "items.parquet").sort_values("catalog_index")
    queries = pd.read_parquet(source / "queries.parquet").sort_values("query_index")
    query_bank = pd.read_parquet(source / "query_bank.parquet")
    rec = pd.read_parquet(source / "rec_positive_events.parquet").sort_values(
        ["user_id", "timestamp", "item_id"]
    )
    search = pd.read_parquet(source / "search_positive_events.parquet").sort_values(
        ["user_id", "timestamp", "item_id"]
    )
    rec_targets = pd.read_parquet(source / "rec_eval_targets.parquet")
    windows = pd.read_csv(source / "windows.csv")

    item_ids = catalog["item_id"].astype(str).tolist()
    item_to_int = {item_id: index + 1 for index, item_id in enumerate(item_ids)}
    catalog_to_int = {
        int(row.catalog_index): index + 1 for index, row in enumerate(catalog.itertuples())
    }
    num_items = len(item_ids)
    users = sorted(set(rec["user_id"].astype(int)) | set(search["user_id"].astype(int)))
    user_to_int = {user: index + 1 for index, user in enumerate(users)}

    rec_by_user = defaultdict(list)
    for row in rec.itertuples(index=False):
        rec_by_user[int(row.user_id)].append(
            {
                "item": catalog_to_int[int(row.catalog_index)],
                "ts": int(row.timestamp),
                "window": int(row.window_id),
            }
        )
    rec_ts_by_user = {
        user: [row["ts"] for row in values] for user, values in rec_by_user.items()
    }

    query_text = {
        int(row.query_index): str(row.query) for row in queries.itertuples(index=False)
    }
    src_session_vocab = {}
    src_sessions_by_user = defaultdict(list)
    src_ts_by_user = defaultdict(list)
    train_search_rows = []
    for session_id, row in enumerate(search.itertuples(index=False), start=1):
        raw_user = int(row.user_id)
        user_id = user_to_int[raw_user]
        item_id = catalog_to_int[int(row.item_index)]
        timestamp = int(row.timestamp)
        query_id = int(row.query_index)
        rec_his = bisect.bisect_left(rec_ts_by_user[raw_user], timestamp)
        src_his = len(src_sessions_by_user[raw_user])
        src_session_vocab[session_id] = {
            "query_id": query_id,
            "query": query_text[query_id],
            "pos_items": [item_id],
            "time_list": [timestamp],
        }
        if rec_his > 0 and src_his > 0:
            train_search_rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "search_session_id": session_id,
                    "query_id": query_id,
                    "ts": timestamp,
                    "window_id": 0,
                    "rec_his": rec_his,
                    "src_session_his": src_his,
                    "src_his": src_his,
                    "all_his": rec_his + src_his,
                }
            )
        src_sessions_by_user[raw_user].append(session_id)
        src_ts_by_user[raw_user].append(timestamp)

    rec_candidates_t0 = []
    for raw_user, values in rec_by_user.items():
        if raw_user not in user_to_int:
            continue
        user_id = user_to_int[raw_user]
        src_timestamps = src_ts_by_user.get(raw_user, [])
        for position, row in enumerate(values):
            if row["window"] != 0:
                break
            src_his = bisect.bisect_left(src_timestamps, row["ts"])
            if position > 0 and src_his > 0:
                rec_candidates_t0.append(
                    {
                        "user_id": user_id,
                        "item_id": row["item"],
                        "ts": row["ts"],
                        "window_id": 0,
                        "rec_his": position,
                        "src_session_his": src_his,
                        "src_his": src_his,
                        "all_his": position + src_his,
                    }
                )

    user_vocab = {
        user_to_int[raw_user]: {
            "rec_his": [row["item"] for row in rec_by_user.get(raw_user, [])],
            "rec_his_ts": rec_ts_by_user.get(raw_user, []),
            "src_session_his": src_sessions_by_user.get(raw_user, []),
        }
        for raw_user in users
    }

    item_vocab = [{"title": "", "description": ""}]
    for row in items.itertuples(index=False):
        item_vocab.append({"title": str(row.title), "description": description(row)})
    max_query = int(queries["query_index"].max())
    query_vocab = [{"query": ""} for _ in range(max_query + 1)]
    for query_id, value in query_text.items():
        query_vocab[query_id] = {"query": value}

    rec_train_all, rec_val = latest_per_user_split(
        rec_candidates_t0, args.max_val_rows, args.seed
    )
    src_train_all, src_val_base = latest_per_user_split(
        train_search_rows, args.max_val_rows, args.seed + 1
    )
    rec_train = stable_sample(
        rec_train_all, args.max_train_rows_per_task, args.seed + 2
    )
    src_train = stable_sample(
        src_train_all, args.max_train_rows_per_task, args.seed + 3
    )

    # Replace held-out Search queries with fixed validation/test query-bank entries.
    bank_by_item_split = defaultdict(list)
    for row in query_bank.itertuples(index=False):
        bank_by_item_split[(catalog_to_int[int(row.catalog_index)], str(row.split))].append(
            int(row.query_index)
        )

    def materialize_heldout_search(base_rows: list[dict], split: str, seed: int):
        result = []
        rng = random.Random(seed)
        next_session = max(src_session_vocab, default=0) + 1
        for row in base_rows:
            choices = bank_by_item_split[(int(row["item_id"]), split)]
            if not choices:
                continue
            value = dict(row)
            query_id = choices[rng.randrange(len(choices))]
            session_id = next_session
            next_session += 1
            src_session_vocab[session_id] = {
                "query_id": query_id,
                "query": query_text[query_id],
                "pos_items": [int(value["item_id"])],
                "time_list": [int(value["ts"])],
            }
            value["query_id"] = query_id
            value["search_session_id"] = session_id
            result.append(value)
        return result

    src_val = materialize_heldout_search(src_val_base, "validation", args.seed + 4)
    src_test = materialize_heldout_search(src_val_base, "test", args.seed + 5)
    rec_val = attach_negatives(
        rec_val, user_vocab, src_session_vocab, num_items,
        args.num_negatives, args.seed + 10,
    )
    src_val = attach_negatives(
        src_val, user_vocab, src_session_vocab, num_items,
        args.num_negatives, args.seed + 11,
    )
    src_test = attach_negatives(
        src_test, user_vocab, src_session_vocab, num_items,
        args.num_negatives, args.seed + 12,
    )

    for name, value in [
        ("user_vocab", user_vocab),
        ("item_vocab", item_vocab),
        ("query_vocab", query_vocab),
        ("src_session_vocab", src_session_vocab),
        ("src_item_set", list(range(1, num_items + 1))),
        ("rec_item_set", list(range(1, num_items + 1))),
    ]:
        dump_pickle(output / "vocab" / f"{name}.pkl", value)
    for name, rows in [
        ("rec_train", rec_train),
        ("src_train", src_train),
        ("rec_val", rec_val),
        ("src_val", src_val),
        ("src_test", src_test),
    ]:
        dump_pickle(output / "dataset" / f"{name}.pkl", pd.DataFrame(rows))

    fixed_search_test = pd.DataFrame(src_test)
    evaluation = {}
    updates = {}
    update_rows_by_window = defaultdict(list)
    for raw_user, values in rec_by_user.items():
        src_his = len(src_sessions_by_user.get(raw_user, []))
        if src_his <= 0:
            continue
        user_id = user_to_int[raw_user]
        for position, row in enumerate(values):
            if row["window"] <= 0:
                continue
            update_rows_by_window[row["window"]].append(
                {
                    "user_id": user_id,
                    "item_id": row["item"],
                    "ts": row["ts"],
                    "window_id": row["window"],
                    "rec_his": position,
                    "src_session_his": src_his,
                    "src_his": src_his,
                    "all_his": position + src_his,
                }
            )
    target_lookup = {
        int(window): group.sort_values(["user_id", "timestamp", "item_id"])
        for window, group in rec_targets.groupby("window_id")
    }
    for window_id in range(1, 11):
        start_ts = int(windows.loc[windows["window_id"] == window_id, "start_ts"].iloc[0])
        candidates = []
        for row in target_lookup.get(window_id, pd.DataFrame()).itertuples(index=False):
            raw_user = int(row.user_id)
            if raw_user not in user_to_int or not src_sessions_by_user.get(raw_user):
                continue
            rec_his = bisect.bisect_left(rec_ts_by_user[raw_user], start_ts)
            src_his = bisect.bisect_left(src_ts_by_user[raw_user], start_ts)
            if rec_his < 5 or src_his <= 0:
                continue
            candidates.append(
                {
                    "user_id": user_to_int[raw_user],
                    "item_id": catalog_to_int[int(row.catalog_index)],
                    "ts": int(row.timestamp),
                    "window_id": window_id,
                    "rec_his": rec_his,
                    "src_session_his": src_his,
                    "src_his": src_his,
                    "all_his": rec_his + src_his,
                }
            )
        candidates = stable_sample(candidates, args.max_eval_rows, args.seed + 100 + window_id)
        rec_eval = attach_negatives(
            candidates, user_vocab, src_session_vocab, num_items,
            args.num_negatives, args.seed + 200 + window_id,
        )
        window_dir = output / "windows" / f"W{window_id}"
        dump_pickle(window_dir / "rec_eval.pkl", pd.DataFrame(rec_eval))
        dump_pickle(window_dir / "src_eval.pkl", fixed_search_test)

        update_rows = update_rows_by_window[window_id]
        dump_pickle(window_dir / "rec_update.pkl", pd.DataFrame(update_rows))
        dump_pickle(window_dir / "src_update.pkl", pd.DataFrame(src_train[:100_000]))
        evaluation[f"W{window_id}"] = {
            "rec_eval": len(rec_eval), "src_eval": len(fixed_search_test)
        }
        updates[f"W{window_id}"] = {"rec_update": len(update_rows)}

    (output / "item_ids.jsonl").write_text(
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
        "name": "MovieLens25M_18K_GenSAR_Temporal_v1",
        "catalog_items": num_items,
        "users": len(users),
        "query_vocab": len(query_vocab) - 1,
        "t0": {
            "rec_candidates_before_cap": len(rec_train_all),
            "src_candidates_before_cap": len(src_train_all),
            "rec_train": len(rec_train),
            "src_train": len(src_train),
            "rec_val": len(rec_val),
            "src_val": len(src_val),
            "fixed_src_test": len(src_test),
        },
        "updates": updates,
        "evaluation": evaluation,
        "search_control": "same held-out Search rows and negatives in W1-W10",
        "future_behavior_used_for_t0_representation": False,
    }
    write_json(output / "stats.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
