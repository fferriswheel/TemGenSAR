#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stats = json.loads((args.data_dir / "stats.json").read_text())
    user_vocab = load_pickle(args.data_dir / "vocab" / "user_vocab.pkl")
    session_vocab = load_pickle(args.data_dir / "vocab" / "src_session_vocab.pkl")
    session_ts = {key: int(value["time_list"][0]) for key, value in session_vocab.items()}
    report = {"status": "passed", "windows": {}}
    previous_end = None
    for window, meta in stats["windows"].items():
        start, end = int(meta["start"]), int(meta["end_exclusive"])
        assert previous_end is None or start == previous_end
        previous_end = end
        window_report = {}
        for task in ("rec", "src"):
            update = pd.read_pickle(args.data_dir / "windows" / window / f"{task}_update.pkl")
            evaluation = pd.read_pickle(args.data_dir / "windows" / window / f"{task}_eval.pkl")
            assert len(update) == meta[f"{task}_update"]
            assert len(evaluation) == meta[f"{task}_eval"]
            for frame in (update, evaluation):
                assert frame["ts"].between(start, end - 1).all()
                assert (frame["rec_his"] > 0).all()
                assert (frame["src_session_his"] > 0).all()
                assert (frame["all_his"] == frame["rec_his"] + frame["src_his"]).all()
                for row in frame.itertuples():
                    user = user_vocab[int(row.user_id)]
                    rec_history_ts = user["rec_his_ts"][: int(row.rec_his)]
                    src_ids = user["src_session_his"][: int(row.src_session_his)]
                    assert rec_history_ts and max(rec_history_ts) < int(row.ts)
                    assert src_ids and max(session_ts[key] for key in src_ids) < int(row.ts)
            assert evaluation["user_id"].is_unique
            assert evaluation["neg_items"].map(len).eq(stats["max_negs"]).all()
            assert all(int(row.item_id) not in row.neg_items for row in evaluation.itertuples())
            update_keys = set(zip(update.user_id, update.item_id, update.ts))
            assert all((row.user_id, row.item_id, row.ts) in update_keys for row in evaluation.itertuples())
            window_report[task] = {
                "update_rows": len(update), "eval_rows": len(evaluation),
                "update_users": int(update["user_id"].nunique()),
                "eval_users": int(evaluation["user_id"].nunique()),
                "update_items": int(update["item_id"].nunique()),
                "eval_items": int(evaluation["item_id"].nunique()),
            }
        report["windows"][window] = window_report
    for task in ("rec", "src"):
        train = pd.read_pickle(args.data_dir / "dataset" / f"{task}_train.pkl")
        val = pd.read_pickle(args.data_dir / "dataset" / f"{task}_val.pkl")
        assert (train["ts"] < stats["t0"]["end_exclusive"]).all()
        assert (val["ts"] < stats["t0"]["end_exclusive"]).all()
        assert len(train) == stats["t0"][f"{task}_train"]
        assert len(val) == stats["t0"][f"{task}_val"]
        assert val["user_id"].is_unique
        train_max = train.groupby("user_id")["ts"].max()
        assert all(int(row.ts) >= int(train_max.get(row.user_id, 0)) for row in val.itertuples())
        assert val["neg_items"].map(len).eq(stats["max_negs"]).all()
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
