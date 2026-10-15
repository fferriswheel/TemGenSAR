#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def load(data_dir, name):
    return pd.read_pickle(data_dir / "dataset" / f"{name}.pkl")


def date_summary(frame):
    dates = pd.to_datetime(frame["ts"], unit="s", utc=True)
    quantiles = dates.quantile([0, 0.1, 0.5, 0.9, 1])
    return {
        key: value.strftime("%Y-%m-%d")
        for key, value in zip(("min", "p10", "p50", "p90", "max"), quantiles)
    }


def check_task(data_dir, task):
    train = load(data_dir, f"{task}_train")
    val = load(data_dir, f"{task}_val")
    test = load(data_dir, f"{task}_test")

    assert val["user_id"].is_unique
    assert test["user_id"].is_unique
    assert set(val["user_id"]) == set(test["user_id"])
    for frame in (train, val, test):
        assert (frame["rec_his"] > 0).all()
        assert (frame["src_session_his"] > 0).all()
        assert (frame["all_his"] == frame["rec_his"] + frame["src_his"]).all()

    paired = val[["user_id", "ts"]].merge(
        test[["user_id", "ts"]], on="user_id", suffixes=("_val", "_test")
    )
    assert (paired["ts_val"] <= paired["ts_test"]).all()

    train_max = train.groupby("user_id")["ts"].max()
    users_with_train = val["user_id"].isin(train_max.index)
    assert users_with_train.all()
    val_with_train = val.loc[users_with_train, ["user_id", "ts"]]
    assert (
        val_with_train["ts"].to_numpy()
        >= val_with_train["user_id"].map(train_max).to_numpy()
    ).all()

    for frame in (val, test):
        assert frame["neg_items"].map(len).eq(999).all()
        assert all(
            int(row.item_id) not in row.neg_items for row in frame.itertuples()
        )

    train_frequency = Counter(int(item) for item in train["item_id"])
    result = {
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "paired_users": len(paired),
        "val_before_test_fraction": float((paired["ts_val"] < paired["ts_test"]).mean()),
        "equal_timestamp_fraction": float((paired["ts_val"] == paired["ts_test"]).mean()),
        "median_val_test_gap_days": float(
            ((paired["ts_test"] - paired["ts_val"]) / 86400).median()
        ),
        "dates": {"val": date_summary(val), "test": date_summary(test)},
        "target_train_frequency": {},
    }
    for split, frame in (("val", val), ("test", test)):
        frequency = pd.Series([train_frequency[int(item)] for item in frame["item_id"]])
        result["target_train_frequency"][split] = {
            "p50": float(frequency.median()),
            "mean": float(frequency.mean()),
            "zero_fraction": float((frequency == 0).mean()),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "data_dir": str(args.data_dir),
        "status": "passed",
        "rec": check_task(args.data_dir, "rec"),
        "src": check_task(args.data_dir, "src"),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
