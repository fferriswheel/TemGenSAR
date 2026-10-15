#!/usr/bin/env python3
"""Summarize R3 rolling metrics, SID churn/collision, and update costs."""

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


METHODS = ("Fixed", "Selective", "Random")
METRICS = ("rec_hit10", "rec_ndcg10", "search_hit10", "search_ndcg10")


def parse_results(log: Path):
    text = log.read_text(encoding="utf-8", errors="replace")
    values = [ast.literal_eval(x) for x in re.findall(r"Mean results:\s+(\{[^\n]+\})", text)]
    if len(values) != 4:
        raise ValueError(f"Expected four result blocks in {log}, got {len(values)}")
    val_rec, val_search, test_rec, test_search = values
    return {
        "val_rec_ndcg5": val_rec["ndcg@5"],
        "val_search_ndcg5": val_search["ndcg@5"],
        "rec_hit10": test_rec["hit@10"],
        "rec_ndcg10": test_rec["ndcg@10"],
        "search_hit10": test_search["hit@10"],
        "search_ndcg10": test_search["ndcg@10"],
    }


def parse_elapsed(log: Path):
    if not log.exists():
        return None
    text = log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\n]+)", text)
    if not matches:
        return None
    raw = matches[-1].strip()
    parts = [float(x) for x in raw.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(raw)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mapping_path(root: Path, update_window: int, method: str):
    if method == "Fixed":
        return root / "stage3/data/temporal_joint_r2a_mappings/J0_fixed_sid_mapping.json"
    if update_window == 1:
        name = "JB_joint_sid_mapping.json" if method == "Selective" else "JRandom_joint_sid_mapping.json"
        return root / "stage3/data/temporal_joint_r2a_mappings" / name
    name = "selective_joint.json" if method == "Selective" else "random_joint.json"
    return root / f"stage3/data/temporal_rolling_r3/W{update_window}/mappings" / name


def sid_stats(previous, current, key):
    changed = sum(previous[item][key] != current[item][key] for item in previous)
    counts = Counter(tuple(row[key]) for row in current.values())
    unique = len(counts)
    collided_items = sum(size for size in counts.values() if size > 1)
    return {
        "changed": changed,
        "changed_fraction": changed / len(current),
        "unique": unique,
        "duplicate_fraction": 1 - unique / len(current),
        "collided_item_fraction": collided_items / len(current),
        "collision_groups": sum(size > 1 for size in counts.values()),
        "max_bucket": max(counts.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    rows = []
    for update_window in range(1, 10):
        eval_window = update_window + 1
        for method in METHODS:
            run = root / f"stage3/runs/temporal_incremental/R3_W{update_window}_{method}_s2024"
            row = {
                "update_window": f"W{update_window}",
                "eval_window": f"W{eval_window}",
                "method": method,
                **parse_results(run / "train.log"),
                "downstream_seconds": parse_elapsed(run / "train.log"),
            }
            row["mean_ndcg10"] = (row["rec_ndcg10"] + row["search_ndcg10"]) / 2
            rows.append(row)
    frame = pd.DataFrame(rows)

    paired = []
    for eval_window, group in frame.groupby("eval_window", sort=False):
        indexed = group.set_index("method")
        for baseline in ("Fixed", "Random"):
            row = {"eval_window": eval_window, "comparison": f"Selective_minus_{baseline}"}
            for metric in (*METRICS, "mean_ndcg10"):
                row[metric] = indexed.loc["Selective", metric] - indexed.loc[baseline, metric]
            paired.append(row)
    paired_frame = pd.DataFrame(paired)

    aggregate = []
    for method, group in frame.groupby("method", sort=False):
        row = {"method": method}
        for metric in (*METRICS, "mean_ndcg10"):
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std()
        aggregate.append(row)
    aggregate_frame = pd.DataFrame(aggregate)

    wins = []
    for comparison, group in paired_frame.groupby("comparison", sort=False):
        for metric in (*METRICS, "mean_ndcg10"):
            values = group[metric]
            wins.append({
                "comparison": comparison,
                "metric": metric,
                "mean_delta": values.mean(),
                "positive_windows": int((values > 0).sum()),
                "zero_windows": int((values == 0).sum()),
                "negative_windows": int((values < 0).sum()),
            })
    wins_frame = pd.DataFrame(wins)

    sid_rows = []
    base = load(mapping_path(root, 1, "Fixed"))
    previous = {method: base for method in METHODS}
    for update_window in range(1, 10):
        for method in METHODS:
            current = load(mapping_path(root, update_window, method))
            search = sid_stats(previous[method], current, "search_sid")
            rec = sid_stats(previous[method], current, "rec_sid")
            sid_rows.append({
                "update_window": f"W{update_window}",
                "method": method,
                **{f"search_{key}": value for key, value in search.items()},
                **{f"rec_{key}": value for key, value in rec.items()},
            })
            previous[method] = current
    sid_frame = pd.DataFrame(sid_rows)

    cost_rows = []
    for update_window in range(2, 10):
        log_dir = root / f"stage3/runs/temporal_rolling_r3/W{update_window}"
        unisar = parse_elapsed(log_dir / "unisar_update.log")
        rec = parse_elapsed(log_dir / "rec_adapter.log")
        search = parse_elapsed(log_dir / "search_adapter.log")
        cost_rows.append({
            "update_window": f"W{update_window}",
            "unisar_seconds": unisar,
            "rec_adapter_seconds": rec,
            "search_adapter_seconds": search,
            "sid_learn_critical_seconds": unisar + max(rec, search),
        })
    cost_frame = pd.DataFrame(cost_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "window_metrics.csv", index=False)
    paired_frame.to_csv(args.output_dir / "window_deltas.csv", index=False)
    aggregate_frame.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    wins_frame.to_csv(args.output_dir / "window_wins.csv", index=False)
    sid_frame.to_csv(args.output_dir / "sid_stats.csv", index=False)
    cost_frame.to_csv(args.output_dir / "costs.csv", index=False)
    payload = {
        "aggregate_metrics": aggregate_frame.to_dict("records"),
        "window_wins": wins_frame.to_dict("records"),
        "final_sid_stats": sid_frame[sid_frame.update_window == "W9"].to_dict("records"),
        "sid_learning_cost": {
            "windows": len(cost_frame),
            "mean_critical_seconds": cost_frame.sid_learn_critical_seconds.mean(),
            "total_critical_seconds": cost_frame.sid_learn_critical_seconds.sum(),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(aggregate_frame.to_string(index=False))
    print("\n" + wins_frame.to_string(index=False))
    print("\nFinal SID stats:\n" + payload["final_sid_stats"].__repr__())
    print("\nSID update costs:\n" + cost_frame.to_string(index=False))


if __name__ == "__main__":
    main()
