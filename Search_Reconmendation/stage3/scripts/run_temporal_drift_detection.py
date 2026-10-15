#!/usr/bin/env python3
"""Calibrate and run task-level Search/Rec temporal drift detection."""

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


WINDOWS = {
    "B1": ("2011-01-01", "2011-07-01"),
    "B2": ("2011-07-01", "2012-01-01"),
    "B3": ("2012-01-01", "2012-07-01"),
    "B4": ("2012-07-01", "2013-01-01"),
    "W1": ("2013-01-01", "2013-07-01"),
    "W2": ("2013-07-01", "2014-01-01"),
    "W3": ("2014-01-01", "2014-07-24"),
}
REFERENCE_PAIRS = [("B1", "B2"), ("B2", "B3"), ("B3", "B4")]
MONITOR_PAIRS = [("B4", "W1"), ("W1", "W2"), ("W2", "W3")]


def quarterly_protocol():
    calibration = {
        f"C{idx + 1}": str(period)
        for idx, period in enumerate(pd.period_range("2010Q1", "2011Q4", freq="Q"))
    }
    monitor = {
        f"W{idx + 1}": str(period)
        for idx, period in enumerate(pd.period_range("2012Q1", "2014Q2", freq="Q"))
    }
    windows = {}
    for name, period in {**calibration, **monitor}.items():
        value = pd.Period(period, freq="Q")
        windows[name] = (
            value.start_time.strftime("%Y-%m-%d"),
            (value + 1).start_time.strftime("%Y-%m-%d"),
        )
    calibration_names = list(calibration)
    monitor_names = list(monitor)
    reference_pairs = list(zip(calibration_names, calibration_names[1:]))
    monitor_pairs = [(calibration_names[-1], monitor_names[0])] + list(
        zip(monitor_names, monitor_names[1:])
    )
    return windows, reference_pairs, monitor_pairs


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize(counter, keys, smoothing=1e-8):
    values = np.asarray([counter.get(key, 0.0) for key in keys], dtype=np.float64)
    values += smoothing
    return values / values.sum()


def js_divergence(left, right):
    middle = 0.5 * (left + right)
    return float(
        0.5 * np.sum(left * np.log2(left / middle))
        + 0.5 * np.sum(right * np.log2(right / middle))
    )


def counter_js(left, right):
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    return js_divergence(normalize(left, keys), normalize(right, keys))


def l2_normalize(array):
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


@torch.no_grad()
def encode_queries(queries, model_name, cache_dir, batch_size, device):
    digest = hashlib.sha1((model_name + "\n" + "\n".join(queries)).encode()).hexdigest()[:16]
    cache = cache_dir / f"query_embeddings.{digest}.npz"
    if cache.exists():
        saved = np.load(cache, allow_pickle=True)
        if saved["queries"].tolist() == queries:
            return saved["embeddings"].astype(np.float32), str(cache)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    rows = []
    for start in range(0, len(queries), batch_size):
        batch = tokenizer(
            queries[start : start + batch_size], padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        ).to(device)
        output = model(**batch).last_hidden_state[:, 0]
        rows.append(output.cpu().numpy())
    embeddings = l2_normalize(np.concatenate(rows).astype(np.float32))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, queries=np.asarray(queries), embeddings=embeddings)
    return embeddings, str(cache)


def rbf_kernel(embeddings):
    similarity = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    distances = np.maximum(2.0 - 2.0 * similarity, 0.0)
    upper = distances[np.triu_indices(len(distances), 1)]
    positive = upper[upper > 1e-12]
    sigma2 = float(np.median(positive)) if len(positive) else 1.0
    return np.exp(-distances / max(2.0 * sigma2, 1e-12)), math.sqrt(sigma2)


def weighted_mmd(left, right, query_index, kernel):
    p = np.zeros(len(query_index), dtype=np.float64)
    q = np.zeros(len(query_index), dtype=np.float64)
    for key, value in left.items():
        p[query_index[key]] = value
    for key, value in right.items():
        q[query_index[key]] = value
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    p /= p.sum()
    q /= q.sum()
    return float(max(p @ kernel @ p + q @ kernel @ q - 2.0 * p @ kernel @ q, 0.0))


def aggregate_user_counters(frame, task):
    users = {}
    for user, group in frame.sort_values(["user_id", "timestamp"]).groupby("user_id", sort=False):
        result = {"events": len(group)}
        if task == "search":
            result["query"] = Counter(group["query"].astype(str))
            item_query = defaultdict(Counter)
            for row in group.itertuples():
                item_query[str(row.item_id)][str(row.query)] += 1
            result["item_query"] = dict(item_query)
        else:
            result["pop"] = Counter(group["item_id"].astype(str))
            transitions = defaultdict(Counter)
            items = group["item_id"].astype(str).tolist()
            for source, target in zip(items, items[1:]):
                transitions[source][target] += 1
            result["transition"] = dict(transitions)
        users[str(user)] = result
    return users


def combine_users(users, multiplicities, task):
    aggregate = {"events": 0}
    if task == "search":
        aggregate.update(query=Counter(), item_query=defaultdict(Counter))
    else:
        aggregate.update(pop=Counter(), transition=defaultdict(Counter))
    for user, multiplier in multiplicities.items():
        if multiplier <= 0:
            continue
        row = users[user]
        aggregate["events"] += row["events"] * multiplier
        if task == "search":
            aggregate["query"].update({k: v * multiplier for k, v in row["query"].items()})
            for item, counts in row["item_query"].items():
                aggregate["item_query"][item].update({k: v * multiplier for k, v in counts.items()})
        else:
            aggregate["pop"].update({k: v * multiplier for k, v in row["pop"].items()})
            for item, counts in row["transition"].items():
                aggregate["transition"][item].update({k: v * multiplier for k, v in counts.items()})
    return aggregate


def original_aggregate(users, task):
    return combine_users(users, {user: 1 for user in users}, task)


def bootstrap_aggregate(users, rng, task, target_events=None, sample_size=None):
    names = np.asarray(list(users), dtype=object)
    if not len(names):
        return original_aggregate(users, task)
    counts = Counter(
        rng.choice(
            names, size=sample_size if sample_size is not None else len(names), replace=True
        ).tolist()
    )
    result = combine_users(users, counts, task)
    # Event-count matching keeps thresholds from being driven by baseline traffic growth.
    if target_events and result["events"] > 0:
        scale = target_events / result["events"]
        # Fractional weights are supported by Counter and all downstream metrics.
        if task == "search":
            result["query"] = Counter({k: v * scale for k, v in result["query"].items()})
            for item in result["item_query"]:
                result["item_query"][item] = Counter(
                    {k: v * scale for k, v in result["item_query"][item].items()}
                )
        else:
            result["pop"] = Counter({k: v * scale for k, v in result["pop"].items()})
            for item in result["transition"]:
                result["transition"][item] = Counter(
                    {k: v * scale for k, v in result["transition"][item].items()}
                )
        result["events"] = target_events
    return result


def centroid_shift(left, right, query_index, embeddings, minimum_support):
    weighted_sum = 0.0
    total_weight = 0.0
    item_values = []
    for item in set(left) & set(right):
        left_counts, right_counts = left[item], right[item]
        left_n, right_n = sum(left_counts.values()), sum(right_counts.values())
        support = min(left_n, right_n)
        if support < minimum_support:
            continue
        left_vec = sum(value * embeddings[query_index[key]] for key, value in left_counts.items())
        right_vec = sum(value * embeddings[query_index[key]] for key, value in right_counts.items())
        left_vec /= max(left_n, 1e-12)
        right_vec /= max(right_n, 1e-12)
        cosine = float(left_vec @ right_vec / max(np.linalg.norm(left_vec) * np.linalg.norm(right_vec), 1e-12))
        drift = 1.0 - np.clip(cosine, -1.0, 1.0)
        weighted_sum += support * drift
        total_weight += support
        item_values.append(drift)
    return weighted_sum / max(total_weight, 1e-12), len(item_values)


def transition_metrics(left, right, minimum_support, top_k):
    js_sum = jaccard_sum = weight_sum = 0.0
    valid = 0
    for source in set(left) & set(right):
        left_counts, right_counts = left[source], right[source]
        support = min(sum(left_counts.values()), sum(right_counts.values()))
        if support < minimum_support:
            continue
        candidates = set(key for key, _ in left_counts.most_common(top_k)) | set(
            key for key, _ in right_counts.most_common(top_k)
        )
        l_top = set(key for key, _ in left_counts.most_common(top_k))
        r_top = set(key for key, _ in right_counts.most_common(top_k))
        union = l_top | r_top
        neighbor_drift = 1.0 - len(l_top & r_top) / max(len(union), 1)
        l_other = sum(value for key, value in left_counts.items() if key not in candidates)
        r_other = sum(value for key, value in right_counts.items() if key not in candidates)
        l_dist = Counter({key: left_counts.get(key, 0.0) for key in candidates})
        r_dist = Counter({key: right_counts.get(key, 0.0) for key in candidates})
        l_dist["__OTHER__"] = l_other
        r_dist["__OTHER__"] = r_other
        js_sum += support * counter_js(l_dist, r_dist)
        jaccard_sum += support * neighbor_drift
        weight_sum += support
        valid += 1
    return js_sum / max(weight_sum, 1e-12), jaccard_sum / max(weight_sum, 1e-12), valid


def search_metrics(left, right, context, minimum_support):
    centroid, valid_items = centroid_shift(
        left["item_query"], right["item_query"], context["query_index"],
        context["query_embeddings"], minimum_support
    )
    return {
        "query_js": counter_js(left["query"], right["query"]),
        "query_mmd": weighted_mmd(
            left["query"], right["query"], context["query_index"], context["query_kernel"]
        ),
        "item_query_centroid": centroid,
        "valid_relation_items": valid_items,
    }


def rec_metrics(left, right, minimum_support, top_k):
    transition_js, neighbor_drift, valid_items = transition_metrics(
        left["transition"], right["transition"], minimum_support, top_k
    )
    return {
        "popularity_js": counter_js(left["pop"], right["pop"]),
        "transition_js": transition_js,
        "neighbor_jaccard_drift": neighbor_drift,
        "valid_relation_items": valid_items,
    }


def empirical_p(null, observed):
    values = np.asarray(null, dtype=np.float64)
    return float((1 + np.sum(values >= observed)) / (len(values) + 1))


def bh_significant(p_values, alpha):
    ordered = sorted(p_values, key=p_values.get)
    cutoff = -1
    for rank, key in enumerate(ordered, 1):
        if p_values[key] <= alpha * rank / len(ordered):
            cutoff = rank
    return {key: rank <= cutoff for rank, key in enumerate(ordered, 1)}


def calibrate(task, users_by_window, metric_fn, reference_pairs, repetitions, rng):
    null = defaultdict(list)
    for left_name, right_name in reference_pairs:
        left_users, right_users = users_by_window[left_name], users_by_window[right_name]
        left_events = sum(row["events"] for row in left_users.values())
        right_events = sum(row["events"] for row in right_users.values())
        # Remove the time label before resampling. Keeping separate left/right pools
        # would preserve any real baseline-period drift in the null distribution.
        pooled = {
            **{f"left::{user}": row for user, row in left_users.items()},
            **{f"right::{user}": row for user, row in right_users.items()},
        }
        for _ in range(repetitions):
            left = bootstrap_aggregate(
                pooled, rng, task, left_events, sample_size=len(left_users)
            )
            right = bootstrap_aggregate(
                pooled, rng, task, right_events, sample_size=len(right_users)
            )
            for key, value in metric_fn(left, right).items():
                if key != "valid_relation_items":
                    null[key].append(float(value))
    return dict(null)


def summarize_null(null):
    result = {}
    for key, values in null.items():
        array = np.asarray(values)
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median)))
        result[key] = {
            "count": len(values), "median": median, "mad": mad,
            "q90": float(np.quantile(array, 0.90)),
            "q95": float(np.quantile(array, 0.95)),
            "q99": float(np.quantile(array, 0.99)),
        }
    return result


def evaluate_pairs(task, users_by_window, metric_fn, null, monitor_pairs, alpha, relation_metric):
    rows = []
    for left_name, right_name in monitor_pairs:
        left = original_aggregate(users_by_window[left_name], task)
        right = original_aggregate(users_by_window[right_name], task)
        metrics = metric_fn(left, right)
        p_values = {key: empirical_p(null[key], value) for key, value in metrics.items() if key in null}
        significant = bh_significant(p_values, alpha)
        task_trigger = sum(significant.values()) >= 2 and significant.get(relation_metric, False)
        rows.append({
            "left": left_name, "right": right_name,
            "left_events": left["events"], "right_events": right["events"],
            "metrics": metrics, "p_values": p_values,
            "bh_significant": significant, "task_trigger": task_trigger,
        })
    return rows


def load_windows(path, task, windows):
    frame = pd.read_json(path, lines=True)
    if not np.issubdtype(frame["timestamp"].dtype, np.datetime64):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True).dt.tz_localize(None)
    result, coverage = {}, {}
    for name, (start, end) in windows.items():
        rows = frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)].copy()
        result[name] = aggregate_user_counters(rows, task)
        sizes = rows.groupby("user_id").size()
        coverage[name] = {
            "start": start, "end_exclusive": end, "events": len(rows),
            "users": int(rows["user_id"].nunique()),
            "items": int(rows["item_id"].nunique()),
            "users_with_2plus_events": int((sizes >= 2).sum()),
        }
        if task == "search":
            coverage[name]["queries"] = int(rows["query"].nunique())
    return result, coverage, frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    parser.add_argument("--minimum-relation-support", type=int, default=5)
    parser.add_argument("--neighbor-top-k", type=int, default=20)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--window-mode", choices=["halfyear", "quarterly"], default="halfyear")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    windows, reference_pairs, monitor_pairs = (
        quarterly_protocol()
        if args.window_mode == "quarterly"
        else (WINDOWS, REFERENCE_PAIRS, MONITOR_PAIRS)
    )
    search_users, search_coverage, search_frame = load_windows(
        args.data_dir / "search_interactions.jsonl", "search", windows
    )
    rec_users, rec_coverage, _ = load_windows(
        args.data_dir / "rec_interactions_chrono.jsonl", "rec", windows
    )
    queries = sorted(search_frame["query"].astype(str).unique().tolist())
    embeddings, cache = encode_queries(
        queries, args.model, args.output_dir / "cache", args.batch_size, args.device
    )
    kernel, sigma = rbf_kernel(embeddings)
    context = {
        "query_index": {query: idx for idx, query in enumerate(queries)},
        "query_embeddings": embeddings,
        "query_kernel": kernel,
    }
    search_fn = lambda left, right: search_metrics(
        left, right, context, args.minimum_relation_support
    )
    rec_fn = lambda left, right: rec_metrics(
        left, right, args.minimum_relation_support, args.neighbor_top_k
    )

    search_null = calibrate(
        "search", search_users, search_fn, reference_pairs,
        args.bootstrap_repetitions, rng
    )
    rec_null = calibrate(
        "rec", rec_users, rec_fn, reference_pairs,
        args.bootstrap_repetitions, rng
    )
    result = {
        "config": {**vars(args), "data_dir": str(args.data_dir), "output_dir": str(args.output_dir)},
        "protocol": {
            "windows": windows, "reference_pairs": reference_pairs,
            "monitor_pairs": monitor_pairs,
            "threshold": "pooled reference-pair user-block bootstrap q95",
            "trigger": "BH-FDR significant on >=2 metrics including relation metric",
            "search_relation_metric": "item_query_centroid",
            "rec_relation_metric": "transition_js",
        },
        "embedding": {
            "model": args.model, "query_count": len(queries),
            "rbf_sigma": sigma, "cache": cache,
        },
        "coverage": {"search": search_coverage, "rec": rec_coverage},
        "null": {"search": summarize_null(search_null), "rec": summarize_null(rec_null)},
        "monitor": {
            "search": evaluate_pairs(
                "search", search_users, search_fn, search_null, monitor_pairs,
                args.fdr_alpha, "item_query_centroid"
            ),
            "rec": evaluate_pairs(
                "rec", rec_users, rec_fn, rec_null, monitor_pairs,
                args.fdr_alpha, "transition_js"
            ),
        },
    }
    result["config"]["output_dir"] = str(result["config"]["output_dir"])
    write_json(args.output_dir / "temporal_drift_report.json", result)
    pd.DataFrame([
        {"task": task, "left": row["left"], "right": row["right"],
         "trigger": row["task_trigger"], **row["metrics"]}
        for task, rows in result["monitor"].items() for row in rows
    ]).to_csv(args.output_dir / "temporal_drift_metrics.csv", index=False)
    print(json.dumps({
        "output": str(args.output_dir / "temporal_drift_report.json"),
        "search": result["monitor"]["search"], "rec": result["monitor"]["rec"]
    }, indent=2))


if __name__ == "__main__":
    main()
