#!/usr/bin/env python3
import csv
import json
import statistics
from pathlib import Path


WINDOWS = (3, 5, 9)
SEEDS = (42, 2024, 2025)
TASK_FILES = {
    "rec": "SeqRecWithSrcHis_sample_neg-999_beam-10.json",
    "search": "PerSrcWithRecHis_sample_neg-999_beam-10.json",
}


def read_metric(path: Path, task: str) -> dict:
    return json.loads((path / TASK_FILES[task]).read_text())["mean_results"]


def main():
    root = Path("stage3/runs/temporal_incremental")
    source = Path("stage3/results/temporal_r4_three_seed/all_metrics.csv")
    output = Path("stage3/results/temporal_search_lora_three_seed")
    output.mkdir(parents=True, exist_ok=True)

    baselines = {}
    with source.open() as handle:
        for row in csv.DictReader(handle):
            baselines[(int(row["seed"]), int(row["update_window"][1:]), row["method"])] = row

    rows = []
    for seed in SEEDS:
        for window in WINDOWS:
            path = root / f"R5_SearchLoRA_W{window}_s{seed}"
            rec = read_metric(path, "rec")
            search = read_metric(path, "search")
            rows.append({
                "seed": seed,
                "update_window": f"W{window}",
                "eval_window": f"W{window + 1}",
                "rec_hit10": rec["hit@10"],
                "rec_ndcg10": rec["ndcg@10"],
                "search_hit10": search["hit@10"],
                "search_ndcg10": search["ndcg@10"],
                "mean_ndcg10": statistics.mean((rec["ndcg@10"], search["ndcg@10"])),
            })

    with (output / "all_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for metric in ("rec_ndcg10", "search_ndcg10", "mean_ndcg10"):
        seed_means = {
            seed: statistics.mean(row[metric] for row in rows if row["seed"] == seed)
            for seed in SEEDS
        }
        summary[metric] = {
            "mean": statistics.mean(seed_means.values()),
            "seed_std": statistics.stdev(seed_means.values()),
            "seed_means": seed_means,
        }

    comparisons = {}
    fields = {
        "rec_ndcg10": "test_rec_ndcg10",
        "search_ndcg10": "test_search_ndcg10",
        "mean_ndcg10": "test_mean_ndcg10",
    }
    for method in ("Fixed", "Random", "Selective", "R4C"):
        comparisons[method] = {}
        for metric, baseline_field in fields.items():
            seed_deltas = {}
            for seed in SEEDS:
                values = []
                for row in rows:
                    if row["seed"] != seed:
                        continue
                    window = int(row["update_window"][1:])
                    baseline = float(baselines[(seed, window, method)][baseline_field])
                    values.append(row[metric] - baseline)
                seed_deltas[seed] = statistics.mean(values)
            comparisons[method][metric] = {
                "mean_delta": statistics.mean(seed_deltas.values()),
                "positive_seeds": sum(value > 0 for value in seed_deltas.values()),
                "seed_deltas": seed_deltas,
            }

    result = {
        "protocol": "Search LoRA enabled for Search; adapter disabled for Rec",
        "windows": list(WINDOWS),
        "seeds": list(SEEDS),
        "trainable_parameters": 294912,
        "trainable_fraction": 0.004789,
        "summary": summary,
        "comparisons": comparisons,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
