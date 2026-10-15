#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import re
from pathlib import Path


TASK_FILES = {
    "rec": "SeqRecWithSrcHis_sample_neg-999_beam-10.json",
    "search": "PerSrcWithRecHis_sample_neg-999_beam-10.json",
}
WINDOWS = (3, 5, 9)
METHOD_PATTERNS = {
    "R3": "R3_W{window}_Selective_s2024",
    "R4B-B1": "R4B_B1_W{window}_s2024",
    "R4B-B2": "R4B_B2_W{window}_s2024",
    "R4C-C0": "R4C_C0_W{window}_s2024",
    "R4C-C01": "R4C_C01_W{window}_s2024",
}
MEAN_RESULTS_RE = re.compile(r"^Mean results:\s+(\{.*\})$", re.MULTILINE)


def read_test_metrics(run_dir: Path) -> dict[str, dict[str, float]]:
    result = {}
    for task, filename in TASK_FILES.items():
        path = run_dir / filename
        if not path.exists():
            continue
        with path.open() as handle:
            result[task] = json.load(handle)["mean_results"]
    return result


def read_validation_metrics(run_dir: Path) -> dict[str, dict[str, float]]:
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return {}
    matches = MEAN_RESULTS_RE.findall(log_path.read_text(errors="replace").replace("\r", "\n"))
    if len(matches) < 2:
        return {}
    # Validation is emitted in the same order as val_tasks: Rec, then Search.
    return {
        "rec": ast.literal_eval(matches[0]),
        "search": ast.literal_eval(matches[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("stage3/runs/temporal_incremental"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("stage3/results/temporal_r4_summary"),
    )
    parser.add_argument("--validation-margin", type=float, default=0.0002)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    run_metrics = {}
    for method, pattern in METHOD_PATTERNS.items():
        for window in WINDOWS:
            run_dir = args.runs_dir / pattern.format(window=window)
            test = read_test_metrics(run_dir)
            validation = read_validation_metrics(run_dir)
            if not test:
                continue
            key = (method, window)
            run_metrics[key] = {"test": test, "validation": validation}
            row = {"method": method, "update_window": f"W{window}", "eval_window": f"W{window + 1}"}
            for split, metrics in (("val", validation), ("test", test)):
                for task in ("rec", "search"):
                    values = metrics.get(task, {})
                    row[f"{split}_{task}_hit10"] = values.get("hit@10")
                    row[f"{split}_{task}_ndcg10"] = values.get("ndcg@10")
                ndcgs = [metrics[t]["ndcg@10"] for t in ("rec", "search") if t in metrics]
                row[f"{split}_mean_ndcg10"] = sum(ndcgs) / len(ndcgs) if ndcgs else None
            rows.append(row)

    columns = list(rows[0])
    with (args.output_dir / "representative_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    gate_rows = []
    for window in WINDOWS:
        base = run_metrics.get(("R3", window))
        candidate = run_metrics.get(("R4C-C0", window))
        if not base or not candidate:
            continue
        base_val = sum(base["validation"][t]["ndcg@10"] for t in TASK_FILES) / 2
        candidate_val = sum(candidate["validation"][t]["ndcg@10"] for t in TASK_FILES) / 2
        accepted = candidate_val - base_val >= args.validation_margin
        chosen = candidate if accepted else base
        base_test = sum(base["test"][t]["ndcg@10"] for t in TASK_FILES) / 2
        chosen_test = sum(chosen["test"][t]["ndcg@10"] for t in TASK_FILES) / 2
        gate_rows.append(
            {
                "update_window": f"W{window}",
                "eval_window": f"W{window + 1}",
                "accepted_r4c_c0": accepted,
                "validation_delta": candidate_val - base_val,
                "base_test_mean_ndcg10": base_test,
                "chosen_test_mean_ndcg10": chosen_test,
                "realized_test_delta": chosen_test - base_test,
            }
        )

    with (args.output_dir / "validation_gate_diagnostic.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gate_rows[0]))
        writer.writeheader()
        writer.writerows(gate_rows)

    summary = {
        "windows": list(WINDOWS),
        "validation_margin": args.validation_margin,
        "note": "The gate is a retrospective diagnostic and must be validated on new windows/seeds.",
        "method_test_mean_ndcg10": {},
        "gate": {
            "accepted_windows": [row["update_window"] for row in gate_rows if row["accepted_r4c_c0"]],
            "mean_realized_test_delta": sum(row["realized_test_delta"] for row in gate_rows) / len(gate_rows),
            "rows": gate_rows,
        },
    }
    for method in METHOD_PATTERNS:
        values = [row["test_mean_ndcg10"] for row in rows if row["method"] == method]
        if values:
            summary["method_test_mean_ndcg10"][method] = sum(values) / len(values)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
