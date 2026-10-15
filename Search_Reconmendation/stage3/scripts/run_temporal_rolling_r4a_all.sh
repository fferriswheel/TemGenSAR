#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
OUTPUT=$ROOT/stage3/runs/temporal_rolling_r4a_instance
mkdir -p "$OUTPUT"
cd "$ROOT"

complete() {
  local dir=$1
  python - "$dir" <<'PY' >/dev/null 2>&1
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
for name in (
    "SeqRecWithSrcHis_sample_neg-999_beam-10.json",
    "PerSrcWithRecHis_sample_neg-999_beam-10.json",
):
    payload = json.loads((p / name).read_text())
    assert len(payload["all_prompt_instance_results"]) == 1
    assert len(payload["all_prompt_instance_results"][0]) == 5000
PY
}

for n in $(seq 1 9); do
  window="W$n"
  pids=()
  for spec in Fixed:4 Selective:5 Random:6; do
    arm=${spec%%:*}
    gpu=${spec##*:}
    dir="$OUTPUT/${window}_${arm}"
    if complete "$dir"; then
      echo "[skip] $window $arm"
      continue
    fi
    bash stage3/scripts/run_temporal_rolling_instance_eval.sh \
      "$gpu" "$window" "$arm" "$dir" >"$OUTPUT/${window}_${arm}.launcher.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done

python stage3/scripts/evaluate_temporal_rolling_r4a.py \
  --annotations-dir stage3/data/temporal_rolling_r4a_annotations \
  --predictions-dir stage3/runs/temporal_rolling_r4a_instance \
  --output-dir stage3/results/temporal_rolling_r4a
