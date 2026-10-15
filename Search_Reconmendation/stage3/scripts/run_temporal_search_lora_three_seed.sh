#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
RUNS=$ROOT/stage3/runs/temporal_incremental
cd "$ROOT"

is_complete() {
  local output=$1
  [[ -f "$output/adapter_model.safetensors" \
    && -f "$output/SeqRecWithSrcHis_sample_neg-999_beam-10.json" \
    && -f "$output/PerSrcWithRecHis_sample_neg-999_beam-10.json" ]]
}

run_one() {
  local window=$1 seed=$2 gpu=$3
  local experiment="R5_SearchLoRA_W${window}_s${seed}"
  local output="$RUNS/$experiment"
  if is_complete "$output"; then
    echo "[skip] $experiment"
    return
  fi
  bash stage3/scripts/run_temporal_search_lora.sh \
    "$gpu" "$window" "$seed" "$experiment"
}

for window in 3 5 9; do
  pids=()
  gpu=4
  for seed in 42 2024 2025; do
    run_one "$window" "$seed" "$gpu" \
      >"stage3/runs/temporal_r5_search_lora_W${window}_s${seed}.log" 2>&1 &
    pids+=("$!")
    gpu=$((gpu + 1))
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done

