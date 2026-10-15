#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
RUNS=$ROOT/stage3/runs/temporal_incremental
INITIAL=$RUNS/CRD_W5_Fixed_s2024
cd "$ROOT"

declare -A DATASETS=(
  [Fixed]=GenSAR_Faithful_20K_CRD_W6_Fixed
  [Support]=GenSAR_Faithful_20K_CRD_W6_Support
  [V3]=GenSAR_Faithful_20K_CRD_W6_V3
)
declare -A SEMANTIC=(
  [Fixed]=J0_fixed_semantic.json
  [Support]=CRD_W6_Support_semantic.json
  [V3]=CRD_W6_V3_semantic.json
)
declare -A COLLABORATE=(
  [Fixed]=J0_fixed_collaborate.json
  [Support]=CRD_W6_Support_collaborate.json
  [V3]=CRD_W6_V3_collaborate.json
)

is_complete() {
  local output=$1
  [[ -f "$output/model.safetensors" \
    && -f "$output/SeqRecWithSrcHis_sample_neg-999_beam-10.json" \
    && -f "$output/PerSrcWithRecHis_sample_neg-999_beam-10.json" ]]
}

run_arm() {
  local arm=$1 seed=$2 gpu=$3
  local experiment="CRD_W6_${arm}_s${seed}"
  local output="$RUNS/$experiment"
  if is_complete "$output"; then
    echo "[skip] $experiment"
    return
  fi
  bash stage3/scripts/run_temporal_incremental_t5.sh "$gpu" \
    "$experiment" "${DATASETS[$arm]}" "${SEMANTIC[$arm]}" \
    "${COLLABORATE[$arm]}" 1800 "$seed" "$INITIAL"
  find "$output" -depth \
    \( -path "$output/checkpoint-*" -o -path "$output/checkpoint-*/*" \) -delete
}

pids=()
run_arm Fixed 2025 4 & pids+=("$!")
run_arm Support 2025 5 & pids+=("$!")
run_arm V3 2025 6 & pids+=("$!")
run_arm Fixed 2026 7 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

pids=()
run_arm Support 2026 4 & pids+=("$!")
run_arm V3 2026 5 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
