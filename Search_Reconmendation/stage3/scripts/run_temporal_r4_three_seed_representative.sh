#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
RUNS=$ROOT/stage3/runs/temporal_incremental
cd "$ROOT"

is_complete() {
  local output=$1
  [[ -f "$output/model.safetensors" \
    && -f "$output/SeqRecWithSrcHis_sample_neg-999_beam-10.json" \
    && -f "$output/PerSrcWithRecHis_sample_neg-999_beam-10.json" ]]
}

cleanup_checkpoint() {
  local output=$1
  [[ -f "$output/model.safetensors" ]] || return 0
  find "$output" -depth \
    \( -path "$output/checkpoint-*" -o -path "$output/checkpoint-*/*" \) -delete
}

run_one() {
  local window=$1 method=$2 seed=$3 gpu=$4
  local prev=$((window - 1))
  local experiment dataset semantic collaborate initial

  case "$method" in
    Fixed)
      experiment="R4S_W${window}_Fixed_s${seed}"
      dataset="GenSAR_Faithful_20K_R3_W${window}_Fixed"
      semantic="W${window}_fixed_semantic.json"
      collaborate="W${window}_fixed_collaborate.json"
      initial="$RUNS/R3_W${prev}_Fixed_s2024"
      ;;
    Random)
      experiment="R4S_W${window}_Random_s${seed}"
      dataset="GenSAR_Faithful_20K_R3_W${window}_Random"
      semantic="W${window}_random_semantic.json"
      collaborate="W${window}_random_collaborate.json"
      initial="$RUNS/R3_W${prev}_Random_s2024"
      ;;
    Selective)
      experiment="R4S_W${window}_Selective_s${seed}"
      dataset="GenSAR_Faithful_20K_R3_W${window}_Selective"
      semantic="W${window}_selective_semantic.json"
      collaborate="W${window}_selective_collaborate.json"
      initial="$RUNS/R3_W${prev}_Selective_s2024"
      ;;
    R4C)
      experiment="R4S_W${window}_R4C_s${seed}"
      dataset="GenSAR_Faithful_20K_R4C_W${window}_C0"
      semantic="r4c_w${window}_c0_semantic.json"
      collaborate="r4c_w${window}_c0_collaborate.json"
      initial="$RUNS/R3_W${prev}_Selective_s2024"
      ;;
    *)
      echo "unknown method: $method" >&2
      return 2
      ;;
  esac

  local output="$RUNS/$experiment"
  if is_complete "$output"; then
    echo "[skip] $experiment"
    cleanup_checkpoint "$output"
    return
  fi

  echo "[run] window=W${window} method=$method seed=$seed gpu=$gpu"
  bash stage3/scripts/run_temporal_incremental_t5.sh "$gpu" \
    "$experiment" "$dataset" "$semantic" "$collaborate" 1800 "$seed" "$initial"
  cleanup_checkpoint "$output"
}

for seed in 42 2025; do
  for window in 3 5 9; do
    pids=()
    methods=(Fixed Random Selective R4C)
    for index in "${!methods[@]}"; do
      method=${methods[$index]}
      gpu=$((index + 4))
      run_one "$window" "$method" "$seed" "$gpu" \
        >"stage3/runs/temporal_r4s_W${window}_${method}_s${seed}.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
  done
done
