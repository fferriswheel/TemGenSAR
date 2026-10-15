#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
GENSAR_DATA=/data/cjk/external/GenSAR/data
RUNS=$ROOT/stage3/runs/temporal_incremental
QUARTERLY=stage3/data/gensar_faithful_20k_quarterly_controlled_search_w1
ITEM_IDS=stage3/data/gensar_faithful_20k_t0_2011q4/item_ids.jsonl
PREPARED=stage3/data/temporal_search_only_three_seed
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

prepare_window() {
  local window=$1
  local prev=$((window - 1)) eval=$((window + 1))
  local dir="$PREPARED/W${window}"
  local previous="stage3/data/temporal_rolling_r3/W${prev}/mappings/selective_joint.json"
  local current="stage3/data/temporal_rolling_r3/W${window}/mappings/selective_joint.json"
  local dataset="GenSAR_Faithful_20K_SearchOnly_W${window}"
  mkdir -p "$dir"

  python stage3/scripts/compose_temporal_task_mappings.py \
    --base-mapping "$previous" \
    --search-mapping "$current" \
    --rec-mapping "$previous" \
    --project-task-fields \
    --output "$dir/mapping.json" >"$dir/compose.log"

  python stage3/scripts/export_sid_mapping_for_gensar.py \
    --mapping "$dir/mapping.json" \
    --item-ids "$ITEM_IDS" \
    --output-dir "$dir/export" \
    --name "search_only_w${window}" >"$dir/export.log"

  # Search uses the current window. Rec receives unchanged-SID rehearsal so this
  # round isolates the mapping policy without changing the shared T5 architecture.
  python stage3/scripts/materialize_temporal_incremental_gensar.py \
    --quarterly-dir "$QUARTERLY" \
    --update-window "W${window}" \
    --eval-window "W${eval}" \
    --output-dir "$GENSAR_DATA/$dataset" \
    --semantic-sid "$dir/export/search_only_w${window}_semantic.json" \
    --collaborate-sid "$dir/export/search_only_w${window}_collaborate.json" \
    --base-mapping "$previous" \
    --current-mapping "$dir/mapping.json" \
    --item-ids "$ITEM_IDS" \
    --update-task both \
    --search-alignment-items "stage3/data/temporal_rolling_r3/W${window}/alignment/search_alignment_items.json" \
    --rec-alignment-items "stage3/data/temporal_rolling_r3/W${window}/alignment/rec_alignment_items.json" \
    --seed 2024 >"$dir/materialize.log"
}

run_one() {
  local window=$1 seed=$2 gpu=$3
  local experiment="R4SO_W${window}_s${seed}"
  local output="$RUNS/$experiment"
  if is_complete "$output"; then
    echo "[skip] $experiment"
    cleanup_checkpoint "$output"
    return
  fi

  echo "[run] window=W${window} seed=$seed gpu=$gpu"
  bash stage3/scripts/run_temporal_incremental_t5.sh "$gpu" \
    "$experiment" "GenSAR_Faithful_20K_SearchOnly_W${window}" \
    "search_only_w${window}_semantic.json" \
    "search_only_w${window}_collaborate.json" \
    1800 "$seed" \
    "$RUNS/R3_W$((window - 1))_Selective_s2024"
  cleanup_checkpoint "$output"
}

for window in 3 5 9; do
  prepare_window "$window"
done

for window in 3 5 9; do
  pids=()
  gpu=4
  for seed in 42 2024 2025; do
    run_one "$window" "$seed" "$gpu" \
      >"stage3/runs/temporal_r4so_W${window}_s${seed}.log" 2>&1 &
    pids+=("$!")
    gpu=$((gpu + 1))
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done

