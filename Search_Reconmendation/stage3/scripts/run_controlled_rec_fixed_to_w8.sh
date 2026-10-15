#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
GENSAR_DATA=/data/cjk/external/GenSAR/data
QUARTERLY=stage3/data/gensar_faithful_20k_quarterly_controlled_rec_w3w6w9_v1
ITEM_IDS=stage3/data/gensar_faithful_20k_t0_2011q4/item_ids.jsonl
MAPPING=stage3/data/temporal_joint_r2a_mappings/J0_fixed_sid_mapping.json
EXPORT=stage3/data/temporal_joint_r2a_mappings/export
POOL=stage3/data/controlled_rec_drift_w3w6w9_v1/W3/downstream/alignment_items.json
RUNS=$ROOT/stage3/runs/temporal_incremental
cd "$ROOT"

is_complete() {
  local output=$1
  [[ -f "$output/model.safetensors" \
    && -f "$output/SeqRecWithSrcHis_sample_neg-999_beam-10.json" \
    && -f "$output/PerSrcWithRecHis_sample_neg-999_beam-10.json" ]]
}

prepare_window() {
  local window=$1 eval=$2 dataset=$3
  python stage3/scripts/materialize_temporal_incremental_gensar.py \
    --quarterly-dir "$QUARTERLY" --update-window "$window" --eval-window "$eval" \
    --output-dir "$GENSAR_DATA/$dataset" \
    --semantic-sid "$EXPORT/J0_fixed_semantic.json" \
    --collaborate-sid "$EXPORT/J0_fixed_collaborate.json" \
    --base-mapping "$MAPPING" --current-mapping "$MAPPING" \
    --item-ids "$ITEM_IDS" --update-task both \
    --alignment-items "$POOL" --balanced-alignment --seed 2024 \
    >"stage3/runs/${dataset}.materialize.log"
}

run_window() {
  local gpu=$1 window=$2 eval=$3 initial=$4
  local experiment="CRD_${window}_Fixed_s2024"
  local dataset="GenSAR_Faithful_20K_CRD_${window}_Fixed"
  local output="$RUNS/$experiment"
  if is_complete "$output"; then
    echo "[skip] $experiment"
    return
  fi
  prepare_window "$window" "$eval" "$dataset"
  bash stage3/scripts/run_temporal_incremental_t5.sh "$gpu" \
    "$experiment" "$dataset" J0_fixed_semantic.json J0_fixed_collaborate.json \
    1800 2024 "$initial"
  find "$output" -depth \
    \( -path "$output/checkpoint-*" -o -path "$output/checkpoint-*/*" \) -delete
}

GPU=${1:-5}
run_window "$GPU" W7 W8 "$RUNS/CRD_W6_Fixed_s2024"
run_window "$GPU" W8 W9 "$RUNS/CRD_W7_Fixed_s2024"
