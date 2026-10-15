#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
GENSAR_DATA=/data/cjk/external/GenSAR/data
BASE=stage3/data/controlled_rec_drift_w3w6w9_v1/W3
QUARTERLY=stage3/data/gensar_faithful_20k_quarterly_controlled_rec_w3w6w9_v1
ITEM_IDS=stage3/data/gensar_faithful_20k_t0_2011q4/item_ids.jsonl
J0=stage3/data/temporal_joint_r2a_mappings/J0_fixed_sid_mapping.json
SELECTIVE=$BASE/selective_sid/materialized/drift_selected_sticky_ratio_0p1/sid_mapping.json
RANDOM_MAPPING=$BASE/random_sid/materialized/drift_selected_sticky_ratio_0p1/sid_mapping.json
OUT=$BASE/downstream
POOL=$OUT/alignment_items.json
INITIAL=$ROOT/stage3/runs/temporal_incremental/CRD_Clean_W2_Fixed_s2024
cd "$ROOT"
mkdir -p "$OUT"

python stage3/scripts/build_priority_alignment_pool.py \
  --item-ids "$ITEM_IDS" \
  --priority-update-stats "$BASE/selective_sid/rec_adapter/update_stats.json" \
  --priority-update-stats "$BASE/random_sid/rec_adapter/update_stats.json" \
  --size 1800 --seed 2024 --output "$POOL" >"$OUT/alignment_pool.log"

prepare() {
  local arm=$1 mapping=$2 name=$3
  local export_dir="$OUT/$arm/export"
  local dataset="GenSAR_Faithful_20K_CRD_W3_${arm}"
  mkdir -p "$export_dir"
  python stage3/scripts/export_sid_mapping_for_gensar.py \
    --mapping "$mapping" --item-ids "$ITEM_IDS" --output-dir "$export_dir" \
    --name "$name" >"$OUT/$arm/export.log"
  python stage3/scripts/materialize_temporal_incremental_gensar.py \
    --quarterly-dir "$QUARTERLY" --update-window W3 --eval-window W4 \
    --output-dir "$GENSAR_DATA/$dataset" \
    --semantic-sid "$export_dir/${name}_semantic.json" \
    --collaborate-sid "$export_dir/${name}_collaborate.json" \
    --base-mapping "$J0" --current-mapping "$mapping" --item-ids "$ITEM_IDS" \
    --update-task both --alignment-items "$POOL" --balanced-alignment \
    --seed 2024 >"$OUT/$arm/materialize.log"
}

prepare Fixed "$J0" crd_w3_fixed
prepare Random "$RANDOM_MAPPING" crd_w3_random
prepare Selective "$SELECTIVE" crd_w3_selective

run_arm() {
  local gpu=$1 arm=$2 name=$3
  local experiment="CRD_W3_${arm}_s2024"
  local dataset="GenSAR_Faithful_20K_CRD_W3_${arm}"
  local run_dir="$ROOT/stage3/runs/temporal_incremental/$experiment"
  if [[ -f "$run_dir/model.safetensors" \
      && -f "$run_dir/SeqRecWithSrcHis_sample_neg-999_beam-10.json" \
      && -f "$run_dir/PerSrcWithRecHis_sample_neg-999_beam-10.json" ]]; then
    echo "[skip] $experiment"
    return
  fi
  bash stage3/scripts/run_temporal_incremental_t5.sh "$gpu" \
    "$experiment" "$dataset" "${name}_semantic.json" "${name}_collaborate.json" \
    1800 2024 "$INITIAL" >"$OUT/$arm/downstream.log" 2>&1
  find "$run_dir" -depth \
    \( -path "$run_dir/checkpoint-*" -o -path "$run_dir/checkpoint-*/*" \) -delete
}

pids=()
run_arm 4 Fixed crd_w3_fixed & pids+=("$!")
run_arm 5 Random crd_w3_random & pids+=("$!")
run_arm 6 Selective crd_w3_selective & pids+=("$!")
for pid in "${pids[@]}"; do
  wait "$pid"
done
