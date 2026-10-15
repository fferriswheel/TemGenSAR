#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
BASE=stage3/data/controlled_rec_drift_w3w6w9_v1/W6
COMMON=/data/cjk/external/GenSAR/data/GenSAR_Faithful_20K_Strict/emb/bge-base-en-v1.5.pt
ANCHOR=stage3/data/gensar_faithful_20k_t0_2011q4/v13_anchor_full_p256/anchor_lookup.npz
MODEL=stage3/data/gensar_faithful_20k_t0_2011q4/v13_full_suffix_p256/direct_private_sid_model.pt
FIXED=stage3/data/temporal_joint_r2a_mappings/J0_fixed_sid_mapping.json
cd "$ROOT"

run_arm() {
  local policy=$1 budget=$2 gpu=$3
  local selection output
  if [[ "$policy" == V3 ]]; then
    selection="$BASE/selections/v3_impact_b${budget}.csv"
  else
    selection="$BASE/selections/support_top${budget}.csv"
  fi
  output="$BASE/Budget_${policy}_K${budget}_sid"
  mkdir -p "$output"
  if [[ ! -f "$output/rec_adapter/rec_adapter_best.pt" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" python -u stage3/scripts/update_v13b_rec_adapter_selective.py \
      --checkpoint "$MODEL" --t0-mapping "$FIXED" \
      --data-dir "$BASE/snapshot" --common-embedding "$COMMON" \
      --updated-rec-embedding stage3/artifacts/unisar_controlled_rec_W6/UniSAR_faithful.pt \
      --t0-rec-embedding stage3/artifacts/unisar_controlled_rec_W5/UniSAR_faithful.pt \
      --anchor-lookup "$ANCHOR" --drift-csv "$selection" \
      --drift-left W5 --drift-right W6 --replay-ratio 2 \
      --output-dir "$output/rec_adapter" --epochs 60 --batch-size 64 \
      --lr 0.0001 --weight-decay 0.0001 --topology-weight 0.5 \
      --commit-weight 0.05 --stability-weight 0.1 --outside-stability-weight 0.1 \
      --deterministic-frozen-modules --patience 10 --device cuda:0 --seed 2026 \
      >"$output/adapter.log" 2>&1
  fi
  if [[ ! -f "$output/materialized/drift_selected_sticky_ratio_0p1/stats.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" python stage3/scripts/materialize_v13b_temporal_rec_sid_sticky.py \
      --base-checkpoint "$MODEL" \
      --adapter-checkpoint "$output/rec_adapter/rec_adapter_best.pt" \
      --base-operational-mapping "$FIXED" --data-dir "$BASE/snapshot" \
      --common-embedding "$COMMON" \
      --updated-rec-embedding stage3/artifacts/unisar_controlled_rec_W6/UniSAR_faithful.pt \
      --anchor-lookup "$ANCHOR" --drift-csv "$selection" \
      --drift-left W5 --drift-right W6 --switch-ratios 0.1 \
      --output-dir "$output/materialized" --batch-size 1024 \
      --device cuda:0 --seed 2026 >"$output/materialize.log"
  fi
}

pids=()
run_arm V3 25 6 & pids+=("$!")
run_arm Support 25 7 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

pids=()
run_arm V3 100 6 & pids+=("$!")
run_arm Support 100 7 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

pids=()
run_arm V3 200 6 & pids+=("$!")
run_arm Support 200 7 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
