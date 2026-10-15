#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
BASE=stage3/data/controlled_rec_drift_w3w6w9_v1
INPUT=$BASE/item_selector_v2_b51/rec_item_drift.csv
COMMON=/data/cjk/external/GenSAR/data/GenSAR_Faithful_20K_Strict/emb/bge-base-en-v1.5.pt
ANCHOR=stage3/data/gensar_faithful_20k_t0_2011q4/v13_anchor_full_p256/anchor_lookup.npz
MODEL=stage3/data/gensar_faithful_20k_t0_2011q4/v13_full_suffix_p256/direct_private_sid_model.pt
FIXED=stage3/data/temporal_joint_r2a_mappings/J0_fixed_sid_mapping.json
BUDGET=${1:?usage: $0 BUDGET [GPU_V3 GPU_SUPPORT GPU_RANDOM]}
GPU_V3=${2:-5}
GPU_SUPPORT=${3:-6}
GPU_RANDOM=${4:-7}
SELECTIONS=$BASE/W9/selections
cd "$ROOT"
mkdir -p "$SELECTIONS"

python stage3/scripts/build_detector_policy_selection.py \
  --input "$INPUT" --output "$SELECTIONS/v3_impact_b${BUDGET}.csv" \
  --left W8 --right W9 --policy impact_topk --budget "$BUDGET" \
  --support-power 0.5 --minimum-current-events 5
python stage3/scripts/build_detector_policy_selection.py \
  --input "$INPUT" --output "$SELECTIONS/support_top${BUDGET}.csv" \
  --left W8 --right W9 --policy support_topk --budget "$BUDGET"
python stage3/scripts/build_matched_random_drift_selection.py \
  --input "$SELECTIONS/v3_impact_b${BUDGET}.csv" \
  --output "$SELECTIONS/random_impact_b${BUDGET}.csv" \
  --left W8 --right W9 --seed 2026

run_arm() {
  local arm=$1 drift=$2 gpu=$3
  local output="$BASE/W9/${arm}_K${BUDGET}_sid"
  mkdir -p "$output"
  if [[ ! -f "$output/rec_adapter/rec_adapter_best.pt" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" python -u stage3/scripts/update_v13b_rec_adapter_selective.py \
      --checkpoint "$MODEL" --t0-mapping "$FIXED" \
      --data-dir "$BASE/W9/snapshot" --common-embedding "$COMMON" \
      --updated-rec-embedding stage3/artifacts/unisar_controlled_rec_W9/UniSAR_faithful.pt \
      --t0-rec-embedding stage3/artifacts/unisar_controlled_rec_W8/UniSAR_faithful.pt \
      --anchor-lookup "$ANCHOR" --drift-csv "$drift" \
      --drift-left W8 --drift-right W9 --replay-ratio 2 \
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
      --base-operational-mapping "$FIXED" --data-dir "$BASE/W9/snapshot" \
      --common-embedding "$COMMON" \
      --updated-rec-embedding stage3/artifacts/unisar_controlled_rec_W9/UniSAR_faithful.pt \
      --anchor-lookup "$ANCHOR" --drift-csv "$drift" \
      --drift-left W8 --drift-right W9 --switch-ratios 0.1 \
      --output-dir "$output/materialized" --batch-size 1024 \
      --device cuda:0 --seed 2026 >"$output/materialize.log"
  fi
}

pids=()
run_arm V3 "$SELECTIONS/v3_impact_b${BUDGET}.csv" "$GPU_V3" & pids+=("$!")
run_arm Support "$SELECTIONS/support_top${BUDGET}.csv" "$GPU_SUPPORT" & pids+=("$!")
run_arm Random "$SELECTIONS/random_impact_b${BUDGET}.csv" "$GPU_RANDOM" & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
