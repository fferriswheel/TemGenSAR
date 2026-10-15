#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
BASE=stage3/data/controlled_rec_drift_w3w6w9_v1
COMMON=/data/cjk/external/GenSAR/data/GenSAR_Faithful_20K_Strict/emb/bge-base-en-v1.5.pt
ANCHOR=stage3/data/gensar_faithful_20k_t0_2011q4/v13_anchor_full_p256/anchor_lookup.npz
MODEL=stage3/data/gensar_faithful_20k_t0_2011q4/v13_full_suffix_p256/direct_private_sid_model.pt
MAPPING=stage3/data/temporal_joint_r2a_mappings/J0_fixed_sid_mapping.json
SOURCE_DRIFT=$BASE/item_selector/rec_item_drift.csv
DRIFT=$BASE/W3/random_sid/matched_random_rec_item_drift.csv
OUTPUT=$BASE/W3/random_sid
cd "$ROOT"
mkdir -p "$OUTPUT"

if [[ ! -f "$DRIFT" ]]; then
  python stage3/scripts/build_matched_random_drift_selection.py \
    --input "$SOURCE_DRIFT" --output "$DRIFT" \
    --left W2 --right W3 --seed 2026 >"$OUTPUT/selection.log"
fi

if [[ ! -f "$OUTPUT/rec_adapter/rec_adapter_best.pt" ]]; then
  CUDA_VISIBLE_DEVICES=5 python -u stage3/scripts/update_v13b_rec_adapter_selective.py \
    --checkpoint "$MODEL" --t0-mapping "$MAPPING" \
    --data-dir "$BASE/W3/snapshot" --common-embedding "$COMMON" \
    --updated-rec-embedding stage3/artifacts/unisar_controlled_rec_W3/UniSAR_faithful.pt \
    --t0-rec-embedding stage3/artifacts/unisar_controlled_rec_clean_W2/UniSAR_faithful.pt \
    --anchor-lookup "$ANCHOR" --drift-csv "$DRIFT" \
    --drift-left W2 --drift-right W3 --replay-ratio 2 \
    --output-dir "$OUTPUT/rec_adapter" --epochs 60 --batch-size 64 \
    --lr 0.0001 --weight-decay 0.0001 --topology-weight 0.5 \
    --commit-weight 0.05 --stability-weight 0.1 --outside-stability-weight 0.1 \
    --deterministic-frozen-modules --patience 10 --device cuda:0 --seed 2026 \
    >"$OUTPUT/adapter.log" 2>&1
fi

if [[ ! -f "$OUTPUT/materialized/drift_selected_sticky_ratio_0p1/stats.json" ]]; then
  CUDA_VISIBLE_DEVICES=5 python stage3/scripts/materialize_v13b_temporal_rec_sid_sticky.py \
    --base-checkpoint "$MODEL" \
    --adapter-checkpoint "$OUTPUT/rec_adapter/rec_adapter_best.pt" \
    --base-operational-mapping "$MAPPING" --data-dir "$BASE/W3/snapshot" \
    --common-embedding "$COMMON" \
    --updated-rec-embedding stage3/artifacts/unisar_controlled_rec_W3/UniSAR_faithful.pt \
    --anchor-lookup "$ANCHOR" --drift-csv "$DRIFT" \
    --drift-left W2 --drift-right W3 --switch-ratios 0.1 \
    --output-dir "$OUTPUT/materialized" --batch-size 1024 \
    --device cuda:0 --seed 2026 >"$OUTPUT/materialize.log"
fi
