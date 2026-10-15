#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
BASE=stage3/data/controlled_rec_drift_w3w6w9_v1/W6
COMMON=/data/cjk/external/GenSAR/data/GenSAR_Faithful_20K_Strict/emb/bge-base-en-v1.5.pt
ANCHOR=stage3/data/gensar_faithful_20k_t0_2011q4/v13_anchor_full_p256/anchor_lookup.npz
MODEL=stage3/data/gensar_faithful_20k_t0_2011q4/v13_full_suffix_p256/direct_private_sid_model.pt
FIXED=stage3/data/temporal_joint_r2a_mappings/J0_fixed_sid_mapping.json
OUTPUT=$BASE/Full_adapter_cost
GPU=${1:-5}
cd "$ROOT"

if [[ ! -f "$OUTPUT/rec_adapter_best.pt" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python -u stage3/scripts/update_v13b_rec_adapter.py \
    --checkpoint "$MODEL" --t0-mapping "$FIXED" \
    --data-dir "$BASE/snapshot" --common-embedding "$COMMON" \
    --updated-rec-embedding stage3/artifacts/unisar_controlled_rec_W6/UniSAR_faithful.pt \
    --t0-rec-embedding stage3/artifacts/unisar_controlled_rec_W5/UniSAR_faithful.pt \
    --anchor-lookup "$ANCHOR" --output-dir "$OUTPUT" \
    --epochs 60 --batch-size 1024 --lr 0.0001 --weight-decay 0.0001 \
    --topology-weight 0.5 --commit-weight 0.05 --stability-weight 0.1 \
    --patience 6 --device cuda:0 --seed 2026 \
    >"$BASE/full_adapter_cost.log" 2>&1
fi
