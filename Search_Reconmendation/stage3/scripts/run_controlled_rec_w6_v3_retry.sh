#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
GENSAR_DATA=/data/cjk/external/GenSAR/data
BASE=stage3/data/controlled_rec_drift_w3w6w9_v1/W6
QUARTERLY=stage3/data/gensar_faithful_20k_quarterly_controlled_rec_w3w6w9_v1
ITEM_IDS=stage3/data/gensar_faithful_20k_t0_2011q4/item_ids.jsonl
COMMON=/data/cjk/external/GenSAR/data/GenSAR_Faithful_20K_Strict/emb/bge-base-en-v1.5.pt
ANCHOR=stage3/data/gensar_faithful_20k_t0_2011q4/v13_anchor_full_p256/anchor_lookup.npz
MODEL=stage3/data/gensar_faithful_20k_t0_2011q4/v13_full_suffix_p256/direct_private_sid_model.pt
FIXED=stage3/data/temporal_joint_r2a_mappings/J0_fixed_sid_mapping.json
SELECTION=$BASE/selections/v3_impact_b51.csv
OUTPUT=$BASE/V3_sid
MAPPING=$OUTPUT/materialized/drift_selected_sticky_ratio_0p1/sid_mapping.json
EXPORT=$BASE/downstream/export
POOL=$BASE/downstream/alignment_items.json
DATASET=GenSAR_Faithful_20K_CRD_W6_V3
EXPERIMENT=CRD_W6_V3_s2024
RUN=$ROOT/stage3/runs/temporal_incremental/$EXPERIMENT
INITIAL=$ROOT/stage3/runs/temporal_incremental/CRD_W5_Fixed_s2024
GPU=${1:-5}
cd "$ROOT"
mkdir -p "$OUTPUT" "$EXPORT"

if [[ ! -f "$OUTPUT/rec_adapter/rec_adapter_best.pt" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python -u stage3/scripts/update_v13b_rec_adapter_selective.py \
    --checkpoint "$MODEL" --t0-mapping "$FIXED" \
    --data-dir "$BASE/snapshot" --common-embedding "$COMMON" \
    --updated-rec-embedding stage3/artifacts/unisar_controlled_rec_W6/UniSAR_faithful.pt \
    --t0-rec-embedding stage3/artifacts/unisar_controlled_rec_W5/UniSAR_faithful.pt \
    --anchor-lookup "$ANCHOR" --drift-csv "$SELECTION" \
    --drift-left W5 --drift-right W6 --replay-ratio 2 \
    --output-dir "$OUTPUT/rec_adapter" --epochs 60 --batch-size 64 \
    --lr 0.0001 --weight-decay 0.0001 --topology-weight 0.5 \
    --commit-weight 0.05 --stability-weight 0.1 --outside-stability-weight 0.1 \
    --deterministic-frozen-modules --patience 10 --device cuda:0 --seed 2026 \
    >"$OUTPUT/adapter.log" 2>&1
fi

if [[ ! -f "$OUTPUT/materialized/drift_selected_sticky_ratio_0p1/stats.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python stage3/scripts/materialize_v13b_temporal_rec_sid_sticky.py \
    --base-checkpoint "$MODEL" \
    --adapter-checkpoint "$OUTPUT/rec_adapter/rec_adapter_best.pt" \
    --base-operational-mapping "$FIXED" --data-dir "$BASE/snapshot" \
    --common-embedding "$COMMON" \
    --updated-rec-embedding stage3/artifacts/unisar_controlled_rec_W6/UniSAR_faithful.pt \
    --anchor-lookup "$ANCHOR" --drift-csv "$SELECTION" \
    --drift-left W5 --drift-right W6 --switch-ratios 0.1 \
    --output-dir "$OUTPUT/materialized" --batch-size 1024 \
    --device cuda:0 --seed 2026 >"$OUTPUT/materialize.log"
fi

python stage3/scripts/export_sid_mapping_for_gensar.py \
  --mapping "$MAPPING" --item-ids "$ITEM_IDS" --output-dir "$EXPORT" \
  --name CRD_W6_V3 >"$OUTPUT/export.log"

python stage3/scripts/materialize_temporal_incremental_gensar.py \
  --quarterly-dir "$QUARTERLY" --update-window W6 --eval-window W7 \
  --output-dir "$GENSAR_DATA/$DATASET" \
  --semantic-sid "$EXPORT/CRD_W6_V3_semantic.json" \
  --collaborate-sid "$EXPORT/CRD_W6_V3_collaborate.json" \
  --base-mapping "$FIXED" --current-mapping "$MAPPING" \
  --item-ids "$ITEM_IDS" --update-task both \
  --alignment-items "$POOL" --balanced-alignment --seed 2024 \
  >"$OUTPUT/downstream_materialize.log"

if [[ ! -f "$RUN/model.safetensors" \
    || ! -f "$RUN/SeqRecWithSrcHis_sample_neg-999_beam-10.json" \
    || ! -f "$RUN/PerSrcWithRecHis_sample_neg-999_beam-10.json" ]]; then
  bash stage3/scripts/run_temporal_incremental_t5.sh "$GPU" \
    "$EXPERIMENT" "$DATASET" CRD_W6_V3_semantic.json CRD_W6_V3_collaborate.json \
    1800 2024 "$INITIAL"
fi

find "$RUN" -depth \
  \( -path "$RUN/checkpoint-*" -o -path "$RUN/checkpoint-*/*" \) -delete
