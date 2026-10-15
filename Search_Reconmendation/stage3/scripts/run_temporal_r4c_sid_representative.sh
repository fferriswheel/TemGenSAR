#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
GENSAR_DATA=/data/cjk/external/GenSAR/data
COMMON=$GENSAR_DATA/GenSAR_Faithful_20K_Strict/emb/bge-base-en-v1.5.pt
ANCHOR=stage3/data/gensar_faithful_20k_t0_2011q4/v13_anchor_full_p256/anchor_lookup.npz
REC_DRIFT=stage3/data/temporal_drift_20k_quarterly_v1/rec_item_drift.csv
OUT=stage3/data/temporal_r4c_representative
mkdir -p "$OUT"
cd "$ROOT"

for n in 3 5 9; do
  w="W$n"
  prev="W$((n - 1))"
  base="stage3/data/temporal_rolling_r3/$prev/mappings/selective_joint.json"
  checkpoint="stage3/data/temporal_rolling_r3/$prev/joint_adapter_checkpoint.pt"
  filtered="stage3/data/temporal_r4b_representative/$w/rec_drift_cooldown.csv"
  for variant in C0 C01 C1; do
    dir="$OUT/$w/$variant"
    mkdir -p "$dir"
    if [[ -f "$dir/rec_sticky/drift_selected_sticky_ratio_0p1/stats.json" ]]; then
      continue
    fi
    extra=(--deterministic-frozen-modules)
    [[ $variant == C01 ]] && extra+=(--transition-contrastive-weight 0.01 --contrastive-temperature 0.1)
    [[ $variant == C1 ]] && extra+=(--transition-contrastive-weight 0.1 --contrastive-temperature 0.1)
    CUDA_VISIBLE_DEVICES=7 python -u stage3/scripts/update_v13b_rec_adapter_selective.py \
      --checkpoint "$checkpoint" --t0-mapping "$base" \
      --data-dir "stage3/data/temporal_rolling_r3/$w/snapshot" \
      --common-embedding "$COMMON" \
      --updated-rec-embedding "stage3/artifacts/unisar_temporal_rolling_r3_$w/UniSAR_faithful.pt" \
      --t0-rec-embedding "stage3/artifacts/unisar_temporal_rolling_r3_$prev/UniSAR_faithful.pt" \
      --anchor-lookup "$ANCHOR" --drift-csv "$filtered" --drift-left "$prev" --drift-right "$w" \
      --replay-ratio 2 --output-dir "$dir/rec_adapter" --epochs 60 --batch-size 64 \
      --lr 0.0001 --weight-decay 0.0001 --topology-weight 0.5 --commit-weight 0.05 \
      --stability-weight 0.1 --outside-stability-weight 0.1 --patience 10 \
      "${extra[@]}" --device cuda:0 --seed 2026 >"$dir/adapter.log" 2>&1
    CUDA_VISIBLE_DEVICES=7 python stage3/scripts/materialize_v13b_temporal_rec_sid_sticky.py \
      --base-checkpoint "$checkpoint" --adapter-checkpoint "$dir/rec_adapter/rec_adapter_best.pt" \
      --base-operational-mapping "$base" --data-dir "stage3/data/temporal_rolling_r3/$w/snapshot" \
      --common-embedding "$COMMON" \
      --updated-rec-embedding "stage3/artifacts/unisar_temporal_rolling_r3_$w/UniSAR_faithful.pt" \
      --anchor-lookup "$ANCHOR" --drift-csv "$filtered" --drift-left "$prev" --drift-right "$w" \
      --switch-ratios 0.1 --output-dir "$dir/rec_sticky" --batch-size 1024 \
      --device cuda:0 --seed 2026 >"$dir/materialize.log"
  done
done
