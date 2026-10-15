#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
SOURCE=stage3/data/gensar_faithful_20k_controlled_rec_w3w6w9_v1
BASE=stage3/data/controlled_rec_drift_w3w6w9_v1
ARTIFACTS=stage3/artifacts
T0_SEARCH_END=1325376000
cd "$ROOT"

export_snapshot() {
  local output=$1 cutoff=$2
  if [[ -f "$output/temporal_snapshot_stats.json" ]]; then
    return
  fi
  python stage3/scripts/export_temporal_snapshot_data.py \
    --source-dir "$SOURCE" --output-dir "$output" \
    --end-exclusive "$cutoff" --search-end-exclusive "$T0_SEARCH_END"
}

train_rec_view() {
  local data=$1 output=$2 init_checkpoint=$3 init_data=$4 start=$5
  if [[ -f "$output/best.pt" && -f "$output/UniSAR_faithful.pt" ]]; then
    echo "[skip] $output"
    return
  fi
  /usr/bin/time -v env CUDA_VISIBLE_DEVICES=5 python -u \
    stage3/scripts/train_unisar_faithful_embeddings.py \
    --data-dir "$data" --output-dir "$output" \
    --dim 64 --heads 4 --layers 1 --dropout 0.1 --max-history 30 \
    --num-negatives 20 --batch-size 512 --epochs 3 --lr 0.0001 \
    --weight-decay 0.00001 --alpha 0.1 --beta 0.1 \
    --max-valid-samples 5000 --workers 4 --seed 2026 --device cuda:0 \
    --zero-unseen-items --init-checkpoint "$init_checkpoint" \
    --init-data-dir "$init_data" --update-task rec \
    --update-start-timestamp "$start" --freeze-for-rec-update
}

export_snapshot "$BASE/W4/snapshot" 1356998400
train_rec_view \
  "$BASE/W4/snapshot" "$ARTIFACTS/unisar_controlled_rec_W4" \
  "$ARTIFACTS/unisar_controlled_rec_W3/best.pt" \
  "$BASE/W3/snapshot" 1349049600

export_snapshot "$BASE/W5/snapshot" 1364774400
train_rec_view \
  "$BASE/W5/snapshot" "$ARTIFACTS/unisar_controlled_rec_W5" \
  "$ARTIFACTS/unisar_controlled_rec_W4/best.pt" \
  "$BASE/W4/snapshot" 1356998400

export_snapshot "$BASE/W6/snapshot" 1372636800
train_rec_view \
  "$BASE/W6/snapshot" "$ARTIFACTS/unisar_controlled_rec_W6" \
  "$ARTIFACTS/unisar_controlled_rec_W5/best.pt" \
  "$BASE/W5/snapshot" 1364774400
