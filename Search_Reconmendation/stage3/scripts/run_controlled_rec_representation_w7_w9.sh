#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
SOURCE=stage3/data/gensar_faithful_20k_controlled_rec_w3w6w9_v1
BASE=stage3/data/controlled_rec_drift_w3w6w9_v1
ARTIFACTS=stage3/artifacts
T0_SEARCH_END=1325376000
GPU=${1:-5}
cd "$ROOT"

export_snapshot() {
  local output=$1 cutoff=$2
  if [[ -f "$output/temporal_snapshot_stats.json" ]]; then
    echo "[skip] snapshot $output"
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
  /usr/bin/time -v env CUDA_VISIBLE_DEVICES="$GPU" python -u \
    stage3/scripts/train_unisar_faithful_embeddings.py \
    --data-dir "$data" --output-dir "$output" \
    --dim 64 --heads 4 --layers 1 --dropout 0.1 --max-history 30 \
    --num-negatives 20 --batch-size 512 --epochs 3 --lr 0.0001 \
    --weight-decay 0.00001 --alpha 0.1 --beta 0.1 \
    --max-valid-samples 5000 --workers 4 --seed 2026 --device cuda:0 \
    --zero-unseen-items --init-checkpoint "$init_checkpoint" \
    --init-data-dir "$init_data" --update-task rec \
    --update-start-timestamp "$start" --freeze-for-rec-update \
    >"$output.train.log" 2>&1
}

export_snapshot "$BASE/W7/snapshot" 1380585600
train_rec_view \
  "$BASE/W7/snapshot" "$ARTIFACTS/unisar_controlled_rec_W7" \
  "$ARTIFACTS/unisar_controlled_rec_W6/best.pt" \
  "$BASE/W6/snapshot" 1372636800

export_snapshot "$BASE/W8/snapshot" 1388534400
train_rec_view \
  "$BASE/W8/snapshot" "$ARTIFACTS/unisar_controlled_rec_W8" \
  "$ARTIFACTS/unisar_controlled_rec_W7/best.pt" \
  "$BASE/W7/snapshot" 1380585600

export_snapshot "$BASE/W9/snapshot" 1396310400
train_rec_view \
  "$BASE/W9/snapshot" "$ARTIFACTS/unisar_controlled_rec_W9" \
  "$ARTIFACTS/unisar_controlled_rec_W8/best.pt" \
  "$BASE/W8/snapshot" 1388534400
