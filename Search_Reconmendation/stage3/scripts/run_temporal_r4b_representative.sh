#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
GENSAR_DATA=/data/cjk/external/GenSAR/data
COMMON=$GENSAR_DATA/GenSAR_Faithful_20K_Strict/emb/bge-base-en-v1.5.pt
ANCHOR=stage3/data/gensar_faithful_20k_t0_2011q4/v13_anchor_full_p256/anchor_lookup.npz
ITEM_IDS=stage3/data/gensar_faithful_20k_t0_2011q4/item_ids.jsonl
QUARTERLY=stage3/data/gensar_faithful_20k_quarterly_controlled_search_w1
REC_DRIFT=stage3/data/temporal_drift_20k_quarterly_v1/rec_item_drift.csv
FIXED=stage3/data/temporal_joint_r2a_mappings/J0_fixed_sid_mapping.json
W1_MAPPING=stage3/data/temporal_joint_r2a_mappings/JB_joint_sid_mapping.json
OUT=stage3/data/temporal_r4b_representative
mkdir -p "$OUT"
cd "$ROOT"

previous_mapping() {
  local n=$1
  if [[ $n -eq 2 ]]; then
    echo "$W1_MAPPING"
  else
    echo "stage3/data/temporal_rolling_r3/W$((n - 1))/mappings/selective_joint.json"
  fi
}

history_args() {
  local n=$1
  printf '%s\n' --history-mapping "$FIXED" --history-mapping "$W1_MAPPING"
  for ((i=2; i<n; i++)); do
    printf '%s\n' --history-mapping "stage3/data/temporal_rolling_r3/W$i/mappings/selective_joint.json"
  done
}

prepare_window() {
  local n=$1
  local gpu=$2
  local w="W$n"
  local prev="W$((n - 1))"
  local eval="W$((n + 1))"
  local dir="$OUT/$w"
  local prev_map
  prev_map=$(previous_mapping "$n")
  local current_search="stage3/data/temporal_rolling_r3/$w/mappings/selective_joint.json"
  local prev_joint="stage3/data/temporal_rolling_r3/$prev/joint_adapter_checkpoint.pt"
  mkdir -p "$dir"

  mapfile -t history < <(history_args "$n")
  python stage3/scripts/filter_temporal_selection_cooldown.py \
    --drift-csv "$REC_DRIFT" --left "$prev" --right "$w" --task rec \
    "${history[@]}" --cooldown-windows 2 \
    --output-csv "$dir/rec_drift_cooldown.csv" \
    --output-summary "$dir/rec_drift_cooldown.json" \
    >"$dir/filter.log"

  CUDA_VISIBLE_DEVICES="$gpu" python -u stage3/scripts/update_v13b_rec_adapter_selective.py \
    --checkpoint "$prev_joint" --t0-mapping "$prev_map" \
    --data-dir "stage3/data/temporal_rolling_r3/$w/snapshot" \
    --common-embedding "$COMMON" \
    --updated-rec-embedding "stage3/artifacts/unisar_temporal_rolling_r3_$w/UniSAR_faithful.pt" \
    --t0-rec-embedding "stage3/artifacts/unisar_temporal_rolling_r3_$prev/UniSAR_faithful.pt" \
    --anchor-lookup "$ANCHOR" --drift-csv "$dir/rec_drift_cooldown.csv" \
    --drift-left "$prev" --drift-right "$w" --replay-ratio 2 \
    --output-dir "$dir/rec_adapter" --epochs 60 --batch-size 64 --lr 0.0001 \
    --weight-decay 0.0001 --topology-weight 0.5 --commit-weight 0.05 \
    --stability-weight 0.1 --outside-stability-weight 0.1 --patience 10 \
    --device cuda:0 --seed 2026 >"$dir/rec_adapter.log" 2>&1

  CUDA_VISIBLE_DEVICES="$gpu" python stage3/scripts/materialize_v13b_temporal_rec_sid_sticky.py \
    --base-checkpoint "$prev_joint" --adapter-checkpoint "$dir/rec_adapter/rec_adapter_best.pt" \
    --base-operational-mapping "$prev_map" \
    --data-dir "stage3/data/temporal_rolling_r3/$w/snapshot" \
    --common-embedding "$COMMON" \
    --updated-rec-embedding "stage3/artifacts/unisar_temporal_rolling_r3_$w/UniSAR_faithful.pt" \
    --anchor-lookup "$ANCHOR" --drift-csv "$dir/rec_drift_cooldown.csv" \
    --drift-left "$prev" --drift-right "$w" --switch-ratios 0.1 \
    --output-dir "$dir/rec_sticky" --batch-size 1024 --device cuda:0 --seed 2026 \
    >"$dir/rec_sticky.log"

  python stage3/scripts/compose_temporal_task_mappings.py \
    --base-mapping "$prev_map" --search-mapping "$current_search" \
    --rec-mapping "$dir/rec_sticky/drift_selected_sticky_ratio_0p1/sid_mapping.json" \
    --project-task-fields --output "$dir/r4b_mapping.json" >"$dir/compose.log"
  python stage3/scripts/export_sid_mapping_for_gensar.py \
    --mapping "$dir/r4b_mapping.json" --item-ids "$ITEM_IDS" \
    --output-dir "$dir/export" --name "r4b_${w,,}" >"$dir/export.log"
  python stage3/scripts/build_rolling_alignment_pools.py \
    --transition R4B "$prev_map" "$dir/r4b_mapping.json" \
    --output-dir "$dir/alignment" >"$dir/alignment.log"

  local search_count rec_count budget
  read -r search_count rec_count < <(python - "$dir/alignment/alignment_summary.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
print(x["search_alignment_items"], x["rec_alignment_items"])
PY
)
  budget=$((8 * (search_count > rec_count ? search_count : rec_count)))
  ((budget < 512)) && budget=512
  ((budget > 4096)) && budget=4096
  printf '%s\n' "$budget" >"$dir/dynamic_alignment_budget.txt"

  for variant in B1 B2; do
    extra=()
    search_items="stage3/data/temporal_rolling_r3/$w/alignment/search_alignment_items.json"
    rec_items="stage3/data/temporal_rolling_r3/$w/alignment/rec_alignment_items.json"
    if [[ $variant == B2 ]]; then
      extra=(--balanced-alignment)
      search_items="$dir/alignment/search_alignment_items.json"
      rec_items="$dir/alignment/rec_alignment_items.json"
    fi
    python stage3/scripts/materialize_temporal_incremental_gensar.py \
      --quarterly-dir "$QUARTERLY" --update-window "$w" --eval-window "$eval" \
      --output-dir "$GENSAR_DATA/GenSAR_Faithful_20K_R4B_${w}_${variant}" \
      --semantic-sid "$dir/export/r4b_${w,,}_semantic.json" \
      --collaborate-sid "$dir/export/r4b_${w,,}_collaborate.json" \
      --base-mapping "$prev_map" --current-mapping "$dir/r4b_mapping.json" \
      --item-ids "$ITEM_IDS" --update-task both \
      --search-alignment-items "$search_items" --rec-alignment-items "$rec_items" \
      "${extra[@]}" --seed 2024 >"$dir/materialize_${variant}.log"
  done
}

for n in 3 5 9; do
  prepare_window "$n" 7 >"$OUT/W${n}_prepare.log" 2>&1
done

for variant in B1 B2; do
  for n in 3 5 9; do
    budget=1800
    [[ $variant == B2 ]] && budget=$(<"$OUT/W$n/dynamic_alignment_budget.txt")
    bash stage3/scripts/run_temporal_incremental_t5.sh 7 \
      "R4B_${variant}_W${n}_s2024" "GenSAR_Faithful_20K_R4B_W${n}_${variant}" \
      "r4b_w${n}_semantic.json" "r4b_w${n}_collaborate.json" "$budget" 2024 \
      "$ROOT/stage3/runs/temporal_incremental/R3_W$((n - 1))_Selective_s2024" \
      >"$OUT/W${n}/${variant}_downstream.log" 2>&1
  done
done
