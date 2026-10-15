#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
GENSAR_DATA=/data/cjk/external/GenSAR/data
QUARTERLY=stage3/data/gensar_faithful_20k_quarterly_controlled_search_w1
ITEM_IDS=stage3/data/gensar_faithful_20k_t0_2011q4/item_ids.jsonl
OUT=stage3/data/temporal_r4c_representative
cd "$ROOT"

prepare() {
  local variant=$1
  local dir="$OUT/W5/$variant"
  local previous=stage3/data/temporal_rolling_r3/W4/mappings/selective_joint.json
  local search=stage3/data/temporal_rolling_r3/W5/mappings/selective_joint.json
  local rec="$dir/rec_sticky/drift_selected_sticky_ratio_0p1/sid_mapping.json"

  python stage3/scripts/compose_temporal_task_mappings.py \
    --base-mapping "$previous" --search-mapping "$search" --rec-mapping "$rec" \
    --project-task-fields --output "$dir/joint_mapping.json" >"$dir/compose.log"
  python stage3/scripts/export_sid_mapping_for_gensar.py \
    --mapping "$dir/joint_mapping.json" --item-ids "$ITEM_IDS" --output-dir "$dir/export" \
    --name "r4c_w5_${variant,,}" >"$dir/export.log"
  python stage3/scripts/materialize_temporal_incremental_gensar.py \
    --quarterly-dir "$QUARTERLY" --update-window W5 --eval-window W6 \
    --output-dir "$GENSAR_DATA/GenSAR_Faithful_20K_R4C_W5_${variant}" \
    --semantic-sid "$dir/export/r4c_w5_${variant,,}_semantic.json" \
    --collaborate-sid "$dir/export/r4c_w5_${variant,,}_collaborate.json" \
    --base-mapping "$previous" --current-mapping "$dir/joint_mapping.json" \
    --item-ids "$ITEM_IDS" --update-task both \
    --search-alignment-items stage3/data/temporal_rolling_r3/W5/alignment/search_alignment_items.json \
    --rec-alignment-items stage3/data/temporal_rolling_r3/W5/alignment/rec_alignment_items.json \
    --seed 2024 >"$dir/materialize_gensar.log"
}

for variant in C0 C01; do prepare "$variant"; done

pids=()
for spec in C0:4 C01:5; do
  variant=${spec%%:*}
  gpu=${spec##*:}
  bash stage3/scripts/run_temporal_incremental_t5.sh "$gpu" \
    "R4C_${variant}_W5_s2024" "GenSAR_Faithful_20K_R4C_W5_${variant}" \
    "r4c_w5_${variant,,}_semantic.json" "r4c_w5_${variant,,}_collaborate.json" 1800 2024 \
    "$ROOT/stage3/runs/temporal_incremental/R3_W4_Selective_s2024" \
    >"$OUT/W5/$variant/downstream.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
