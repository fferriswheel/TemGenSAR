#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
GENSAR_DATA=/data/cjk/external/GenSAR/data
QUARTERLY=stage3/data/gensar_faithful_20k_quarterly_controlled_search_w1
ITEM_IDS=stage3/data/gensar_faithful_20k_t0_2011q4/item_ids.jsonl
OUT=stage3/data/temporal_r4c_representative
cd "$ROOT"

prepare() {
  local n=$1 variant=$2
  local w="W$n" prev="W$((n - 1))" eval="W$((n + 1))"
  local dir="$OUT/$w/$variant"
  local previous="stage3/data/temporal_rolling_r3/$prev/mappings/selective_joint.json"
  local search="stage3/data/temporal_rolling_r3/$w/mappings/selective_joint.json"
  local rec="$dir/rec_sticky/drift_selected_sticky_ratio_0p1/sid_mapping.json"
  python stage3/scripts/compose_temporal_task_mappings.py \
    --base-mapping "$previous" --search-mapping "$search" --rec-mapping "$rec" \
    --project-task-fields --output "$dir/joint_mapping.json" >"$dir/compose.log"
  python stage3/scripts/export_sid_mapping_for_gensar.py \
    --mapping "$dir/joint_mapping.json" --item-ids "$ITEM_IDS" --output-dir "$dir/export" \
    --name "r4c_${w,,}_${variant,,}" >"$dir/export.log"
  python stage3/scripts/materialize_temporal_incremental_gensar.py \
    --quarterly-dir "$QUARTERLY" --update-window "$w" --eval-window "$eval" \
    --output-dir "$GENSAR_DATA/GenSAR_Faithful_20K_R4C_${w}_${variant}" \
    --semantic-sid "$dir/export/r4c_${w,,}_${variant,,}_semantic.json" \
    --collaborate-sid "$dir/export/r4c_${w,,}_${variant,,}_collaborate.json" \
    --base-mapping "$previous" --current-mapping "$dir/joint_mapping.json" \
    --item-ids "$ITEM_IDS" --update-task both \
    --search-alignment-items "stage3/data/temporal_rolling_r3/$w/alignment/search_alignment_items.json" \
    --rec-alignment-items "stage3/data/temporal_rolling_r3/$w/alignment/rec_alignment_items.json" \
    --seed 2024 >"$dir/materialize_gensar.log"
}

for w in 3 9; do
  for variant in C0 C01; do prepare "$w" "$variant"; done
done

pids=()
for spec in 3:C0:4 3:C01:5 9:C0:6 9:C01:7; do
  n=${spec%%:*}; rest=${spec#*:}; variant=${rest%%:*}; gpu=${rest##*:}
  name="r4c_w${n}_${variant,,}"
  bash stage3/scripts/run_temporal_incremental_t5.sh "$gpu" \
    "R4C_${variant}_W${n}_s2024" "GenSAR_Faithful_20K_R4C_W${n}_${variant}" \
    "${name}_semantic.json" "${name}_collaborate.json" 1800 2024 \
    "$ROOT/stage3/runs/temporal_incremental/R3_W$((n - 1))_Selective_s2024" \
    >"$OUT/W${n}/${variant}/downstream.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
