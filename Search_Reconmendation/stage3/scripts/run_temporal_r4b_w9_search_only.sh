#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
GENSAR_DATA=/data/cjk/external/GenSAR/data
DIR=stage3/data/temporal_r4b_representative/W9_search_only
PREVIOUS=stage3/data/temporal_rolling_r3/W8/mappings/selective_joint.json
CURRENT=stage3/data/temporal_rolling_r3/W9/mappings/selective_joint.json
ITEM_IDS=stage3/data/gensar_faithful_20k_t0_2011q4/item_ids.jsonl
QUARTERLY=stage3/data/gensar_faithful_20k_quarterly_controlled_search_w1
mkdir -p "$DIR"
cd "$ROOT"

python stage3/scripts/compose_temporal_task_mappings.py \
  --base-mapping "$PREVIOUS" --search-mapping "$CURRENT" --rec-mapping "$PREVIOUS" \
  --project-task-fields --output "$DIR/mapping.json" >"$DIR/compose.log"
python stage3/scripts/export_sid_mapping_for_gensar.py \
  --mapping "$DIR/mapping.json" --item-ids "$ITEM_IDS" --output-dir "$DIR/export" \
  --name r4b_w9_search_only >"$DIR/export.log"

python stage3/scripts/materialize_temporal_incremental_gensar.py \
  --quarterly-dir "$QUARTERLY" --update-window W9 --eval-window W10 \
  --output-dir "$GENSAR_DATA/GenSAR_Faithful_20K_R4B_W9_SearchOnly" \
  --semantic-sid "$DIR/export/r4b_w9_search_only_semantic.json" \
  --collaborate-sid "$DIR/export/r4b_w9_search_only_collaborate.json" \
  --base-mapping "$PREVIOUS" --current-mapping "$DIR/mapping.json" \
  --item-ids "$ITEM_IDS" --update-task both \
  --search-alignment-items stage3/data/temporal_rolling_r3/W9/alignment/search_alignment_items.json \
  --rec-alignment-items stage3/data/temporal_rolling_r3/W9/alignment/rec_alignment_items.json \
  --seed 2024 >"$DIR/materialize.log"

bash stage3/scripts/run_temporal_incremental_t5.sh 7 \
  R4B_SearchOnly_W9_s2024 GenSAR_Faithful_20K_R4B_W9_SearchOnly \
  r4b_w9_search_only_semantic.json r4b_w9_search_only_collaborate.json \
  1800 2024 "$ROOT/stage3/runs/temporal_incremental/R3_W8_Selective_s2024" \
  >"$DIR/downstream.log" 2>&1
