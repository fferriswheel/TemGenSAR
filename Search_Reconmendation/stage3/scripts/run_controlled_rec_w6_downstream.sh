#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
GENSAR_DATA=/data/cjk/external/GenSAR/data
BASE=stage3/data/controlled_rec_drift_w3w6w9_v1/W6
QUARTERLY=stage3/data/gensar_faithful_20k_quarterly_controlled_rec_w3w6w9_v1
ITEM_IDS=stage3/data/gensar_faithful_20k_t0_2011q4/item_ids.jsonl
FIXED_MAPPING=stage3/data/temporal_joint_r2a_mappings/J0_fixed_sid_mapping.json
FIXED_EXPORT=stage3/data/temporal_joint_r2a_mappings/export
EXPORT=$BASE/downstream/export
POOL=$BASE/downstream/alignment_items.json
RUNS=$ROOT/stage3/runs/temporal_incremental
INITIAL=$RUNS/CRD_W5_Fixed_s2024
cd "$ROOT"

declare -A MAPPINGS=(
  [Fixed]="$FIXED_MAPPING"
  [Random]="$BASE/Random_sid/materialized/drift_selected_sticky_ratio_0p1/sid_mapping.json"
  [Support]="$BASE/Support_sid/materialized/drift_selected_sticky_ratio_0p1/sid_mapping.json"
  [V1]="$BASE/V1_sid/materialized/drift_selected_sticky_ratio_0p1/sid_mapping.json"
  [V2]="$BASE/V2_sid/materialized/drift_selected_sticky_ratio_0p1/sid_mapping.json"
)

declare -A NAMES=(
  [Fixed]=J0_fixed
  [Random]=CRD_W6_Random
  [Support]=CRD_W6_Support
  [V1]=CRD_W6_V1
  [V2]=CRD_W6_V2
)

is_complete() {
  local output=$1
  [[ -f "$output/model.safetensors" \
    && -f "$output/SeqRecWithSrcHis_sample_neg-999_beam-10.json" \
    && -f "$output/PerSrcWithRecHis_sample_neg-999_beam-10.json" ]]
}

prepare_shared_inputs() {
  mkdir -p "$EXPORT" "$(dirname "$POOL")"
  python stage3/scripts/build_priority_alignment_pool.py \
    --item-ids "$ITEM_IDS" \
    --priority-update-stats "$BASE/Random_sid/rec_adapter/update_stats.json" \
    --priority-update-stats "$BASE/Support_sid/rec_adapter/update_stats.json" \
    --priority-update-stats "$BASE/V1_sid/rec_adapter/update_stats.json" \
    --priority-update-stats "$BASE/V2_sid/rec_adapter/update_stats.json" \
    --size 1800 --seed 2024 --output "$POOL"

  for arm in Random Support V1 V2; do
    python stage3/scripts/export_sid_mapping_for_gensar.py \
      --mapping "${MAPPINGS[$arm]}" --item-ids "$ITEM_IDS" \
      --output-dir "$EXPORT" --name "${NAMES[$arm]}"
  done
}

prepare_arm() {
  local arm=$1
  local dataset="GenSAR_Faithful_20K_CRD_W6_${arm}"
  local semantic collaborate
  if [[ "$arm" == Fixed ]]; then
    semantic="$FIXED_EXPORT/J0_fixed_semantic.json"
    collaborate="$FIXED_EXPORT/J0_fixed_collaborate.json"
  else
    semantic="$EXPORT/${NAMES[$arm]}_semantic.json"
    collaborate="$EXPORT/${NAMES[$arm]}_collaborate.json"
  fi
  python stage3/scripts/materialize_temporal_incremental_gensar.py \
    --quarterly-dir "$QUARTERLY" --update-window W6 --eval-window W7 \
    --output-dir "$GENSAR_DATA/$dataset" \
    --semantic-sid "$semantic" --collaborate-sid "$collaborate" \
    --base-mapping "$FIXED_MAPPING" --current-mapping "${MAPPINGS[$arm]}" \
    --item-ids "$ITEM_IDS" --update-task both \
    --alignment-items "$POOL" --balanced-alignment --seed 2024 \
    >"stage3/runs/${dataset}.materialize.log"
}

verify_shared_protocol() {
  local reference="$GENSAR_DATA/GenSAR_Faithful_20K_CRD_W6_Fixed"
  local relative arm candidate
  for arm in Random Support V1 V2; do
    candidate="$GENSAR_DATA/GenSAR_Faithful_20K_CRD_W6_${arm}"
    for relative in \
      dataset/rec_train.pkl dataset/src_train.pkl \
      dataset/rec_val.pkl dataset/src_val.pkl \
      dataset/rec_test.pkl dataset/src_test.pkl \
      vocab/rec_item_set.pkl vocab/src_item_set.pkl; do
      cmp --silent "$reference/$relative" "$candidate/$relative" || {
        echo "Protocol mismatch for $arm: $relative" >&2
        exit 1
      }
    done
  done
  echo "[ok] all W6 arms use byte-identical train/val/test data and alignment item sets"
}

run_arm() {
  local arm=$1 gpu=$2 seed=${3:-2024}
  local experiment="CRD_W6_${arm}_s${seed}"
  local dataset="GenSAR_Faithful_20K_CRD_W6_${arm}"
  local output="$RUNS/$experiment"
  local semantic collaborate
  if is_complete "$output"; then
    echo "[skip] $experiment"
    return
  fi
  if [[ "$arm" == Fixed ]]; then
    semantic=J0_fixed_semantic.json
    collaborate=J0_fixed_collaborate.json
  else
    semantic="${NAMES[$arm]}_semantic.json"
    collaborate="${NAMES[$arm]}_collaborate.json"
  fi
  bash stage3/scripts/run_temporal_incremental_t5.sh "$gpu" \
    "$experiment" "$dataset" "$semantic" "$collaborate" \
    1800 "$seed" "$INITIAL"
  find "$output" -depth \
    \( -path "$output/checkpoint-*" -o -path "$output/checkpoint-*/*" \) -delete
}

if [[ ! -f "$INITIAL/model.safetensors" ]]; then
  echo "Missing completed W5 checkpoint: $INITIAL/model.safetensors" >&2
  exit 1
fi

prepare_shared_inputs
for arm in Fixed Random Support V1 V2; do prepare_arm "$arm"; done
verify_shared_protocol

pids=()
run_arm Fixed 4 & pids+=("$!")
run_arm Support 5 & pids+=("$!")
run_arm V1 6 & pids+=("$!")
run_arm V2 7 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
run_arm Random 4
