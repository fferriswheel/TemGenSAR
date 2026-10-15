#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
GENSAR_DATA=/data/cjk/external/GenSAR/data
BASE=stage3/data/controlled_rec_drift_w3w6w9_v1/W6
QUARTERLY=stage3/data/gensar_faithful_20k_quarterly_controlled_rec_w3w6w9_v1
ITEM_IDS=stage3/data/gensar_faithful_20k_t0_2011q4/item_ids.jsonl
FIXED=stage3/data/temporal_joint_r2a_mappings/J0_fixed_sid_mapping.json
FIXED_EXPORT=stage3/data/temporal_joint_r2a_mappings/export
OUT=$BASE/budget_downstream
EXPORT=$OUT/export
POOL=$OUT/alignment_items.json
RUNS=$ROOT/stage3/runs/temporal_incremental
INITIAL=$RUNS/CRD_W5_Fixed_s2024
cd "$ROOT"

mapping_for() {
  local policy=$1 budget=$2
  if [[ "$policy" == Fixed ]]; then
    printf '%s\n' "$FIXED"
  elif [[ "$budget" == 51 ]]; then
    printf '%s\n' "$BASE/${policy}_sid/materialized/drift_selected_sticky_ratio_0p1/sid_mapping.json"
  else
    printf '%s\n' "$BASE/Budget_${policy}_K${budget}_sid/materialized/drift_selected_sticky_ratio_0p1/sid_mapping.json"
  fi
}

stats_for() {
  local policy=$1 budget=$2
  if [[ "$budget" == 51 ]]; then
    printf '%s\n' "$BASE/${policy}_sid/rec_adapter/update_stats.json"
  else
    printf '%s\n' "$BASE/Budget_${policy}_K${budget}_sid/rec_adapter/update_stats.json"
  fi
}

is_complete() {
  local output=$1
  [[ -f "$output/model.safetensors" \
    && -f "$output/SeqRecWithSrcHis_sample_neg-999_beam-10.json" \
    && -f "$output/PerSrcWithRecHis_sample_neg-999_beam-10.json" ]]
}

prepare_inputs() {
  mkdir -p "$EXPORT"
  python stage3/scripts/build_priority_alignment_pool.py \
    --item-ids "$ITEM_IDS" \
    --priority-update-stats "$(stats_for V3 200)" \
    --priority-update-stats "$(stats_for Support 200)" \
    --size 1800 --seed 2024 --output "$POOL"
  local policy budget mapping name
  for policy in V3 Support; do
    for budget in 25 51 100 200; do
      mapping=$(mapping_for "$policy" "$budget")
      name="CRD_W6_Budget_${policy}_K${budget}"
      python stage3/scripts/export_sid_mapping_for_gensar.py \
        --mapping "$mapping" --item-ids "$ITEM_IDS" \
        --output-dir "$EXPORT" --name "$name"
    done
  done
}

prepare_arm() {
  local policy=$1 budget=$2 mapping name dataset semantic collaborate
  mapping=$(mapping_for "$policy" "$budget")
  if [[ "$policy" == Fixed ]]; then
    name=J0_fixed
    dataset=GenSAR_Faithful_20K_CRD_W6_Budget_Fixed
    semantic="$FIXED_EXPORT/J0_fixed_semantic.json"
    collaborate="$FIXED_EXPORT/J0_fixed_collaborate.json"
  else
    name="CRD_W6_Budget_${policy}_K${budget}"
    dataset="GenSAR_Faithful_20K_CRD_W6_Budget_${policy}_K${budget}"
    semantic="$EXPORT/${name}_semantic.json"
    collaborate="$EXPORT/${name}_collaborate.json"
  fi
  python stage3/scripts/materialize_temporal_incremental_gensar.py \
    --quarterly-dir "$QUARTERLY" --update-window W6 --eval-window W7 \
    --output-dir "$GENSAR_DATA/$dataset" \
    --semantic-sid "$semantic" --collaborate-sid "$collaborate" \
    --base-mapping "$FIXED" --current-mapping "$mapping" \
    --item-ids "$ITEM_IDS" --update-task both \
    --alignment-items "$POOL" --balanced-alignment --seed 2024 \
    >"$OUT/${name}.materialize.log"
}

verify_protocol() {
  local reference="$GENSAR_DATA/GenSAR_Faithful_20K_CRD_W6_Budget_Fixed"
  local policy budget candidate relative
  for policy in V3 Support; do
    for budget in 25 51 100 200; do
      candidate="$GENSAR_DATA/GenSAR_Faithful_20K_CRD_W6_Budget_${policy}_K${budget}"
      for relative in \
        dataset/rec_train.pkl dataset/src_train.pkl \
        dataset/rec_val.pkl dataset/src_val.pkl \
        dataset/rec_test.pkl dataset/src_test.pkl \
        vocab/rec_item_set.pkl vocab/src_item_set.pkl; do
        cmp --silent "$reference/$relative" "$candidate/$relative" || {
          echo "Protocol mismatch for ${policy} K=${budget}: $relative" >&2
          exit 1
        }
      done
    done
  done
  echo "[ok] budget arms use identical data and alignment item sets"
}

run_arm() {
  local policy=$1 budget=$2 gpu=$3 name dataset semantic collaborate experiment output
  if [[ "$policy" == Fixed ]]; then
    experiment=CRD_W6_Budget_Fixed_s2024
    dataset=GenSAR_Faithful_20K_CRD_W6_Budget_Fixed
    semantic=J0_fixed_semantic.json
    collaborate=J0_fixed_collaborate.json
  else
    name="CRD_W6_Budget_${policy}_K${budget}"
    experiment="${name}_s2024"
    dataset="GenSAR_Faithful_20K_${name}"
    semantic="${name}_semantic.json"
    collaborate="${name}_collaborate.json"
  fi
  output="$RUNS/$experiment"
  if is_complete "$output"; then
    echo "[skip] $experiment"
    return
  fi
  bash stage3/scripts/run_temporal_incremental_t5.sh "$gpu" \
    "$experiment" "$dataset" "$semantic" "$collaborate" \
    1800 2024 "$INITIAL"
  find "$output" -depth \
    \( -path "$output/checkpoint-*" -o -path "$output/checkpoint-*/*" \) -delete
}

prepare_inputs
prepare_arm Fixed 0
for policy in V3 Support; do
  for budget in 25 51 100 200; do prepare_arm "$policy" "$budget"; done
done
verify_protocol

pids=()
run_arm Fixed 0 4 & pids+=("$!")
run_arm V3 25 5 & pids+=("$!")
run_arm V3 51 6 & pids+=("$!")
run_arm V3 100 7 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

pids=()
run_arm V3 200 4 & pids+=("$!")
run_arm Support 25 5 & pids+=("$!")
run_arm Support 51 6 & pids+=("$!")
run_arm Support 100 7 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

run_arm Support 200 4
