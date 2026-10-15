#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
GENSAR_DATA=/data/cjk/external/GenSAR/data
BASE=stage3/data/controlled_rec_drift_w3w6w9_v1
QUARTERLY=stage3/data/gensar_faithful_20k_quarterly_controlled_rec_w3w6w9_v1
ITEM_IDS=stage3/data/gensar_faithful_20k_t0_2011q4/item_ids.jsonl
FIXED=stage3/data/temporal_joint_r2a_mappings/J0_fixed_sid_mapping.json
FIXED_EXPORT=stage3/data/temporal_joint_r2a_mappings/export
RUNS=$ROOT/stage3/runs/temporal_incremental
INITIAL=$RUNS/CRD_W8_Fixed_s2024
BUDGET=${1:?usage: $0 BUDGET}
OUT=$BASE/W9/downstream_K${BUDGET}
EXPORT=$OUT/export
POOL=$OUT/alignment_items.json
cd "$ROOT"

mapping_for() {
  local method=$1
  if [[ "$method" == Fixed ]]; then
    printf '%s\n' "$FIXED"
  else
    printf '%s\n' "$BASE/W9/${method}_K${BUDGET}_sid/materialized/drift_selected_sticky_ratio_0p1/sid_mapping.json"
  fi
}

stats_for() {
  printf '%s\n' "$BASE/W9/${1}_K${BUDGET}_sid/rec_adapter/update_stats.json"
}

is_complete() {
  local output=$1
  [[ -f "$output/model.safetensors" \
    && -f "$output/SeqRecWithSrcHis_sample_neg-999_beam-10.json" \
    && -f "$output/PerSrcWithRecHis_sample_neg-999_beam-10.json" ]]
}

mkdir -p "$EXPORT"
python stage3/scripts/build_priority_alignment_pool.py \
  --item-ids "$ITEM_IDS" \
  --priority-update-stats "$(stats_for V3)" \
  --priority-update-stats "$(stats_for Support)" \
  --priority-update-stats "$(stats_for Random)" \
  --size 1800 --seed 2024 --output "$POOL"

for method in V3 Support Random; do
  name="CRD_W9_${method}_K${BUDGET}"
  python stage3/scripts/export_sid_mapping_for_gensar.py \
    --mapping "$(mapping_for "$method")" --item-ids "$ITEM_IDS" \
    --output-dir "$EXPORT" --name "$name"
done

prepare_arm() {
  local method=$1 mapping name dataset semantic collaborate
  mapping=$(mapping_for "$method")
  if [[ "$method" == Fixed ]]; then
    name=J0_fixed
    dataset="GenSAR_Faithful_20K_CRD_W9_Fixed_K${BUDGET}"
    semantic="$FIXED_EXPORT/J0_fixed_semantic.json"
    collaborate="$FIXED_EXPORT/J0_fixed_collaborate.json"
  else
    name="CRD_W9_${method}_K${BUDGET}"
    dataset="GenSAR_Faithful_20K_${name}"
    semantic="$EXPORT/${name}_semantic.json"
    collaborate="$EXPORT/${name}_collaborate.json"
  fi
  python stage3/scripts/materialize_temporal_incremental_gensar.py \
    --quarterly-dir "$QUARTERLY" --update-window W9 --eval-window W10 \
    --output-dir "$GENSAR_DATA/$dataset" \
    --semantic-sid "$semantic" --collaborate-sid "$collaborate" \
    --base-mapping "$FIXED" --current-mapping "$mapping" \
    --item-ids "$ITEM_IDS" --update-task both \
    --alignment-items "$POOL" --balanced-alignment --seed 2024 \
    >"$OUT/${name}.materialize.log"
}

for method in Fixed V3 Support Random; do prepare_arm "$method"; done

reference="$GENSAR_DATA/GenSAR_Faithful_20K_CRD_W9_Fixed_K${BUDGET}"
for method in V3 Support Random; do
  candidate="$GENSAR_DATA/GenSAR_Faithful_20K_CRD_W9_${method}_K${BUDGET}"
  for relative in dataset/rec_train.pkl dataset/src_train.pkl dataset/rec_val.pkl \
    dataset/src_val.pkl dataset/rec_test.pkl dataset/src_test.pkl \
    vocab/rec_item_set.pkl vocab/src_item_set.pkl; do
    cmp --silent "$reference/$relative" "$candidate/$relative" || {
      echo "Protocol mismatch for $method: $relative" >&2
      exit 1
    }
  done
done
echo "[ok] W9 arms use identical data and alignment item sets"

until is_complete "$INITIAL"; do
  echo "[wait] common W8 Fixed checkpoint is not complete"
  sleep 30
done

run_arm() {
  local method=$1 gpu=$2 name dataset semantic collaborate experiment output
  if [[ "$method" == Fixed ]]; then
    experiment="CRD_W9_Fixed_K${BUDGET}_s2024"
    dataset="GenSAR_Faithful_20K_CRD_W9_Fixed_K${BUDGET}"
    semantic=J0_fixed_semantic.json
    collaborate=J0_fixed_collaborate.json
  else
    name="CRD_W9_${method}_K${BUDGET}"
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

pids=()
run_arm Fixed 4 & pids+=("$!")
run_arm V3 5 & pids+=("$!")
run_arm Support 6 & pids+=("$!")
run_arm Random 7 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
