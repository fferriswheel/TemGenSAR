#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
GENSAR_DATA=/data/cjk/external/GenSAR/data
BASE=stage3/data/controlled_rec_drift_w3w6w9_v1/W6
QUARTERLY=stage3/data/gensar_faithful_20k_quarterly_controlled_rec_w3w6w9_v1
ITEM_IDS=stage3/data/gensar_faithful_20k_t0_2011q4/item_ids.jsonl
FIXED=stage3/data/temporal_joint_r2a_mappings/J0_fixed_sid_mapping.json
CURRENT=$BASE/V3_sid/materialized/drift_selected_sticky_ratio_0p1/sid_mapping.json
SEMANTIC=$BASE/downstream/export/CRD_W6_V3_semantic.json
COLLABORATE=$BASE/downstream/export/CRD_W6_V3_collaborate.json
POOL=$BASE/alignment_repeat_v2/common_alignment_items.json
SCHEDULE_ROOT=$BASE/alignment_repeat_v2/V3
INITIAL=$ROOT/stage3/runs/temporal_incremental/CRD_W5_Fixed_s2024
RUNS=$ROOT/stage3/runs/temporal_incremental
cd "$ROOT"

is_complete() {
  local output=$1
  [[ -f "$output/model.safetensors" \
    && -f "$output/SeqRecWithSrcHis_sample_neg-999_beam-10.json" \
    && -f "$output/PerSrcWithRecHis_sample_neg-999_beam-10.json" ]]
}

build_schedule() {
  local factor=$1 output="$SCHEDULE_ROOT/R${factor}"
  python stage3/scripts/build_changed_item_alignment_schedule.py \
    --base-pool "$POOL" --base-mapping "$FIXED" --current-mapping "$CURRENT" \
    --protected-update-stats "$BASE/V3_sid/rec_adapter/update_stats.json" \
    --protected-update-stats "$BASE/Support_sid/rec_adapter/update_stats.json" \
    --protected-update-stats "$BASE/Random_sid/rec_adapter/update_stats.json" \
    --repeat-factor "$factor" --seed 2024 --output-dir "$output"
}

build_common_pool() {
  python stage3/scripts/build_priority_alignment_pool.py \
    --item-ids "$ITEM_IDS" \
    --priority-update-stats "$BASE/V3_sid/rec_adapter/update_stats.json" \
    --priority-update-stats "$BASE/Support_sid/rec_adapter/update_stats.json" \
    --priority-update-stats "$BASE/Random_sid/rec_adapter/update_stats.json" \
    --size 1800 --seed 2024 --output "$POOL"
}

prepare_arm() {
  local factor=$1
  local schedule="$SCHEDULE_ROOT/R${factor}"
  local dataset="GenSAR_Faithful_20K_CRD_W6_V3_A1_R${factor}"
  python stage3/scripts/materialize_temporal_incremental_gensar.py \
    --quarterly-dir "$QUARTERLY" --update-window W6 --eval-window W7 \
    --output-dir "$GENSAR_DATA/$dataset" \
    --semantic-sid "$SEMANTIC" --collaborate-sid "$COLLABORATE" \
    --base-mapping "$FIXED" --current-mapping "$CURRENT" \
    --item-ids "$ITEM_IDS" --update-task both \
    --search-alignment-items "$schedule/search_alignment_schedule.json" \
    --rec-alignment-items "$schedule/rec_alignment_schedule.json" \
    --preserve-alignment-duplicates --balanced-alignment --seed 2024 \
    >"$schedule/materialize.log"
}

verify_arm() {
  local factor=$1
  local reference="$GENSAR_DATA/GenSAR_Faithful_20K_CRD_W6_V3"
  local candidate="$GENSAR_DATA/GenSAR_Faithful_20K_CRD_W6_V3_A1_R${factor}"
  # Only the alignment schedules may differ from the original W6 dataset.
  for relative in \
    dataset/rec_train.pkl dataset/src_train.pkl \
    dataset/rec_val.pkl dataset/src_val.pkl \
    dataset/rec_test.pkl dataset/src_test.pkl; do
    cmp --silent "$reference/$relative" "$candidate/$relative" || {
      echo "Protocol mismatch for R${factor}: $relative" >&2
      exit 1
    }
  done
  cmp --silent "$SEMANTIC" "$candidate/emb/CRD_W6_V3_semantic.json"
  cmp --silent "$COLLABORATE" "$candidate/emb/CRD_W6_V3_collaborate.json"
  python - "$factor" "$candidate" <<'PY'
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

factor = int(sys.argv[1])
root = Path(sys.argv[2])
rec = pickle.load((root / "vocab/rec_item_set.pkl").open("rb"))
search = pickle.load((root / "vocab/src_item_set.pkl").open("rb"))
manifest = json.loads((root / "incremental_manifest.json").read_text())
if len(rec) != 1800 or len(search) != 1800:
    raise SystemExit("alignment occurrence budget changed")
if len(set(search)) != 1800:
    raise SystemExit("Search alignment schedule must remain unique")
counts = Counter(rec)
repeated = [value for value in counts.values() if value == factor]
expected_repeated = 1800 if factor == 1 else 33
if len(repeated) != expected_repeated or max(counts.values()) != factor:
    raise SystemExit(f"unexpected Rec schedule distribution: {Counter(counts.values())}")
if not manifest["alignment_duplicates_preserved"]:
    raise SystemExit("materializer did not preserve schedule duplicates")
print(json.dumps({
    "factor": factor,
    "rec_occurrences": len(rec),
    "rec_unique": len(set(rec)),
    "search_occurrences": len(search),
    "search_unique": len(set(search)),
    "repeated_recoded_items": len(repeated),
}))
PY
}

run_arm() {
  local factor=$1 gpu=$2
  local experiment="CRD_W6_V3_A1_R${factor}_s2024"
  local dataset="GenSAR_Faithful_20K_CRD_W6_V3_A1_R${factor}"
  local output="$RUNS/$experiment"
  if is_complete "$output"; then
    echo "[skip] $experiment"
    return
  fi
  bash stage3/scripts/run_temporal_incremental_t5.sh "$gpu" \
    "$experiment" "$dataset" CRD_W6_V3_semantic.json CRD_W6_V3_collaborate.json \
    1800 2024 "$INITIAL"
  find "$output" -depth \
    \( -path "$output/checkpoint-*" -o -path "$output/checkpoint-*/*" \) -delete
}

build_common_pool
for factor in 1 2 4 8; do
  build_schedule "$factor"
  prepare_arm "$factor"
  verify_arm "$factor"
done

pids=()
run_arm 1 4 & pids+=("$!")
run_arm 2 5 & pids+=("$!")
run_arm 4 6 & pids+=("$!")
run_arm 8 7 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
