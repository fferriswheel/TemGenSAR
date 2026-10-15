#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
GENSAR_DATA=/data/cjk/external/GenSAR/data
cd "$ROOT"

START_WINDOW=${1:-3}
END_WINDOW=${2:-9}
COMMON=/data/cjk/external/GenSAR/data/GenSAR_Faithful_20K_Strict/emb/bge-base-en-v1.5.pt
ANCHOR=stage3/data/gensar_faithful_20k_t0_2011q4/v13_anchor_full_p256/anchor_lookup.npz
ITEM_IDS=stage3/data/gensar_faithful_20k_t0_2011q4/item_ids.jsonl
SOURCE=stage3/data/gensar_faithful_20k_controlled_search_w1
QUARTERLY=stage3/data/gensar_faithful_20k_quarterly_controlled_search_w1
REC_DRIFT=stage3/data/temporal_drift_20k_quarterly_v1/rec_item_drift.csv
SEARCH_DRIFT=stage3/data/temporal_search_item_drift_controlled_w1/search_item_drift.csv

window_end() {
  python - "$1" <<'PY'
import json, sys
x = json.load(open("stage3/data/gensar_faithful_20k_quarterly_controlled_search_w1/stats.json"))
print(x["windows"][f"W{sys.argv[1]}"]["end_exclusive"])
PY
}

run_step() {
  local marker=$1
  shift
  if [[ -f "$marker" ]]; then
    echo "[skip] $marker"
    return
  fi
  mkdir -p "$(dirname "$marker")"
  "$@"
  touch "$marker"
}

for ((n=START_WINDOW; n<=END_WINDOW; n++)); do
  prev=$((n - 1))
  next=$((n + 1))
  w="W$n"
  pw="W$prev"
  nw="W$next"
  dir="stage3/data/temporal_rolling_r3/$w"
  logs="stage3/runs/temporal_rolling_r3/$w"
  mkdir -p "$dir" "$logs"

  if [[ $n -eq 3 ]]; then
    prev_sel=stage3/data/temporal_rolling_r3/W2/mappings/selective_joint.json
    prev_rand=stage3/data/temporal_rolling_r3/W2/mappings/random_joint.json
    prev_joint=stage3/data/temporal_rolling_r3/W2/joint_adapter_checkpoint.pt
  else
    prev_sel="stage3/data/temporal_rolling_r3/$pw/mappings/selective_joint.json"
    prev_rand="stage3/data/temporal_rolling_r3/$pw/mappings/random_joint.json"
    prev_joint="stage3/data/temporal_rolling_r3/$pw/joint_adapter_checkpoint.pt"
  fi
  prev_snap="stage3/data/temporal_rolling_r3/$pw/snapshot"
  prev_uni="stage3/artifacts/unisar_temporal_rolling_r3_$pw"
  fixed_map=stage3/data/temporal_joint_r2a_mappings/J0_fixed_sid_mapping.json

  cutoff=$(window_end "$n")
  start=$(window_end "$prev")
  run_step "$dir/.snapshot.done" bash -c \
    "python stage3/scripts/export_temporal_snapshot_data.py --source-dir '$SOURCE' --output-dir '$dir/snapshot' --end-exclusive '$cutoff' 2>&1 | tee '$logs/snapshot.log'"

  run_step "$dir/.unisar.done" bash -c \
    "/usr/bin/time -v env CUDA_VISIBLE_DEVICES=4 python -u stage3/scripts/train_unisar_faithful_embeddings.py \
      --data-dir '$dir/snapshot' --output-dir 'stage3/artifacts/unisar_temporal_rolling_r3_$w' \
      --dim 64 --heads 4 --layers 1 --dropout 0.1 --max-history 30 --num-negatives 20 \
      --batch-size 512 --epochs 3 --lr 0.0001 --weight-decay 0.00001 --alpha 0.1 --beta 0.1 \
      --max-valid-samples 5000 --workers 4 --seed 2026 --device cuda:0 --zero-unseen-items \
      --init-checkpoint '$prev_uni/best.pt' --init-data-dir '$prev_snap' --update-task rec \
      --update-start-timestamp '$start' --freeze-for-rec-update 2>&1 | tee '$logs/unisar_update.log'"

  if [[ ! -f "$dir/.adapters.done" ]]; then
    (
      /usr/bin/time -v env CUDA_VISIBLE_DEVICES=4 python -u stage3/scripts/update_v13b_rec_adapter_selective.py \
        --checkpoint "$prev_joint" --t0-mapping "$prev_sel" --data-dir "$dir/snapshot" \
        --common-embedding "$COMMON" --updated-rec-embedding "stage3/artifacts/unisar_temporal_rolling_r3_$w/UniSAR_faithful.pt" \
        --t0-rec-embedding "$prev_uni/UniSAR_faithful.pt" --anchor-lookup "$ANCHOR" \
        --drift-csv "$REC_DRIFT" --drift-left "$pw" --drift-right "$w" --replay-ratio 2 \
        --output-dir "$dir/rec_adapter" --epochs 60 --batch-size 64 --lr 0.0001 --weight-decay 0.0001 \
        --topology-weight 0.5 --commit-weight 0.05 --stability-weight 0.1 --outside-stability-weight 0.1 \
        --patience 10 --device cuda:0 --seed 2026 2>&1 | tee "$logs/rec_adapter.log"
    ) & rec_pid=$!
    (
      /usr/bin/time -v env CUDA_VISIBLE_DEVICES=5 python -u stage3/scripts/update_v13b_search_adapter_selective.py \
        --checkpoint "$prev_joint" --t0-mapping "$prev_sel" --data-dir "$dir/snapshot" --t0-data-dir "$prev_snap" \
        --common-embedding "$COMMON" --rec-embedding "stage3/artifacts/unisar_temporal_rolling_r3_$w/UniSAR_faithful.pt" \
        --anchor-lookup "$ANCHOR" --drift-csv "$SEARCH_DRIFT" --drift-left "$pw" --drift-right "$w" \
        --replay-ratio 2 --output-dir "$dir/search_adapter" --embedding-cache-dir "$dir/search_embedding_cache" \
        --epochs 60 --batch-size 64 --lr 0.0001 --weight-decay 0.0001 --relevance-weight 0.2 \
        --identity-weight 0.5 --topology-weight 0.5 --commit-weight 0.05 --stability-weight 0.1 \
        --outside-stability-weight 0.1 --patience 10 --device cuda:0 --seed 2026 2>&1 | tee "$logs/search_adapter.log"
    ) & search_pid=$!
    wait "$rec_pid"
    wait "$search_pid"
    touch "$dir/.adapters.done"
  fi

  if [[ ! -f "$dir/.sid.done" ]]; then
    rec_common=(--base-checkpoint "$prev_joint" --adapter-checkpoint "$dir/rec_adapter/rec_adapter_best.pt" \
      --data-dir "$dir/snapshot" --common-embedding "$COMMON" \
      --updated-rec-embedding "stage3/artifacts/unisar_temporal_rolling_r3_$w/UniSAR_faithful.pt" \
      --anchor-lookup "$ANCHOR" --drift-csv "$REC_DRIFT" --drift-left "$pw" --drift-right "$w" \
      --switch-ratios 0.1 --batch-size 1024 --device cuda:0 --seed 2026)
    search_common=(--base-checkpoint "$prev_joint" --adapter-checkpoint "$dir/search_adapter/search_adapter_best.pt" \
      --data-dir "$dir/snapshot" --common-embedding "$COMMON" \
      --rec-embedding "stage3/artifacts/unisar_temporal_rolling_r3_$w/UniSAR_faithful.pt" \
      --anchor-lookup "$ANCHOR" --drift-csv "$SEARCH_DRIFT" --drift-left "$pw" --drift-right "$w" \
      --switch-ratios 0.1 --batch-size 1024 --device cuda:0 --seed 2026)
    (CUDA_VISIBLE_DEVICES=4 python stage3/scripts/materialize_v13b_temporal_rec_sid_sticky.py \
      "${rec_common[@]}" --base-operational-mapping "$prev_sel" --output-dir "$dir/rec_sticky_selective" \
      >"$logs/rec_sticky_selective.log" 2>&1) & p1=$!
    (CUDA_VISIBLE_DEVICES=5 python stage3/scripts/materialize_v13b_temporal_search_sid_sticky.py \
      "${search_common[@]}" --base-operational-mapping "$prev_sel" \
      --embedding-cache-dir "$dir/search_embedding_cache/selective_materialize" --output-dir "$dir/search_sticky_selective" \
      >"$logs/search_sticky_selective.log" 2>&1) & p2=$!
    (CUDA_VISIBLE_DEVICES=6 python stage3/scripts/materialize_v13b_temporal_rec_sid_sticky.py \
      "${rec_common[@]}" --base-operational-mapping "$prev_rand" --output-dir "$dir/rec_sticky_random" \
      >"$logs/rec_sticky_random.log" 2>&1) & p3=$!
    (CUDA_VISIBLE_DEVICES=7 python stage3/scripts/materialize_v13b_temporal_search_sid_sticky.py \
      "${search_common[@]}" --base-operational-mapping "$prev_rand" \
      --embedding-cache-dir "$dir/search_embedding_cache/random_materialize" --output-dir "$dir/search_sticky_random" \
      >"$logs/search_sticky_random.log" 2>&1) & p4=$!
    wait "$p1"; wait "$p2"; wait "$p3"; wait "$p4"

    python stage3/scripts/compose_temporal_task_mappings.py --base-mapping "$prev_sel" \
      --search-mapping "$dir/search_sticky_selective/drift_selected_sticky_ratio_0p1/sid_mapping.json" \
      --rec-mapping "$dir/rec_sticky_selective/drift_selected_sticky_ratio_0p1/sid_mapping.json" \
      --output "$dir/mappings/selective_joint.json" >"$logs/compose_selective.log"
    python stage3/scripts/compose_temporal_task_mappings.py --base-mapping "$prev_rand" \
      --search-mapping "$dir/search_sticky_random/budget_random_sticky_ratio_0p1/sid_mapping.json" \
      --rec-mapping "$dir/rec_sticky_random/budget_random_sticky_ratio_0p1/sid_mapping.json" \
      --output "$dir/mappings/random_joint.json" >"$logs/compose_random.log"
    python stage3/scripts/merge_v13_task_adapters.py --base-checkpoint "$prev_joint" \
      --search-adapter "$dir/search_adapter/search_adapter_best.pt" --rec-adapter "$dir/rec_adapter/rec_adapter_best.pt" \
      --output "$dir/joint_adapter_checkpoint.pt" --window "$w" >"$logs/merge_adapters.log"
    touch "$dir/.sid.done"
  fi

  run_step "$dir/.export.done" bash -c "
    python stage3/scripts/export_sid_mapping_for_gensar.py --mapping '$fixed_map' --item-ids '$ITEM_IDS' --output-dir '$dir/exports/fixed' --name '${w}_fixed'
    python stage3/scripts/export_sid_mapping_for_gensar.py --mapping '$dir/mappings/selective_joint.json' --item-ids '$ITEM_IDS' --output-dir '$dir/exports/selective' --name '${w}_selective'
    python stage3/scripts/export_sid_mapping_for_gensar.py --mapping '$dir/mappings/random_joint.json' --item-ids '$ITEM_IDS' --output-dir '$dir/exports/random' --name '${w}_random'
    python stage3/scripts/build_rolling_alignment_pools.py \
      --transition Selective '$prev_sel' '$dir/mappings/selective_joint.json' \
      --transition Random '$prev_rand' '$dir/mappings/random_joint.json' --output-dir '$dir/alignment'"

  if [[ ! -f "$dir/.datasets.done" ]]; then
    for arm in Fixed Selective Random; do
      low=$(tr '[:upper:]' '[:lower:]' <<<"$arm")
      if [[ $arm == Fixed ]]; then base_map=$fixed_map; cur_map=$fixed_map; fi
      if [[ $arm == Selective ]]; then base_map=$prev_sel; cur_map="$dir/mappings/selective_joint.json"; fi
      if [[ $arm == Random ]]; then base_map=$prev_rand; cur_map="$dir/mappings/random_joint.json"; fi
      python stage3/scripts/materialize_temporal_incremental_gensar.py --quarterly-dir "$QUARTERLY" \
        --update-window "$w" --eval-window "$nw" --output-dir "$GENSAR_DATA/GenSAR_Faithful_20K_R3_${w}_${arm}" \
        --semantic-sid "$dir/exports/$low/${w}_${low}_semantic.json" \
        --collaborate-sid "$dir/exports/$low/${w}_${low}_collaborate.json" \
        --base-mapping "$base_map" --current-mapping "$cur_map" --item-ids "$ITEM_IDS" --update-task both \
        --search-alignment-items "$dir/alignment/search_alignment_items.json" \
        --rec-alignment-items "$dir/alignment/rec_alignment_items.json" --seed 2024 \
        >"$logs/materialize_${low}.log"
    done
    touch "$dir/.datasets.done"
  fi

  if [[ ! -f "$dir/.downstream.done" ]]; then
    for arm_gpu in Fixed:4 Selective:5 Random:6; do
      arm=${arm_gpu%%:*}; gpu=${arm_gpu##*:}; low=$(tr '[:upper:]' '[:lower:]' <<<"$arm")
      prev_run="stage3/runs/temporal_incremental/R3_${pw}_${arm}_s2024"
      (
        bash stage3/scripts/run_temporal_incremental_t5.sh "$gpu" "R3_${w}_${arm}_s2024" \
          "GenSAR_Faithful_20K_R3_${w}_${arm}" "${w}_${low}_semantic.json" "${w}_${low}_collaborate.json" \
          1800 2024 "$ROOT/$prev_run"
      ) & eval "pid_$arm=$!"
    done
    wait "$pid_Fixed"; wait "$pid_Selective"; wait "$pid_Random"
    touch "$dir/.downstream.done"
  fi

  python stage3/scripts/summarize_joint_temporal_three_seed.py \
    --run Fixed:2024="stage3/runs/temporal_incremental/R3_${w}_Fixed_s2024" \
    --run Selective:2024="stage3/runs/temporal_incremental/R3_${w}_Selective_s2024" \
    --run Random:2024="stage3/runs/temporal_incremental/R3_${w}_Random_s2024" \
    --output-dir "stage3/results/temporal_rolling_r3_${w}" >"$logs/downstream_summary.log"

  # The root model is the best checkpoint reloaded by Trainer; discard bulky optimizer checkpoints.
  rm -rf stage3/runs/temporal_incremental/R3_${pw}_{Fixed,Selective,Random}_s2024/checkpoint-*
done
