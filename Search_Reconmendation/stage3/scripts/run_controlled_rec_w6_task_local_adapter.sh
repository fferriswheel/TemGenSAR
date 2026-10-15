#!/usr/bin/env bash
set -euo pipefail

CUDA_ID=${1:-4}
SEED=${2:-2024}
ROOT=/data/cjk/Search_Reconmendation
GENSAR_ROOT=/data/cjk/external/GenSAR/finetune
SOURCE=/data/cjk/external/GenSAR/data/GenSAR_Faithful_20K_CRD_W6_V3_A1_R2
DATASET=GenSAR_Faithful_20K_CRD_W6_V3_TaskLocal_AffectedReplay
DATASET_DIR=/data/cjk/external/GenSAR/data/$DATASET
INITIAL=$ROOT/stage3/runs/temporal_incremental/CRD_W6_V3_A1_R1_s2024
EXPERIMENT=CRD_W6_V3_TaskLocalRec_AffectedReplay_s${SEED}
OUTPUT=$ROOT/stage3/runs/temporal_incremental/$EXPERIMENT

cd "$ROOT"
python stage3/scripts/build_affected_rec_adapter_dataset.py \
  --source-dataset "$SOURCE" \
  --output-dataset "$DATASET_DIR" \
  --base-mapping stage3/data/temporal_joint_r2a_mappings/J0_fixed_sid_mapping.json \
  --current-mapping stage3/data/controlled_rec_drift_w3w6w9_v1/W6/V3_sid/materialized/drift_selected_sticky_ratio_0p1/sid_mapping.json \
  --id-maps stage3/data/gensar_faithful_20k_quarterly_controlled_rec_w3w6w9_v1/id_maps.json \
  --replay-size 4000 --max-rec-history 5 --seed "$SEED"

python - "$DATASET_DIR" <<'PY'
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "affected_rec_adapter_manifest.json").read_text())
assert manifest["recoded_items"] == 33
assert manifest["affected_union_rows"] == 5344
assert manifest["replay_rows"] == 4000
assert manifest["output_rec_rows"] == 9344
rec_items = pickle.load((root / "vocab/rec_item_set.pkl").open("rb"))
counts = Counter(rec_items)
assert len(rec_items) == 1800 and len(counts) == 1767
assert Counter(counts.values()) == Counter({1: 1734, 2: 33})
print("dataset validation passed")
PY

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"
cd "$GENSAR_ROOT"
/usr/bin/time -v env CUDA_VISIBLE_DEVICES="$CUDA_ID" python -u main_task_local_lora.py \
  --model_name_or_path t5-small \
  --initial_model_checkpoint "$INITIAL" \
  --selective_adapter_task rec \
  --max_input_len 2048 \
  --dataset "$DATASET" \
  --semantic_index_file CRD_W6_V3_semantic.json \
  --collaborate_index_file CRD_W6_V3_collaborate.json \
  --complete_sid_vocab true \
  --sid_codebook_size 256 \
  --max_rec_his_len 5 \
  --max_src_session_his_len 5 \
  --max_session_item_len 1 \
  --max_query_len 64 \
  --max_doc_len 256 \
  --max_query_per_doc 1 \
  --doc2query_file item2query_t0_nq20 \
  --train_tasks SeqRecWithSrcHis,CollaborativeItem2Index,CollaborativeIndex2Item \
  --val_tasks SeqRecWithSrcHis \
  --test_tasks SeqRecWithSrcHis,PerSrcWithRecHis \
  --train_prompt_sample_num 1,1,1 \
  --train_data_sample_num 0,1800,1800 \
  --val_prompt_ids 0 \
  --val_data_sample_num 0 \
  --test_data_sample_num 0,0 \
  --num_negs 999 \
  --num_beams 10 \
  --max_new_tokens 32 \
  --only_train_response true \
  --save_instance_results true \
  --output_dir "$OUTPUT" \
  --dataloader_num_workers 8 \
  --do_train true \
  --do_eval true \
  --do_predict true \
  --num_train_epochs 1 \
  --learning_rate 0.0001 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.01 \
  --weight_decay 0.0000001 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 1 \
  --load_best_model_at_end true \
  --metric_for_best_model ndcg@5 \
  --greater_is_better true \
  --eval_metrics hit@1,hit@5,hit@10,hit@20,ndcg@5,ndcg@10,ndcg@20 \
  --logging_steps 10 \
  --seed "$SEED" \
  --report_to none 2>&1 | tee "$OUTPUT/train.log"

find "$OUTPUT" -depth \
  \( -path "$OUTPUT/checkpoint-*" -o -path "$OUTPUT/checkpoint-*/*" \) -delete
