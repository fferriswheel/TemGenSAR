#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 CUDA_ID EXPERIMENT CHECKPOINT DATASET SEMANTIC_FILE COLLABORATE_FILE TASK" >&2
  exit 2
fi

CUDA_ID=$1
EXPERIMENT=$2
CHECKPOINT=$3
DATASET=$4
SEMANTIC_FILE=$5
COLLABORATE_FILE=$6
TASK=$7
ROOT=/data/cjk/Search_Reconmendation
GENSAR_ROOT=/data/cjk/external/GenSAR/finetune
OUTPUT=$ROOT/stage3/runs/temporal_r1/$EXPERIMENT

if [[ "$TASK" == "rec" ]]; then
  TEST_TASK=SeqRecWithSrcHis
elif [[ "$TASK" == "search" ]]; then
  TEST_TASK=PerSrcWithRecHis
else
  echo "TASK must be rec or search" >&2
  exit 2
fi

mkdir -p "$OUTPUT"
cd "$GENSAR_ROOT"

/usr/bin/time -v env CUDA_VISIBLE_DEVICES="$CUDA_ID" python -u main.py \
  --model_name_or_path t5-small \
  --max_input_len 2048 \
  --dataset "$DATASET" \
  --semantic_index_file "$SEMANTIC_FILE" \
  --collaborate_index_file "$COLLABORATE_FILE" \
  --complete_sid_vocab true \
  --sid_codebook_size 256 \
  --max_rec_his_len 5 \
  --max_src_session_his_len 5 \
  --max_session_item_len 1 \
  --max_query_len 64 \
  --max_doc_len 256 \
  --max_query_per_doc 1 \
  --doc2query_file item2query_t0_nq20 \
  --train_tasks "$TEST_TASK" \
  --val_tasks "$TEST_TASK" \
  --test_tasks "$TEST_TASK" \
  --train_prompt_sample_num 1 \
  --train_data_sample_num 0 \
  --val_prompt_ids 0 \
  --val_data_sample_num 0 \
  --test_data_sample_num 0 \
  --num_negs 999 \
  --num_beams 10 \
  --max_new_tokens 32 \
  --only_train_response true \
  --save_instance_results true \
  --output_dir "$OUTPUT" \
  --dataloader_num_workers 8 \
  --do_train false \
  --do_eval false \
  --do_predict true \
  --resume_from_checkpoint "$CHECKPOINT" \
  --per_device_eval_batch_size 32 \
  --eval_metrics hit@1,hit@5,hit@10,ndcg@5,ndcg@10 \
  --seed 2024 \
  --report_to none 2>&1 | tee "$OUTPUT/predict.log"
