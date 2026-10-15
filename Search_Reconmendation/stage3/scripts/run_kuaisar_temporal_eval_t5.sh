#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 CUDA_ID EXPERIMENT DATASET SEMANTIC_FILE COLLABORATE_FILE CHECKPOINT [SEED]" >&2
  exit 2
fi

CUDA_ID=$1
EXPERIMENT=$2
DATASET=$3
SEMANTIC_FILE=$4
COLLABORATE_FILE=$5
CHECKPOINT=$6
SEED=${7:-2026}

ROOT=/data/cjk/Search_Reconmendation
GENSAR_ROOT=/data/cjk/external/GenSAR/finetune
OUTPUT=$ROOT/stage3/runs/kuaisar_temporal/$EXPERIMENT

mkdir -p "$OUTPUT"
cd "$GENSAR_ROOT"

/usr/bin/time -v env CUDA_VISIBLE_DEVICES="$CUDA_ID" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u main.py \
  --model_name_or_path t5-small \
  --resume_from_checkpoint "$CHECKPOINT" \
  --max_input_len 512 \
  --dataset "$DATASET" \
  --semantic_index_file "$SEMANTIC_FILE" \
  --collaborate_index_file "$COLLABORATE_FILE" \
  --complete_sid_vocab true \
  --sid_codebook_size 256 \
  --max_rec_his_len 5 \
  --max_src_session_his_len 5 \
  --max_session_item_len 1 \
  --max_query_len 32 \
  --max_doc_len 96 \
  --max_query_per_doc 1 \
  --doc2query_file item2query_w0_w2_nq20 \
  --val_tasks SeqRecWithSrcHis,PerSrcWithRecHis \
  --val_data_sample_num 0,0 \
  --test_tasks SeqRecWithSrcHis,PerSrcWithRecHis \
  --test_data_sample_num 0,0 \
  --val_prompt_ids 0 \
  --num_negs 99 \
  --num_beams 10 \
  --max_new_tokens 32 \
  --only_train_response true \
  --save_instance_results true \
  --output_dir "$OUTPUT" \
  --dataloader_num_workers 8 \
  --do_train false \
  --do_eval false \
  --do_predict true \
  --per_device_eval_batch_size 32 \
  --eval_metrics hit@1,hit@5,hit@10,hit@20,ndcg@5,ndcg@10,ndcg@20 \
  --seed "$SEED" \
  --report_to none 2>&1 | tee "$OUTPUT/eval.log"
