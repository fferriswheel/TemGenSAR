#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "usage: $0 CUDA_ID EXPERIMENT DATASET INITIAL_CHECKPOINT [ALIGNMENT_BUDGET] [EPOCHS]" >&2
  exit 2
fi

CUDA_ID=$1
EXPERIMENT=$2
DATASET=$3
INITIAL=$4
ALIGNMENT_BUDGET=${5:-23}
EPOCHS=${6:-1}
SEMANTIC_FILE=${SEMANTIC_FILE:-v13b_w1_k150_p256_semantic.json}
COLLABORATE_FILE=${COLLABORATE_FILE:-v13b_w1_k150_p256_collaborate.json}

ROOT=/data/cjk/Search_Reconmendation
GENSAR_ROOT=/data/cjk/external/GenSAR/finetune
OUTPUT=$ROOT/stage3/runs/movielens25m_temporal/$EXPERIMENT

if (( ALIGNMENT_BUDGET > 0 )); then
  TRAIN_TASKS=SeqRecWithSrcHis,PerSrcWithRecHis,CollaborativeItem2Index,CollaborativeIndex2Item
  TRAIN_PROMPTS=1,1,1,1
  TRAIN_SAMPLES=0,0,"$ALIGNMENT_BUDGET","$ALIGNMENT_BUDGET"
else
  TRAIN_TASKS=SeqRecWithSrcHis,PerSrcWithRecHis
  TRAIN_PROMPTS=1,1
  TRAIN_SAMPLES=0,0
fi

mkdir -p "$OUTPUT"
cd "$GENSAR_ROOT"

/usr/bin/time -v env CUDA_VISIBLE_DEVICES="$CUDA_ID" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u main.py \
  --model_name_or_path t5-small \
  --initial_model_checkpoint "$INITIAL" \
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
  --doc2query_file item2query_t0_nq20 \
  --train_tasks "$TRAIN_TASKS" \
  --val_tasks SeqRecWithSrcHis,PerSrcWithRecHis \
  --test_tasks SeqRecWithSrcHis,PerSrcWithRecHis \
  --train_prompt_sample_num "$TRAIN_PROMPTS" \
  --train_data_sample_num "$TRAIN_SAMPLES" \
  --val_prompt_ids 0 \
  --val_data_sample_num 0,0 \
  --test_data_sample_num 0,0 \
  --num_negs 99 \
  --num_beams 10 \
  --max_new_tokens 32 \
  --only_train_response true \
  --save_instance_results true \
  --output_dir "$OUTPUT" \
  --dataloader_num_workers 8 \
  --do_train true \
  --do_eval true \
  --do_predict true \
  --num_train_epochs "$EPOCHS" \
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
  --seed 2026 \
  --report_to none 2>&1 | tee "$OUTPUT/train.log"
