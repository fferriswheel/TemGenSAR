#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 8 ]]; then
  echo "usage: $0 CUDA_ID EXPERIMENT DATASET SEMANTIC_FILE COLLABORATE_FILE [ALIGNMENT_BUDGET] [SEED] [INITIAL_CHECKPOINT]" >&2
  exit 2
fi

CUDA_ID=$1
EXPERIMENT=$2
DATASET=$3
SEMANTIC_FILE=$4
COLLABORATE_FILE=$5
ALIGNMENT_BUDGET=${6:-2567}
SEED=${7:-2024}
ROOT=/data/cjk/Search_Reconmendation
GENSAR_ROOT=/data/cjk/external/GenSAR/finetune
INITIAL=${8:-$ROOT/stage3/runs/temporal_t0/v13b_w1_noupdate_seed2024}
OUTPUT=$ROOT/stage3/runs/temporal_incremental/$EXPERIMENT

mkdir -p "$OUTPUT"
cd "$GENSAR_ROOT"

/usr/bin/time -v env CUDA_VISIBLE_DEVICES="$CUDA_ID" python -u main.py \
  --model_name_or_path t5-small \
  --initial_model_checkpoint "$INITIAL" \
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
  --train_tasks SeqRecWithSrcHis,PerSrcWithRecHis,Item2Index,Index2Item \
  --val_tasks SeqRecWithSrcHis,PerSrcWithRecHis \
  --test_tasks SeqRecWithSrcHis,PerSrcWithRecHis \
  --train_prompt_sample_num 1,1,1,1 \
  --train_data_sample_num 0,0,"$ALIGNMENT_BUDGET","$ALIGNMENT_BUDGET" \
  --val_prompt_ids 0 \
  --val_data_sample_num 0,0 \
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
