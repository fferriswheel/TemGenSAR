#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
GENSAR_ROOT=/data/cjk/external/GenSAR/finetune
INITIAL=$ROOT/stage3/runs/temporal_t0/gensar_w1_noupdate_seed2024
OUTPUT=$ROOT/stage3/runs/temporal_e4/t5_incremental
mkdir -p "$OUTPUT"
cd "$GENSAR_ROOT"

/usr/bin/time -v env CUDA_VISIBLE_DEVICES=6 python -u main.py \
  --model_name_or_path t5-small --initial_model_checkpoint "$INITIAL" \
  --max_input_len 2048 --dataset GenSAR_Faithful_20K_W1Inc_W2Eval_E4_GenSARUpdate \
  --semantic_index_file GenSAR_Faithful_20K_RecW1_GenSAR_Update_BGE_UniSAR_p256_collision_semantic.json \
  --collaborate_index_file GenSAR_Faithful_20K_RecW1_GenSAR_Update_BGE_UniSAR_p256_collision_collaborate.json \
  --complete_sid_vocab true --sid_codebook_size 256 \
  --max_rec_his_len 5 --max_src_session_his_len 5 --max_session_item_len 1 \
  --max_query_len 64 --max_doc_len 256 --max_query_per_doc 1 \
  --doc2query_file item2query_t0_nq20 \
  --train_tasks SeqRecWithSrcHis,PerSrcWithRecHis,Item2Index,Index2Item \
  --val_tasks SeqRecWithSrcHis,PerSrcWithRecHis \
  --test_tasks SeqRecWithSrcHis,PerSrcWithRecHis \
  --train_prompt_sample_num 1,1,1,1 \
  --train_data_sample_num 0,0,19999,19999 \
  --val_prompt_ids 0 --val_data_sample_num 0,0 --test_data_sample_num 0,0 \
  --num_negs 999 --num_beams 10 --max_new_tokens 32 --only_train_response true \
  --output_dir "$OUTPUT" --dataloader_num_workers 8 \
  --do_train true --do_eval true --do_predict true --num_train_epochs 1 \
  --learning_rate 0.0001 --lr_scheduler_type cosine --warmup_ratio 0.01 \
  --weight_decay 0.0000001 --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 --gradient_accumulation_steps 4 \
  --eval_strategy epoch --save_strategy epoch --save_total_limit 1 \
  --load_best_model_at_end true --metric_for_best_model ndcg@5 \
  --greater_is_better true \
  --eval_metrics hit@1,hit@5,hit@10,hit@20,ndcg@5,ndcg@10,ndcg@20 \
  --logging_steps 10 --seed 2024 --report_to none 2>&1 | tee "$OUTPUT/train.log"
