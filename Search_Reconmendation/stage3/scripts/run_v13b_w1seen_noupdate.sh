#!/usr/bin/env bash
set -euo pipefail

GENSAR_ROOT=/data/cjk/external/GenSAR/finetune
OUTPUT=/data/cjk/Search_Reconmendation/stage3/runs/temporal_t0/v13b_w1seen_noupdate_seed2024
CHECKPOINT=/data/cjk/Search_Reconmendation/stage3/runs/temporal_t0/v13b_w1_noupdate_seed2024

mkdir -p "$OUTPUT"
cd "$GENSAR_ROOT"

/usr/bin/time -v env CUDA_VISIBLE_DEVICES=4 python -u main.py \
  --model_name_or_path t5-small \
  --max_input_len 2048 \
  --dataset GenSAR_Faithful_20K_Quarterly_W1Seen_V13B_T0_CG \
  --semantic_index_file v13b_t0_cg_p256_semantic.json \
  --collaborate_index_file v13b_t0_cg_p256_collaborate.json \
  --complete_sid_vocab true \
  --sid_codebook_size 256 \
  --max_rec_his_len 5 \
  --max_src_session_his_len 5 \
  --max_session_item_len 1 \
  --max_query_len 64 \
  --max_doc_len 256 \
  --max_query_per_doc 1 \
  --doc2query_file item2query_t0_nq20 \
  --train_tasks SeqRecWithSrcHis,PerSrcWithRecHis,QGenWithRecHis,AdHocSrc_Doc2Query,Item2Index,Index2Item \
  --val_tasks SeqRecWithSrcHis,PerSrcWithRecHis \
  --test_tasks SeqRecWithSrcHis,PerSrcWithRecHis \
  --train_prompt_sample_num 1,1,1,1,1,1 \
  --train_data_sample_num 0,100000,1000,1000,1000,1000 \
  --val_prompt_ids 0 \
  --val_data_sample_num 0,0 \
  --test_data_sample_num 0,0 \
  --num_negs 999 \
  --num_beams 10 \
  --max_new_tokens 32 \
  --only_train_response true \
  --output_dir "$OUTPUT" \
  --dataloader_num_workers 8 \
  --do_train false \
  --do_eval false \
  --do_predict true \
  --resume_from_checkpoint "$CHECKPOINT" \
  --per_device_eval_batch_size 32 \
  --eval_metrics hit@1,hit@5,hit@10,hit@20,ndcg@5,ndcg@10,ndcg@20 \
  --seed 2024 \
  --report_to none 2>&1 | tee "$OUTPUT/predict.log"
