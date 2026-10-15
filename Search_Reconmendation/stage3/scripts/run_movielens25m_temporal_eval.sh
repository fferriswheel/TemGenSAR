#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 WINDOW [GPU]" >&2
  exit 2
fi

WINDOW=$1
GPU=${2:-4}
ROOT=/data/cjk/Search_Reconmendation
GENSAR_ROOT=/data/cjk/external/GenSAR
SOURCE=$ROOT/stage3/data/movielens25m_gensar_temporal_v1
DATASET=MovieLens25M_Fixed18K_${WINDOW}_V13B_T0CG
DATA_DIR=$GENSAR_ROOT/data/$DATASET
CHECKPOINT=$ROOT/stage3/runs/movielens25m_t0/v13b_cg_smoke_seed2026
OUTPUT=$ROOT/stage3/runs/movielens25m_temporal/${WINDOW}_fixed_t0_sid_model

python "$ROOT/stage3/scripts/materialize_quarterly_gensar_window.py" \
  --quarterly-dir "$SOURCE" \
  --window "$WINDOW" \
  --output-dir "$DATA_DIR" \
  --semantic-sid "$SOURCE/emb/v13b_t0_cg_p256_semantic.json" \
  --collaborate-sid "$SOURCE/emb/v13b_t0_cg_p256_collaborate.json" \
  --doc2query "$SOURCE/vocab/item2query_t0_nq20.pkl"

mkdir -p "$OUTPUT"
cd "$GENSAR_ROOT/finetune"

/usr/bin/time -v env CUDA_VISIBLE_DEVICES="$GPU" python -u main.py \
  --model_name_or_path "$CHECKPOINT" \
  --max_input_len 512 \
  --dataset "$DATASET" \
  --semantic_index_file v13b_t0_cg_p256_semantic.json \
  --collaborate_index_file v13b_t0_cg_p256_collaborate.json \
  --complete_sid_vocab true \
  --sid_codebook_size 256 \
  --max_rec_his_len 5 \
  --max_src_session_his_len 5 \
  --max_session_item_len 1 \
  --max_query_len 32 \
  --max_doc_len 96 \
  --max_query_per_doc 1 \
  --doc2query_file item2query_t0_nq20 \
  --train_tasks SeqRecWithSrcHis,PerSrcWithRecHis \
  --val_tasks SeqRecWithSrcHis,PerSrcWithRecHis \
  --test_tasks SeqRecWithSrcHis,PerSrcWithRecHis \
  --train_prompt_sample_num 1,1 \
  --train_data_sample_num 0,0 \
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
  --do_train false \
  --do_eval false \
  --do_predict true \
  --resume_from_checkpoint "$CHECKPOINT" \
  --per_device_eval_batch_size 32 \
  --eval_metrics hit@1,hit@5,hit@10,hit@20,ndcg@5,ndcg@10,ndcg@20 \
  --seed 2026 \
  --report_to none 2>&1 | tee "$OUTPUT/predict.log"
