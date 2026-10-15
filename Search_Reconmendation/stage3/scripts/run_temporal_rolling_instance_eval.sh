#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 CUDA_ID UPDATE_WINDOW CHECKPOINT_ARM OUTPUT_DIR [DATA_ARM]" >&2
  exit 2
fi

CUDA_ID=$1
UPDATE_WINDOW=$2
ARM=$3
ROOT=/data/cjk/Search_Reconmendation
GENSAR_ROOT=/data/cjk/external/GenSAR/finetune
OUTPUT=$(realpath -m "$4")
DATA_ARM=${5:-$ARM}
DATASET="GenSAR_Faithful_20K_R3_${UPDATE_WINDOW}_${DATA_ARM}"
CHECKPOINT="$ROOT/stage3/runs/temporal_incremental/R3_${UPDATE_WINDOW}_${ARM}_s2024"
DATA_DIR="/data/cjk/external/GenSAR/data/$DATASET"

SEMANTIC_FILE=$(find "$DATA_DIR/emb" -maxdepth 1 \( -type l -o -type f \) -name '*semantic.json' -printf '%f\n')
COLLABORATE_FILE=$(find "$DATA_DIR/emb" -maxdepth 1 \( -type l -o -type f \) -name '*collaborate.json' -printf '%f\n')
if [[ -z "$SEMANTIC_FILE" || -z "$COLLABORATE_FILE" ]]; then
  echo "Could not resolve SID files for $DATASET" >&2
  exit 1
fi

mkdir -p "$OUTPUT"
cd "$GENSAR_ROOT"

/usr/bin/time -v env CUDA_VISIBLE_DEVICES="$CUDA_ID" python -u main.py \
  --model_name_or_path "$CHECKPOINT" \
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
  --train_tasks SeqRecWithSrcHis,PerSrcWithRecHis \
  --val_tasks SeqRecWithSrcHis,PerSrcWithRecHis \
  --test_tasks SeqRecWithSrcHis,PerSrcWithRecHis \
  --train_prompt_sample_num 1,1 \
  --train_data_sample_num 0,0 \
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
  --do_train false \
  --do_eval false \
  --do_predict true \
  --resume_from_checkpoint "$CHECKPOINT" \
  --per_device_eval_batch_size 32 \
  --eval_metrics hit@1,hit@5,hit@10,ndcg@5,ndcg@10 \
  --seed 2024 \
  --report_to none 2>&1 | tee "$OUTPUT/predict.log"
