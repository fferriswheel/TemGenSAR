#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
INDEX_ROOT=/data/cjk/external/GenSAR/index
DATASET=GenSAR_Faithful_20K_RecW1_GenSAR_Update
CKPT=output/${DATASET}_BGE_UniSAR_p256
LOG=$ROOT/stage3/runs/temporal_e4
mkdir -p "$LOG"
cd "$INDEX_ROOT"

/usr/bin/time -v python -u main.py \
  --seed 2024 --lr 0.001 --epochs 500 --batch_size 1024 --num_workers 4 \
  --eval_step 10 --learner AdamW --lr_scheduler_type linear --warmup_epochs 50 \
  --dataset "$DATASET" --semantic_emb bge-base-en-v1.5 \
  --collaborate_emb UniSAR_rec_w1 --shared_e_dim 64 --specific_e_dim 32 \
  --shared_num_emb_list 256 256 --semantic_num_emb_list 256 256 \
  --collaborate_num_emb_list 256 256 --shared_sk_epsilons 0 0 \
  --specific_sk_epsilons 0 0.003 --weight_decay 0.00001 --dropout_prob 0 \
  --bn 0 --loss_type mse --kmeans_init 1 --kmeans_iters 100 --sk_iters 50 \
  --device cuda:6 --quant_loss_weight 1 --beta 0.25 --layers 512 256 128 64 \
  --save_limit 1 --ckpt_dir "$CKPT" 2>&1 | tee "$LOG/rqvae.log"

/usr/bin/time -v python -u generate_indices.py --dataset "$DATASET" \
  --ckpt_path "$CKPT" --ckpt_type best_collision_model --device cuda:6 \
  2>&1 | tee "$LOG/generate.log"
