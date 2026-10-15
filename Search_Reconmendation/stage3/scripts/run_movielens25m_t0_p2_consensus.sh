#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
GENSAR_ROOT=/data/cjk/external/GenSAR/finetune
DATA=$ROOT/stage3/data/movielens25m_fixed18k_init_t0_v1
GENSAR_DATA=$ROOT/stage3/data/movielens25m_gensar_temporal_v1
COMMON=$ROOT/stage3/artifacts/movielens25m_common_t0_v1/bge_common.pt
REC=$ROOT/stage3/artifacts/movielens25m_unisar_t0_pilot_v1/UniSAR_faithful.pt
SEARCH=$ROOT/stage3/artifacts/movielens25m_search_t0_v3_dim768_hitselect/search_embedding.pt
ARTIFACT=$ROOT/stage3/artifacts/movielens25m_p2_consensus_t0_v1
RUN=$ROOT/stage3/runs/movielens25m_t0/p2_consensus_cg_smoke_seed2026
GPU=${GPU:-4}

mkdir -p "$ARTIFACT/anchor" "$ARTIFACT/suffix" "$ARTIFACT/suffix_cg" "$RUN"
cd "$ROOT"

/usr/bin/time -v env CUDA_VISIBLE_DEVICES="$GPU" python -u \
  stage3/scripts/train_multiview_anchor_prefix_v13.py \
  --data-dir "$DATA" \
  --output-dir "$ARTIFACT/anchor" \
  --common-embedding "$COMMON" \
  --rec-behavior-embedding "$REC" \
  --search-embedding "$SEARCH" \
  --embedding-cache-dir "$ARTIFACT/cache" \
  --max-items 18191 \
  --anchor-mode consensus \
  --prefix-codebook-sizes 256,256 \
  --max-queries-per-item 20 \
  --hidden-dim 256 \
  --latent-dim 64 \
  --dropout 0.05 \
  --anchor-epochs 100 \
  --anchor-warmup-epochs 10 \
  --batch-size 1024 \
  --anchor-lr 0.001 \
  --weight-decay 0.00001 \
  --vq-beta 0.25 \
  --balance-temperature 0.05 \
  --commit-weight 0.05 \
  --balance-weight 0.02 \
  --gate-diversity-weight 0 \
  --consensus-alpha-max 0.5 \
  --consensus-aux-recon-weight 0.25 \
  --consensus-agreement-weight 0.2 \
  --consensus-common-weight 0.05 \
  --consensus-consistency-weight 0.2 \
  --task-view-dropout 0.3 \
  --minimum-code-usage 128 \
  --selection-underuse-weight 0.1 \
  --grad-clip 1 \
  --log-every 5 \
  --device cuda:0 \
  --seed 2026 2>&1 | tee "$ARTIFACT/anchor.run.log"

/usr/bin/time -v env CUDA_VISIBLE_DEVICES="$GPU" python -u \
  stage3/scripts/build_neural_rq_sid_v9_direct_private.py \
  --data-dir "$DATA" \
  --output-dir "$ARTIFACT/suffix" \
  --common-embedding "$COMMON" \
  --rec-behavior-embedding "$REC" \
  --search-embedding "$SEARCH" \
  --anchor-lookup "$ARTIFACT/anchor/anchor_lookup.npz" \
  --anchor-checkpoint "$ARTIFACT/anchor/anchor_prefix_best.pt" \
  --max-items 18191 \
  --prefix-codebook-sizes 256,256 \
  --search-codebook-sizes 256,256 \
  --rec-codebook-sizes 256,256 \
  --prefix-assignment argmin \
  --max-queries-per-item 20 \
  --neighbor-window 2 \
  --max-neighbors-per-item 200 \
  --embedding-cache-dir "$ARTIFACT/cache" \
  --hidden-dim 256 \
  --latent-dim 64 \
  --private-rank 32 \
  --conditioned-residual-decoder \
  --split-search-private \
  --hierarchical-search-vq \
  --direct-sinkhorn-vq \
  --direct-sinkhorn-epsilon 0.003 \
  --direct-sinkhorn-iters 50 \
  --search-relevance-dim 24 \
  --bridge-hidden-dim 128 \
  --dropout 0.05 \
  --task-epochs 100 \
  --continuous-warmup-epochs 10 \
  --batch-size 1024 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --vq-beta 0.25 \
  --balance-temperature 0.05 \
  --search-target-scale 0.5 \
  --task-mse-weight 10 \
  --task-cos-weight 0.1 \
  --full-mse-weight 1 \
  --query-contrast-weight 0.2 \
  --query-contrast-temperature 0.1 \
  --identity-recon-weight 2 \
  --identity-cos-scale 0.1 \
  --identity-anchor-weight 0.5 \
  --identity-topology-weight 0.5 \
  --topology-weight 1 \
  --redundancy-weight 0.05 \
  --full-sid-collision-weight 0.02 \
  --full-sid-temperature 0.05 \
  --full-sid-topk 4 \
  --assignment-confidence-weight 0.01 \
  --commit-weight 0.05 \
  --balance-weight 0.02 \
  --conditional-sinkhorn-weight 0.02 \
  --conditional-sinkhorn-epsilon 0.03 \
  --conditional-sinkhorn-iters 10 \
  --conditional-sinkhorn-temperature 0.1 \
  --collision-warmup-epochs 10 \
  --collision-controller-gain 2 \
  --search-collision-target 0.08 \
  --rec-collision-target 0.08 \
  --minimum-code-usage 128 \
  --selection-collision-weight 2 \
  --selection-underuse-weight 0.2 \
  --prefix-group-size 16 \
  --export-sinkhorn-rounds 0 \
  --export-sinkhorn-epsilon 0.003 \
  --export-sinkhorn-iters 50 \
  --grad-clip 1 \
  --log-every 5 \
  --device cuda:0 \
  --seed 2026 2>&1 | tee "$ARTIFACT/suffix.run.log"

/usr/bin/time -v env CUDA_VISIBLE_DEVICES="$GPU" python -u \
  stage3/scripts/refine_v13_collision_group_sinkhorn.py \
  --checkpoint "$ARTIFACT/suffix/direct_private_sid_model.pt" \
  --mapping "$ARTIFACT/suffix/sid_mapping.raw.json" \
  --output-dir "$ARTIFACT/suffix_cg" \
  --epsilon 0.003 \
  --sk-iters 100 \
  --max-rounds 20 \
  --batch-size 1024 \
  --device cuda:0 2>&1 | tee "$ARTIFACT/suffix_cg.run.log"

python stage3/scripts/export_sid_mapping_for_gensar.py \
  --mapping "$ARTIFACT/suffix_cg/sid_mapping.v13b_cg_sinkhorn.json" \
  --item-ids "$DATA/item_ids.jsonl" \
  --output-dir "$GENSAR_DATA/emb" \
  --name p2_consensus_t0_cg_p256 2>&1 | tee "$ARTIFACT/export.run.log"

cd "$GENSAR_ROOT"
/usr/bin/time -v env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u main.py \
  --model_name_or_path t5-small \
  --max_input_len 512 \
  --dataset MovieLens25M_Fixed18K_T0_V13B_CG \
  --semantic_index_file p2_consensus_t0_cg_p256_semantic.json \
  --collaborate_index_file p2_consensus_t0_cg_p256_collaborate.json \
  --complete_sid_vocab true \
  --sid_codebook_size 256 \
  --max_rec_his_len 5 \
  --max_src_session_his_len 5 \
  --max_session_item_len 1 \
  --max_query_len 32 \
  --max_doc_len 96 \
  --max_query_per_doc 1 \
  --doc2query_file item2query_t0_nq20 \
  --train_tasks SeqRecWithSrcHis,PerSrcWithRecHis,QGenWithRecHis,AdHocSrc_Doc2Query,Item2Index,Index2Item \
  --val_tasks SeqRecWithSrcHis,PerSrcWithRecHis \
  --train_prompt_sample_num 1,1,1,1,1,1 \
  --train_data_sample_num 100000,100000,1000,1000,1000,1000 \
  --val_prompt_ids 0 \
  --val_data_sample_num 0,0 \
  --num_negs 99 \
  --num_beams 10 \
  --max_new_tokens 32 \
  --only_train_response true \
  --output_dir "$RUN" \
  --dataloader_num_workers 8 \
  --do_train true \
  --do_eval true \
  --do_predict false \
  --num_train_epochs 1 \
  --learning_rate 0.001 \
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
  --logging_steps 50 \
  --seed 2026 \
  --report_to none 2>&1 | tee "$RUN/train.log"
