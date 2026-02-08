#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=$PBS_JOBFS/triton-cache
export DATASET_IMPL=dataset_v5

mkdir -p "$TRITON_CACHE_DIR"

TORCHRUN=${TORCHRUN:-/scratch/li96/zl9731/envs/ms-swift/1.0.0/bin/torchrun}

LOG_DIR=/scratch/li96/zl9731/cs16/Model/output/VLM/test
mkdir -p "$LOG_DIR"
nvidia-smi > "$LOG_DIR/nvidia-smi.log" 2>&1
(while sleep 60; do nvidia-smi >> "$LOG_DIR/nvidia-smi.log" 2>&1; done) &
SMI_PID=$!
trap 'kill $SMI_PID' EXIT

${TORCHRUN} --nproc_per_node=${NPROC_PER_NODE} /scratch/li96/zl9731/cs16/vri-food/VLM/train/trainer.py \
  --base_model /scratch/li96/zl9731/cs16/Model/output/VLM/v3-20251222-211237/checkpoint-7633-merged \
  --init_from_checkpoint /scratch/li96/zl9731/cs16/Model/output/VLM/v6.2 \
  --dataset $PBS_JOBFS/data/vlm_eval_A_v5.jsonl \
  --val_dataset $PBS_JOBFS/data/vlm_eval_A_v5.jsonl \
  --output_dir /scratch/li96/zl9731/cs16/Data/dataAB_v5/output \
  --deepspeed "" \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --enable_vision_lora true \
  --enable_text_lora true \
  --eval_dump_predictions /scratch/li96/zl9731/cs16/Data/dataAB_v5/vlm_infer_A_v6.2.jsonl \
  --eval_max_new_tokens 512 \
  --dataloader_num_workers 12 \
  --eval_generate_batch_size 1 \
  --max_steps 0 --epochs 0 --save_total_limit 1 \
  --skip_save \
  --label_threshold 0.3 \
  --lambda_lm 1.0 --lambda_lm_title 0.1 --lambda_lm_ing 1.0 \
  --lambda_cuisine 0.0 --lambda_meal 0.0 --lambda_dish 0.0 \
  --lambda_amount 0.0 --lambda_ratio 0.0 --lambda_hinge 0.0 --lambda_total_weight 0.0
