#!/usr/bin/env bash
set -euo pipefail

# SFT training template (edit paths as needed)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

DATASET=${DATASET:-/mnt/hdd_1/home/cs16/vri-food/VLM/rlhf/sft.jsonl}
VAL_DATASET=${VAL_DATASET:-/mnt/hdd_1/home/cs16/Data/dataAB_v5/vlm_val_A_v5.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/hdd_1/home/cs16/Model/output/VLM/rlhf_sft}
BASE_MODEL=${BASE_MODEL:-/mnt/hdd_1/home/cs16/Model/output/VLM/v3-20251222-211237/checkpoint-7633-merged}
INIT_CKPT=${INIT_CKPT:-/mnt/hdd_1/home/cs16/Model/output/VLM/v6.2-3}
DEEPSPEED=${DEEPSPEED:-/mnt/hdd_1/home/cs16/vri-food/VLM/train/deepspeed_zero2.json}
TORCHRUN=${TORCHRUN:-/mnt/hdd_1/home/cs16/miniconda3/envs/py310/bin/torchrun}

mkdir -p "$OUTPUT_DIR"

${TORCHRUN} --nproc_per_node=1 /mnt/hdd_1/home/cs16/vri-food/VLM/train/trainer.py \
  --dataset "$DATASET" \
  --val_dataset "$VAL_DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed "$DEEPSPEED" \
  --base_model "$BASE_MODEL" \
  --init_from_checkpoint "$INIT_CKPT" \
  --save_lora_only \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --enable_vision_lora true \
  --enable_text_lora true \
  --epochs 1 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.1 \
  --logging_steps 50 \
  --eval_steps 500 \
  --save_steps 1000 \
  --save_total_limit 2 \
  --lambda_lm 1.0 --lambda_lm_title 0.1 --lambda_lm_ing 1.0 \
  --lambda_cuisine 0.0 --lambda_meal 0.0 --lambda_dish 0.0 \
  --lambda_amount 0.0 --lambda_ratio 0.0 --lambda_hinge 0.0 --lambda_total_weight 0.0
