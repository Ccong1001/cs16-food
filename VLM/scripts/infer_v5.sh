#!/usr/bin/env bash
set -euo pipefail


export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

TORCHRUN=${TORCHRUN:-/mnt/hdd_1/home/cs16/miniconda3/envs/swift/bin/torchrun}

${TORCHRUN} --nproc_per_node=1 vri-food/VLM/train/trainer.py \
  --base_model /mnt/hdd_1/home/cs16/Model/output/VLM/v3-20251222-211237/checkpoint-7633-merged \
  --init_from_checkpoint /mnt/hdd_1/home/cs16/Model/output/VLM/v5.2 \
  --dataset /mnt/hdd_1/home/cs16/Data/dataAB_v5/vlm_eval_A_v5.jsonl \
  --val_dataset /mnt/hdd_1/home/cs16/Data/dataAB_v5/vlm_eval_A_v5.jsonl \
  --output_dir /mnt/hdd_1/home/cs16/Data/dataAB_v5/output \
  --deepspeed "" \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --enable_vision_lora false \
  --enable_text_lora true \
  --eval_dump_predictions /mnt/hdd_1/home/cs16/Data/dataAB_v5/vlm_infer_A_v5.2.jsonl \
  --eval_max_new_tokens 512 \
  --eval_generate_batch_size 1 \
  --max_steps 0 --epochs 0 --save_total_limit 1 \
  --skip_save \
  --label_threshold 0.3 \
  --lambda_lm 1.0 --lambda_lm_title 0.5 --lambda_lm_ing 0.1 \
  --lambda_cuisine 0.3 --lambda_meal 0.3 --lambda_dish 1.0 \
  --lambda_amount 0.0 --lambda_ratio 0.0 --lambda_hinge 0.0
