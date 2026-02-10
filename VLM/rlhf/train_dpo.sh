#!/usr/bin/env bash
set -euo pipefail

# 24GiB
# It is recommended to use padding_free. For more details, please refer to:
# https://github.com/modelscope/ms-swift/blob/main/examples/train/padding_free/dpo.sh

export CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=4
export NPROC_PER_NODE=1
export TOKENIZERS_PARALLELISM=false

swift rlhf \
--rlhf_type dpo \
--deepspeed 'zero2' \
--model /mnt/hdd_1/home/cs16/Model/output/VLM/v3-20251222-211237 \
--tuner_type lora \
--dataset /mnt/hdd_1/home/cs16/vri-food/VLM/rlhf/dpo_pairs.jsonl \
--load_from_cache_file true \
--split_dataset_ratio 0.01 \
--torch_dtype bfloat16 \
--bnb_4bit_compute_dtype bfloat16 \
--bnb_4bit_quant_type nf4 \
--bnb_4bit_use_double_quant true \
--quant_method bnb \
--quant_bits 4 \
--lorap_lr_ratio 10 \
--freeze_vit true \
--freeze_aligner false \
--freeze_llm false \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 32 \
--max_length 2048 \
--output_dir /mnt/hdd_1/home/cs16/Model/output/VLM/rlhf_dpo \
--num_train_epochs 1 \
--save_steps 1000 \
--eval_steps 1000 \
--save_total_limit 2 \
--logging_steps 50 \
--seed 42 \
--learning_rate 5e-5 \
--target_modules all-linear \
--lora_rank 32 \
--lora_alpha 64 \
--adam_beta1 0.9 \
--adam_beta2 0.95 \
--adam_epsilon 1e-08 \
--weight_decay 0.1 \
--max_grad_norm 1 \
--lr_scheduler_type cosine \
--warmup_ratio 0.05 \
--warmup_steps 0 \
--gradient_checkpointing true \
--rpo_alpha 0.1 \
--dataloader_num_workers 4 \
--dataset_num_proc 4 \
--model_name vlm_swift_qlora_dpo_v1
