export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export NPROC_PER_NODE=1
export TOKENIZERS_PARALLELISM=false

TORCHRUN=${TORCHRUN:-/mnt/hdd_1/home/cs16/miniconda3/envs/swift/bin/torchrun}


LOSS_SCHEDULE='[
  {"start":0,"end":1000,"lambda_lm_title":0.3,"lambda_lm_ing":0.5},
  {"start":1000,"end":2000,"lambda_lm_title":0.3,"lambda_lm_ing":1.0}
]'

${TORCHRUN} --nproc_per_node=${NPROC_PER_NODE} vri-food/VLM/train/trainer.py \
  --dataset /mnt/hdd_1/home/cs16/Data/dataAB_v5/vlm_train_AB_v5.jsonl \
  --output_dir /mnt/hdd_1/home/cs16/Model/output/VLM/v5.2-2 \
  --deepspeed /mnt/hdd_1/home/cs16/vri-food/VLM/train/deepspeed_zero2.json \
  --base_model /mnt/hdd_1/home/cs16/Model/output/VLM/v3-20251222-211237/checkpoint-7633-merged \
  --init_from_checkpoint /mnt/hdd_1/home/cs16/Model/output/VLM/v5.2-1 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --enable_vision_lora false \
  --enable_text_lora true \
  --val_dataset /mnt/hdd_1/home/cs16/Data/dataAB_v5/vlm_val_A_v5.jsonl \
  --max_steps 2000 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.1 \
  --logging_steps 128 \
  --eval_steps 500 \
  --save_steps 2000 \
  --save_total_limit 3 \
  --lambda_lm 1.0 --lambda_lm_title 0.3 --lambda_lm_ing 1.0 \
  --lambda_cuisine 0.1 --lambda_meal 0.1 --lambda_dish 0.5 \
  --lambda_amount 0.0 --lambda_ratio 0.0 --lambda_hinge 0.0 \
  --loss_schedule "$LOSS_SCHEDULE"