export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export NPROC_PER_NODE=4
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=$PBS_JOBFS/triton-cache

mkdir -p "$TRITON_CACHE_DIR"

TORCHRUN=${TORCHRUN:-/scratch/li96/zl9731/envs/ms-swift/1.0.0/bin/torchrun}

LOG_DIR=/scratch/li96/zl9731/cs16/Model/output/VLM/v6.1
mkdir -p "$LOG_DIR"
nvidia-smi > "$LOG_DIR/nvidia-smi.log" 2>&1
(while sleep 600; do nvidia-smi >> "$LOG_DIR/nvidia-smi.log" 2>&1; done) &
SMI_PID=$!
trap 'kill $SMI_PID' EXIT

${TORCHRUN} --nproc_per_node=${NPROC_PER_NODE} /scratch/li96/zl9731/cs16/vri-food/VLM/train/trainer.py \
  --dataset $PBS_JOBFS/data/vlm_train_AB_v5.jsonl \
  --val_dataset /scratch/li96/zl9731/cs16/Data/dataAB_v5/vlm_val_A_v5.jsonl \
  --output_dir /scratch/li96/zl9731/cs16/Model/output/VLM/v6.1 \
  --deepspeed /scratch/li96/zl9731/cs16/vri-food/VLM/train/deepspeed_zero2.json \
  --base_model /scratch/li96/zl9731/cs16/Model/output/VLM/v3-20251222-211237/checkpoint-7633-merged \
  --init_from_checkpoint /scratch/li96/zl9731/cs16/Model/output/VLM/v6.0 \
  --save_lora_only \
  --train_heads false \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --train_batch_size 64 \
  --gradient_accumulation_steps 2 \
  --enable_vision_lora true \
  --enable_text_lora true \
  --train_heads false \
  --epochs 2 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.1 \
  --logging_steps 100 \
  --eval_steps 1000 \
  --save_steps 4000 \
  --save_total_limit 3 \
  --lambda_lm 1.0 --lambda_lm_title 0.3 --lambda_lm_ing 1.0 \
  --lambda_cuisine 0.0 --lambda_meal 0.0 --lambda_dish 0.0 \
  --lambda_amount 0.0 --lambda_ratio 0.0 --lambda_hinge 0.0 \
  --loss_schedule "$LOSS_SCHEDULE"
