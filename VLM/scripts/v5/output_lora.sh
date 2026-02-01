export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export NPROC_PER_NODE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=$PBS_JOBFS/triton-cache

mkdir -p "$TRITON_CACHE_DIR"

TORCHRUN=${TORCHRUN:-/scratch/li96/zl9731/envs/ms-swift/1.0.0/bin/torchrun}

LOG_DIR=/scratch/li96/zl9731/cs16/Model/output/VLM/v5.4-3-lora
mkdir -p "$LOG_DIR"
nvidia-smi > "$LOG_DIR/nvidia-smi.log" 2>&1
(while sleep 600; do nvidia-smi >> "$LOG_DIR/nvidia-smi.log" 2>&1; done) &
SMI_PID=$!
trap 'kill $SMI_PID' EXIT

${TORCHRUN} --nproc_per_node=${NPROC_PER_NODE} /scratch/li96/zl9731/cs16/vri-food/VLM/train/trainer.py \
  --dataset $PBS_JOBFS/data/vlm_train_AB_v5.jsonl \
  --output_dir /scratch/li96/zl9731/cs16/Model/output/VLM/v5.4-3-lora \
  --deepspeed "" \
  --base_model /scratch/li96/zl9731/cs16/Model/output/VLM/v3-20251222-211237/checkpoint-7633-merged \
  --init_from_checkpoint /scratch/li96/zl9731/cs16/Model/output/VLM/v5.4-3 \
  --processor_base /scratch/li96/zl9731/cs16/Model/output/VLM/v5.4-3 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --dataloader_num_workers 12 \
  --enable_vision_lora true \
  --enable_text_lora true \
  --val_dataset $PBS_JOBFS/data/vlm_val_A_v5_head1.jsonl \
  --max_steps 1 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.05 \
  --logging_steps 1 \
  --eval_steps 1 \
  --save_steps 1 \
  --save_total_limit 1 \
  --save_lora_only \
  --lambda_lm 0.8 --lambda_lm_title 0.05 --lambda_lm_ing 0.8 \
  --lambda_cuisine 0.005 --lambda_meal 0.005 --lambda_dish 0.01 \
  --lambda_amount 0.4 --lambda_ratio 0.5 --lambda_hinge 0.4

# Export multitask heads from v5.4-3 into the LoRA output dir
python3 /scratch/li96/zl9731/cs16/vri-food/VLM/train/export_multitask_heads.py
