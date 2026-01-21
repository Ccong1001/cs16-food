export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export NPROC_PER_NODE=1

swift sft \
  --model /mnt/hdd_1/home/cs16/Model/Qwen3-1.7B \
  --check_model false \
  --dataset /mnt/hdd_1/home/cs16/Data/dataA_recipe/T1_prompt.jsonl \
  --split_dataset_ratio 0.1 \
  --train_type lora \
  --quant_method bnb \
  --quant_bits 4 \
  --bnb_4bit_compute_dtype bfloat16 \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_use_double_quant true \
  --torch_dtype bfloat16 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lorap_lr_ratio 10 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_length 1024 \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --warmup_steps 0 \
  --gradient_checkpointing false \
  --save_steps 1000 \
  --eval_steps 1000 \
  --save_total_limit 1 \
  --logging_steps 100 \
  --output_dir /mnt/hdd_1/home/cs16/Model/output/T1 \
  --model_name T1_qlora_v8 \
  --seed 42 \
  --init_weights true
