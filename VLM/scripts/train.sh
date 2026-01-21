export CUDA_VISIBLE_DEVICES=1,2,3
export OMP_NUM_THREADS=4
export NPROC_PER_NODE=3

swift sft \
--deepspeed 'zero2' \
--model /mnt/hdd_1/home/cs16/Model/Qwen3-VL-8B-Instruct \
--dataset /mnt/hdd_1/home/cs16/Data/dataAB_g/vlm_train_AB.jsonl \
--val_dataset /mnt/hdd_1/home/cs16/Data/dataA_g/vlm_eval_A.jsonl \
--train_type lora \
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
--output_dir /mnt/hdd_1/home/cs16/Model/output/VLM \
--num_train_epochs 1 \
--save_steps 1000 \
--eval_steps 1000 \
--save_total_limit 10 \
--logging_steps 50 \
--seed 42 \
--learning_rate 5e-5 \
--init_weights true \
--target_modules='all-linear' \
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
--model_name vlm_swift_qlora_v1 \