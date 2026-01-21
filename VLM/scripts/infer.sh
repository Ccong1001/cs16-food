CUDA_VISIBLE_DEVICES=0 \
IMAGE_MAX_TOKEN_NUM=1024 \

swift infer \
    --model /mnt/hdd_1/home/cs16/Model/Qwen3-VL-8B-Instruct \
    --adapters /mnt/hdd_1/home/cs16/Model/output/VLM/v2-20251220-175455/checkpoint-2078 \
    --merge_lora false \
    --stream true