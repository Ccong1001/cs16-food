# This will store the full, unquantized weights.
swift export \
    --model /mnt/hdd_1/home/cs16/Model/Qwen3-VL-8B-Instruct \
    --adapters /mnt/hdd_1/home/cs16/Model/output/VLM/v3-20251222-211237/checkpoint-7633 \
    --merge_lora true