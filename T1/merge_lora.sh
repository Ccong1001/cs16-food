# This will store the full, unquantized weights.
swift export \
    --model /mnt/hdd_1/home/cs16/model/Qwen3-1.7B \
    --adapters /mnt/hdd_1/home/cs16/model/output/T1/v7-20251209-232320/checkpoint-1110 \
    --merge_lora true