# This will store the full, unquantized weights.
swift export \
    --model /scratch/li96/zl9731/cs16/Model/Qwen3-VL-8B-Instruct \
    --adapters /scratch/li96/zl9731/cs16/Model/output/VLM/v3-20251222-211237/checkpoint-7633 \
    --merge_lora true