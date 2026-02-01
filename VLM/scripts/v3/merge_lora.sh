# This will store the full, unquantized weights.
swift export \
    --model_type qwen3_vl \
    --model /scratch/li96/zl9731/cs16/Model/output/VLM/v5.3-2 \
    --adapters /scratch/li96/zl9731/cs16/Model/output/VLM/test-2 \
    --template qwen3_vl \
    --merge_lora true