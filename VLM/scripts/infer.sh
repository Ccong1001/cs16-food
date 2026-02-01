CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \

swift infer \
    --model_type qwen3_vl \
    --model /scratch/li96/zl9731/cs16/Model/output/VLM/v5.3-2  \
    --template qwen3_vl \
    --stream true