# VLM Deployment

This folder contains inference and RAG integration scripts for single-image outputs in structured JSON.

## Files
- inference_example.py: LM-only inference
- inference_with_ratio.py: LM + ratio inference (JSON output)
- run_rag_on_vlm.py: send VLM JSON to RAG (canonical + foodkey)

## Inference (with ratio + JSON output)
Set DATASET_IMPL and output JSON:

DATASET_IMPL=dataset_v5 python3 /scratch/li96/zl9731/cs16/vri-food/VLM/deployment/inference_with_ratio.py \
  /scratch/li96/zl9731/cs16/Model/output/VLM/v6.0 \
  /jobfs/159727058.gadi-pbs/data/images_448/1137_18f837b4bb3d1e400979a7957bd5ece8.jpg \
  --base_model /scratch/li96/zl9731/cs16/Model/output/VLM/v3-20251222-211237/checkpoint-7633-bnb-int4 \
  --use_4bit \
  --device cuda:0 \
  --output_json /scratch/li96/zl9731/cs16/vri-food/VLM/deployment/vlm_output.json


Output: `vlm_output.json`

## One-shot RAG (canonical + foodkey)

python3 /scratch/li96/zl9731/cs16/vri-food/VLM/deployment/run_rag_on_vlm.py \
  --input /scratch/li96/zl9731/cs16/vri-food/VLM/deployment/vlm_output.json \
  --output /scratch/li96/zl9731/cs16/vri-food/VLM/deployment/rag_output.json \
  --device cuda:0 \
  --emb_model_dir /scratch/li96/zl9731/cs16/Model/Qwen3-Embedding-0.6B \
  --expand_local_model_path /scratch/li96/zl9731/cs16/Model/Qwen3-8B \
  --kb /scratch/li96/zl9731/cs16/RAG/ausnut_kb_measures.tagged.jsonl

Output: `rag_output.json`

