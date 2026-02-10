# Build DPO Pairs
```bash
python /mnt/hdd_1/home/cs16/vri-food/VLM/rlhf/build_dpo_pairs.py \
  --true /mnt/hdd_1/home/cs16/vri-food/VLM/rlhf/data/true_sample.json \
  --candidates /mnt/hdd_1/home/cs16/vri-food/VLM/rlhf/data/multi_samples.jsonl \
  --candidates_rag /mnt/hdd_1/home/cs16/vri-food/VLM/rlhf/data/multi_samples_rag.jsonl \
  --output /mnt/hdd_1/home/cs16/vri-food/VLM/rlhf/data/dpo_pairs.jsonl \
  --lowercase_name
```

# RLHF Utilities

**1) Multi-sample inference (same image, multiple generations)**
```bash
BASE_DIR=/mnt/hdd_1/home/cs16 \
DATASET_IMPL=dataset_v5 python3 /mnt/hdd_1/home/cs16/vri-food/VLM/rlhf/multi_sample_infer.py \
  /mnt/hdd_1/home/cs16/Model/output/VLM/v5.4-3 \
  /mnt/hdd_1/home/cs16/demo_webpage/food_img/Corndogs-7832ef6.jpg \
  --base_model /mnt/hdd_1/home/cs16/Model/output/VLM/v3-20251222-211237/checkpoint-7633-merged \
  --device cuda:0 \
  --num_samples 8 \
  --temperature 0.7 \
  --top_p 0.9 \
  --output /mnt/hdd_1/home/cs16/vri-food/VLM/rlhf/multi_samples.jsonl
```

**2) Run RAG on multi-sample JSONL**
```bash
BASE_DIR=/mnt/hdd_1/home/cs16 \
python3 /mnt/hdd_1/home/cs16/vri-food/VLM/rlhf/run_rag_on_multi_samples.py \
  --input /mnt/hdd_1/home/cs16/vri-food/VLM/rlhf/multi_samples.jsonl \
  --output /mnt/hdd_1/home/cs16/vri-food/VLM/rlhf/multi_samples_rag.jsonl \
  --device cuda:0 \
  --emb_model_dir /mnt/ssd_2/cs16/model/Qwen3-Embedding-0.6B \
  --expand_local_model_path /mnt/ssd_2/cs16/model/Qwen3-8B \
  --kb /mnt/hdd_1/home/cs16/RAG/ausnut_kb_measures.tagged.jsonl
```
