#!/usr/bin/env python
# -*- coding: utf-8 -*-

# CUDA_VISIBLE_DEVICES=2 python /mnt/hdd_1/home/cs16/vri-food/T1/infer.py --shard 0 --world_size 2

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
from tqdm import tqdm

# ================== 参数 & 常量 ==================

MODEL_DIR = "/mnt/hdd_1/home/cs16/Model/output/T1/v7-20251209-232320/checkpoint-1110-merged"

INSTRUCTION = (
    "You are a recipe classifier.\n"
    "Given the recipe text below, extract the following fields:\n"
    "- cuisine_type\n"
    "- dish_type\n"
    "- meal_type\n"
    "Return JSON only. Do not invent labels. "
    "Only output labels that appear in the dataset schema."
)

# 从刚才生成的 T1_C_input.jsonl 读
IN_PATH = Path("/mnt/hdd_1/home/cs16/Data/dataB_recipe/T1_B_input.jsonl")
# 推理结果写到这里（每个 shard 一个文件）
OUT_PATH = Path("/mnt/hdd_1/home/cs16/Data/dataB_recipe/T1_B_result.jsonl")

BATCH_SIZE = 128
MAX_NEW_TOKENS = 64

# ================== 模型加载 ==================

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    model.eval()
    return tokenizer, model, device


def build_messages(recipe_text: str):
    return [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": recipe_text},
    ]

# ================== 批量推理 ==================

@torch.inference_mode()
def batch_parse(tokenizer, model, device, recipe_list):
    messages_list = [build_messages(t) for t in recipe_list]
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=768,  # 标题 + 配料，768 够用
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    results = []
    input_len = inputs["input_ids"].shape[1]

    for i in range(len(recipe_list)):
        gen_ids = outputs[i, input_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # 尝试从中间截取第一个 {...}
            txt = text.strip()
            if "{" in txt and "}" in txt:
                sub = txt[txt.find("{"): txt.rfind("}") + 1]
                try:
                    parsed = json.loads(sub)
                except json.JSONDecodeError:
                    parsed = None

        results.append({"raw_output": text, "parsed": parsed})

    return results

# ================== 主逻辑（带 shard） ==================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0, help="当前进程的 shard id，从 0 开始")
    parser.add_argument("--world_size", type=int, default=1, help="总 shard 数（= 并行进程数）")
    args = parser.parse_args()

    assert 0 <= args.shard < args.world_size, "shard 必须在 [0, world_size) 范围内"

    print(f"[T1 infer] shard = {args.shard}, world_size = {args.world_size}")
    print(f"[T1 infer] input  = {IN_PATH}")

    tokenizer, model, device = load_model()

    total_lines = sum(1 for _ in IN_PATH.open("r", encoding="utf-8"))
    print(f"[T1 infer] Total lines (all shards): {total_lines}")

    # 每个 shard 单独写一个文件
    out_path_shard = OUT_PATH.with_name(
        OUT_PATH.stem + f".shard{args.shard}" + OUT_PATH.suffix
    )
    out_path_shard.parent.mkdir(parents=True, exist_ok=True)
    print(f"[T1 infer] This shard output -> {out_path_shard}")

    out_f = out_path_shard.open("w", encoding="utf-8")

    buf = []
    buf_meta = []
    line_idx = 0

    with IN_PATH.open("r", encoding="utf-8") as fin, tqdm(
        total=total_lines, desc=f"T1 shard {args.shard}"
    ) as pbar:
        for line in fin:
            pbar.update(1)
            line = line.strip()
            if not line:
                continue

            # shard 切分：只处理本 shard 的行
            if line_idx % args.world_size != args.shard:
                line_idx += 1
                continue

            obj = json.loads(line)
            # 和训练对齐：字段名叫 input
            recipe_text = obj.get("input")

            if recipe_text is None:
                obj["parsed"] = None
                obj["model_raw_output"] = ""
                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                line_idx += 1
                continue

            buf.append(recipe_text)
            buf_meta.append(obj)

            if len(buf) == BATCH_SIZE:
                results = batch_parse(tokenizer, model, device, buf)
                for meta, res in zip(buf_meta, results):
                    meta["model_raw_output"] = res["raw_output"]
                    meta["parsed"] = res["parsed"]
                    out_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                buf, buf_meta = [], []

            line_idx += 1

        # 收尾
        if buf:
            results = batch_parse(tokenizer, model, device, buf)
            for meta, res in zip(buf_meta, results):
                meta["model_raw_output"] = res["raw_output"]
                meta["parsed"] = res["parsed"]
                out_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    out_f.close()
    print(f"[T1 infer] Done shard {args.shard}. Saved to {out_path_shard}")


if __name__ == "__main__":
    main()
