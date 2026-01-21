#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# CUDA_VISIBLE_DEVICES=0 python /mnt/hdd_1/home/cs16/vri-food/T0/infer.py --shard 0 --world_size 8 --batch_size 128 --max_new_tokens 128
# CUDA_VISIBLE_DEVICES=1 python /mnt/hdd_1/home/cs16/vri-food/T0/infer.py --shard 1 --world_size 8  --batch_size 128 --max_new_tokens 128
# CUDA_VISIBLE_DEVICES=2 python /mnt/hdd_1/home/cs16/vri-food/T0/infer.py --shard 2 --world_size 8 --batch_size 128 --max_new_tokens 128
# CUDA_VISIBLE_DEVICES=3 python /mnt/hdd_1/home/cs16/vri-food/T0/infer.py --shard 3 --world_size 8 --batch_size 128 --max_new_tokens 128
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_DIR = Path("/mnt/hdd_1/home/cs16/Model/output/T0/v1-20251221-230803/checkpoint-9917-merged")
DEFAULT_INPUT = Path("/mnt/hdd_1/home/cs16/Data/dataB_ingredient/T0_B_infer.jsonl")
DEFAULT_OUTPUT = Path("/mnt/hdd_1/home/cs16/Data/dataB_ingredient/T0_B_infer_result.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T0 inference with messages-only JSONL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 JSONL，包含 messages")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 JSONL，写入 model_raw_output/parsed")
    parser.add_argument("--shard", type=int, default=0, help="当前进程 shard id，从 0 开始")
    parser.add_argument("--world_size", type=int, default=1, help="总 shard 数")
    parser.add_argument("--batch_size", type=int, default=64, help="推理 batch 大小")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="生成最大新 token 数")
    return parser.parse_args()


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return tokenizer, model, device


def attempt_parse(text: str) -> Optional[Any]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


@torch.inference_mode()
def batch_generate(tokenizer, model, device, messages_list: List[List[Dict[str, Any]]], max_new_tokens: int):
    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    input_len = inputs["input_ids"].shape[1]
    texts = [
        tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
        for out in outputs
    ]
    parsed = [attempt_parse(t) for t in texts]
    return texts, parsed


def main():
    args = parse_args()
    assert 0 <= args.shard < args.world_size, "shard 必须在 [0, world_size) 范围内"

    tokenizer, model, device = load_model()

    total_lines = sum(1 for _ in args.input.open("r", encoding="utf-8"))
    out_path_shard = args.output.with_name(
        args.output.stem + f".shard{args.shard}" + args.output.suffix
    )
    out_path_shard.parent.mkdir(parents=True, exist_ok=True)

    buf_msgs: List[List[Dict[str, Any]]] = []
    buf_meta: List[Dict[str, Any]] = []
    line_idx = 0

    with args.input.open("r", encoding="utf-8") as fin, \
            out_path_shard.open("w", encoding="utf-8") as fout, \
            tqdm(total=total_lines, desc=f"T0 shard {args.shard}") as pbar:
        for line in fin:
            pbar.update(1)
            line = line.strip()
            if not line:
                line_idx += 1
                continue
            obj = json.loads(line)

            if line_idx % args.world_size != args.shard:
                line_idx += 1
                continue

            messages = obj.get("messages")
            if not isinstance(messages, list) or not messages:
                obj["model_raw_output"] = ""
                obj["parsed"] = None
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                line_idx += 1
                continue

            buf_msgs.append(messages)
            buf_meta.append(obj)

            if len(buf_msgs) == args.batch_size:
                texts, parsed = batch_generate(tokenizer, model, device, buf_msgs, args.max_new_tokens)
                for meta, t, p in zip(buf_meta, texts, parsed):
                    meta["model_raw_output"] = t
                    meta["parsed"] = p
                    fout.write(json.dumps(meta, ensure_ascii=False) + "\n")
                buf_msgs, buf_meta = [], []

            line_idx += 1

        if buf_msgs:
            texts, parsed = batch_generate(tokenizer, model, device, buf_msgs, args.max_new_tokens)
            for meta, t, p in zip(buf_meta, texts, parsed):
                meta["model_raw_output"] = t
                meta["parsed"] = p
                fout.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"[T0 infer] Done shard {args.shard}. Saved to {out_path_shard}")


if __name__ == "__main__":
    main()
