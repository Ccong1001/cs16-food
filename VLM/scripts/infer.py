#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CUDA_VISIBLE_DEVICES=0 \
IMAGE_MAX_TOKEN_NUM=1024 \
python vri-food/VLM/infer.py \
  --base_model /mnt/hdd_1/home/cs16/Model/Qwen3-VL-8B-Instruct \
  --no_lora \
  --input /mnt/hdd_1/home/cs16/Data/dataA_g/vlm_eval_A_base.jsonl \
  --output /mnt/hdd_1/home/cs16/Data/dataA_g/vlm_infer_A_base.jsonl \
  --batch_size 32 \
  --device_map auto \
  --max_new_tokens 2048

---

CUDA_VISIBLE_DEVICES=1,2,3 \
IMAGE_MAX_TOKEN_NUM=1024 \
python vri-food/VLM/infer.py \
  --input /mnt/hdd_1/home/cs16/Data/dataA_g/vlm_eval_A.jsonl \
  --output /mnt/hdd_1/home/cs16/Data/dataA_g/vlm_infer_A_v3.jsonl \
  --batch_size 32 \
  --device_map auto \
  --max_new_tokens 2048

CUDA_VISIBLE_DEVICES=1,2,3 \
IMAGE_MAX_TOKEN_NUM=1024 \
python vri-food/VLM/infer.py \
  --input /mnt/hdd_1/home/cs16/Data/dataA_g/vlm_train_A.jsonl \
  --output /mnt/hdd_1/home/cs16/Data/dataA_g/vlm_train_infer_A_v3.jsonl \
  --batch_size 32 \
  --device_map auto \
  --max_new_tokens 2048

-----------

CUDA_VISIBLE_DEVICES=0 \
IMAGE_MAX_TOKEN_NUM=1024 \
python vri-food/VLM/infer.py \
  --input /mnt/hdd_1/home/cs16/Data/dataAB_v4/vlm_eval_visible_A.jsonl \
  --output /mnt/hdd_1/home/cs16/Data/dataA_g/vlm_infer_A_v4.jsonl \
  --ckpt_dir /mnt/hdd_1/home/cs16/Model/output/VLM/v4-20251229-222658/checkpoint-10000 \
  --batch_size 8 \
  --device_map auto \
  --max_new_tokens 2048
"""


import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# 默认路径，可在命令行参数里覆盖
BASE_MODEL = "/mnt/hdd_1/home/cs16/Model/Qwen3-VL-8B-Instruct"
CKPT_DIR = "/mnt/hdd_1/home/cs16/Model/output/VLM/v3-20251222-211237/checkpoint-7633"
EVAL_JSONL = "/mnt/hdd_1/home/cs16/Data/dataA_g/vlm_eval_A.jsonl"
DEFAULT_OUT = "/mnt/hdd_1/home/cs16/Data/dataA_g/vlm_infer_A_v3.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM 推理脚本：读取 JSONL，逐条生成回复并保存。")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL, help="基座模型路径")
    parser.add_argument("--ckpt_dir", type=str, default=CKPT_DIR, help="LoRA 检查点目录，要求内含 adapter_model.safetensors")
    parser.add_argument("--no_lora", action="store_true", help="仅用基座模型，不加载 LoRA adapter")
    parser.add_argument("--input", type=str, default=EVAL_JSONL, help="输入 JSONL，每行包含 images + messages")
    parser.add_argument("--output", type=str, default=DEFAULT_OUT, help="输出 JSONL，写入模型生成文本")
    parser.add_argument("--batch_size", type=int, default=1, help="推理 batch 大小，显存足够时可加大")
    parser.add_argument("--device_map", type=str, default="auto", help="模型切分策略：auto/balanced/balanced_low_0 或单卡 id")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="生成最大新 token 数")
    parser.add_argument("--temperature", type=float, default=0.2, help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="生成 top-p")
    return parser.parse_args()


def parse_device_map(device_map_arg: str):
    """支持 auto/balanced/balanced_low_0，或单个数字表示指定 GPU。"""
    if device_map_arg is None:
        return "auto"
    lower = device_map_arg.strip().lower()
    if lower in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return lower
    # 单卡 id
    try:
        gpu_id = int(lower)
        return {"": gpu_id}
    except ValueError:
        return "auto"


def build_model(base_model_path: str, ckpt_dir: str, device_map_arg: str, use_lora: bool):
    use_cuda = torch.cuda.is_available()
    device_map = parse_device_map(device_map_arg)
    max_memory = None
    if isinstance(device_map, str) and device_map in {"auto", "balanced", "balanced_low_0"} and torch.cuda.device_count() > 1:
        # 让 accelerate 自动切多卡，默认给每张卡 22GiB 配额，可按需调整
        max_memory = {i: "22GiB" for i in range(torch.cuda.device_count())}

    if use_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
        )
    else:
        # CPU 模式用全精度，避免 bf16 不支持的问题
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            trust_remote_code=True,
        )

    if use_lora:
        # 使用训练时保存的 LoRA 配置与权重
        model = PeftModel.from_pretrained(base_model, ckpt_dir)
    else:
        model = base_model

    model.eval()
    return model


def load_image(image_path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(image_path).convert("RGB")
        return img
    except Exception:
        return None


def extract_user_content(item: Dict) -> str:
    messages = item.get("messages") or []
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return content if isinstance(content, str) else ""
    return ""


def build_mm_messages(user_content: str, image: Image.Image) -> Dict:
    # 构造多模态 messages，文本去掉 <image> 占位符
    text_only = user_content.replace("<image>", "").strip()
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_only},
            ],
        }
    ]


def generate_batch(
    model,
    processor,
    batch_data: List[Tuple[Dict, Image.Image, str]],
    args: argparse.Namespace,
) -> List[str]:
    prompts = []
    images = []
    for _, image, user_content in batch_data:
        messages = build_mm_messages(user_content, image)
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
        images.append(image)

    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    do_sample = args.temperature > 0
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})

    output_ids = model.generate(**inputs, **gen_kwargs)
    gen = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return [g.strip() for g in gen]


def infer(args: argparse.Namespace) -> None:
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = "left"
    model = build_model(args.base_model, args.ckpt_dir, args.device_map, use_lora=not args.no_lora)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = sum(1 for _ in in_path.open("r", encoding="utf-8"))
    skipped_json = skipped_img = skipped_msg = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        buf: List[Tuple[Dict, Image.Image, str]] = []
        for line in tqdm(fin, desc="Infer", total=total_lines):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped_json += 1
                continue

            images = item.get("images") or []
            img_path = None
            if images:
                first = images[0]
                if isinstance(first, dict):
                    img_path = first.get("path")
                elif isinstance(first, str):
                    img_path = first
            if not img_path:
                skipped_img += 1
                continue

            image = load_image(img_path)
            if image is None:
                skipped_img += 1
                continue

            user_content = extract_user_content(item)
            if not user_content:
                skipped_msg += 1
                continue

            buf.append((item, image, user_content))

            # 满 batch 执行一次推理
            if len(buf) >= max(1, args.batch_size):
                outputs = generate_batch(model, processor, buf, args)
                for (meta, _, _), out_text in zip(buf, outputs):
                    meta["model_output"] = out_text
                    fout.write(json.dumps(meta, ensure_ascii=False) + "\n")
                buf = []

        # 处理剩余不足一个 batch 的数据
        if buf:
            outputs = generate_batch(model, processor, buf, args)
            for (meta, _, _), out_text in zip(buf, outputs):
                meta["model_output"] = out_text
                fout.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(
        f"[Done] 推理完成，结果已写入: {out_path} "
        f"(json错误 {skipped_json}, 无图/读图失败 {skipped_img}, 无user消息 {skipped_msg})"
    )


def main():
    args = parse_args()
    infer(args)


if __name__ == "__main__":
    main()
