#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiTaskVLM 推理代码 - 支持提取 ratio 信息

用途：
1. 加载完整的 MultiTaskVLM 模型（包含分类头）
2. 同时获取生成文本和 ratio logits
3. 用于需要食材比例信息的场景
"""

import sys
import os
import json
import re
import importlib
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from typing import Dict, List, Optional, Tuple

# 添加 train 目录到路径以导入 model.py
CURRENT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = CURRENT_DIR.parent / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

try:
    from model import MultiTaskVLM  # type: ignore
except ImportError:
    print("警告: 无法导入 MultiTaskVLM，将使用基座模型进行推理")
    MultiTaskVLM = None

_dataset_impl = os.environ.get("DATASET_IMPL", "dataset")
try:
    _dataset_module = importlib.import_module(_dataset_impl)
except Exception:
    _dataset_module = importlib.import_module("dataset")

USER_PROMPT_TEXT = getattr(
    _dataset_module,
    "USER_PROMPT_TEXT",
    "Given a food image, return ONLY a JSON object with this schema:\n"
    "{\n"
    '  "title": "<title>",\n'
    '  "ingredients": [\n'
    '    {"name": "<name>", "note": "<note>"}\n'
    "  ]\n"
    "}\n"
    "Do not output cuisine/meal/dish labels, amounts, ratios, or any extra text.\n"
    "Output valid JSON only (no markdown/code fences).",
)


def load_multitask_model(
    checkpoint_path: str,
    base_model_path: Optional[str] = None,
    processor_base: Optional[str] = None,
    device: str = "cuda:0",
    use_4bit: bool = False
):
    """
    加载完整的 MultiTaskVLM 模型
    
    Args:
        checkpoint_path: checkpoint 或 merged 模型路径
        base_model_path: 基座模型路径（若 checkpoint 为 LoRA adapter 则必须提供）
        processor_base: Processor 路径（默认 checkpoint_path）
        device: 设备
        use_4bit: 是否使用 4bit 量化
        
    Returns:
        model, processor
    """
    if MultiTaskVLM is None:
        raise ImportError("无法导入 MultiTaskVLM，请检查 train/model.py 是否存在")
    
    checkpoint_path = Path(checkpoint_path)
    print(f"加载分类头权重目录: {checkpoint_path}")
    if not base_model_path:
        raise ValueError("必须指定 --base_model，指向合并后的主模型权重目录（如 Qwen3-VL-8B-Instruct）！")

    # 加载 processor
    processor_src = processor_base or str(base_model_path)
    processor = AutoProcessor.from_pretrained(processor_src, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = "right"

    # 配置量化
    quant_config = None
    if use_4bit and device != "cpu":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # 加载主模型（合并后的 Qwen3-VL-8B-Instruct）
    if device != "cpu":
        # V100 只支持 float16
        dtype = torch.float16
    else:
        dtype = torch.float32
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        dtype=dtype,
        quantization_config=quant_config,
        device_map=device if device != "cpu" else {"": "cpu"},
        trust_remote_code=True
    )

    # 包装为 MultiTaskVLM
    hidden_size = getattr(base_model.config, "hidden_size", None)
    if hidden_size is None and hasattr(base_model.config, "text_config"):
        hidden_size = getattr(base_model.config.text_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("无法确定 hidden_size")
    model = MultiTaskVLM(
        base_model=base_model,
        hidden_size=hidden_size,
        num_cuisine=22,
        num_meal=4,
        num_dish=20,
        num_amount_levels=5
    )

    # 加载分类头权重
    state_dict_path = checkpoint_path / "model.safetensors"
    if not state_dict_path.exists():
        state_dict_path = checkpoint_path / "pytorch_model.bin"
    if state_dict_path.exists():
        print(f"加载分类头权重: {state_dict_path}")
        if state_dict_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(str(state_dict_path))
        else:
            state_dict = torch.load(str(state_dict_path), map_location="cpu")
        head_state = {
            k: v for k, v in state_dict.items()
            if any(h in k for h in ["cuisine_head", "meal_head", "dish_head", "amount_head", "ratio_head"])
        }
        if head_state:
            model.load_state_dict(head_state, strict=False)
            print(f"✓ 加载了 {len(head_state)} 个分类头参数")
        else:
            print("⚠️  警告: 未找到分类头权重，将使用随机初始化")
    else:
        print(f"⚠️  警告: 未找到权重文件: {state_dict_path}")

    model = model.to(device)
    model = model.to(dtype)
    model.eval()
    if hasattr(model.model.config, "use_cache"):
        model.model.config.use_cache = True

    return model, processor


def tokenize_ingredients(
    ingredient_names: List[str],
    processor,
    max_ingredients: int = 20,
    max_token_len: int = 16
) -> Dict[str, torch.Tensor]:
    """
    将食材名称转为 token IDs
    
    Args:
        ingredient_names: 食材名称列表，如 ["pasta", "beef mince", ...]
        processor: 处理器
        max_ingredients: 最大食材数
        max_token_len: 每个食材的最大 token 长度
        
    Returns:
        {
            'ingredient_token_ids': (1, max_ing, max_len),
            'ingredient_token_mask': (1, max_ing, max_len),
            'ingredient_mask': (1, max_ing)
        }
    """
    tokenizer = processor.tokenizer
    
    # 初始化
    ingredient_token_ids = torch.zeros((1, max_ingredients, max_token_len), dtype=torch.long)
    ingredient_token_mask = torch.zeros((1, max_ingredients, max_token_len), dtype=torch.float)
    ingredient_mask = torch.zeros((1, max_ingredients), dtype=torch.float)
    
    for i, name in enumerate(ingredient_names[:max_ingredients]):
        tokens = tokenizer.encode(name, add_special_tokens=False, max_length=max_token_len, truncation=True)
        token_len = len(tokens)
        
        ingredient_token_ids[0, i, :token_len] = torch.tensor(tokens)
        ingredient_token_mask[0, i, :token_len] = 1.0
        ingredient_mask[0, i] = 1.0
    
    return {
        'ingredient_token_ids': ingredient_token_ids,
        'ingredient_token_mask': ingredient_token_mask,
        'ingredient_mask': ingredient_mask
    }


def _parse_ingredients_from_text(text: str) -> List[Dict[str, str]]:
    """从模型生成文本中提取食材名称和备注列表。"""
    items: List[Dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        # 格式: - <name> | <note>
        raw = line.lstrip("-").strip()
        name = ""
        note = ""
        if "|" in raw:
            parts = [p.strip() for p in raw.split("|")]
            name = parts[0] if len(parts) > 0 else ""
            note = parts[1] if len(parts) > 1 else ""
        else:
            name = raw.strip()
        if name:
            items.append({"name": name, "note": note})
    return items


def _extract_json(raw: str) -> Dict:
    try:
        match = re.search(r"```json\s*(.*?)```", raw, flags=re.S | re.I)
        content = match.group(1) if match else raw
        return json.loads(content)
    except Exception:
        return {}


def _parse_ingredients(text: str) -> List[Dict[str, str]]:
    obj = _extract_json(text)
    if isinstance(obj, dict) and "ingredients" in obj:
        items: List[Dict[str, str]] = []
        for ing in obj.get("ingredients", []) or []:
            if not isinstance(ing, dict):
                continue
            name = (ing or {}).get("name", "")
            note = (ing or {}).get("note", "")
            if name:
                items.append({"name": name, "note": note})
        if items:
            return items
    return _parse_ingredients_from_text(text)




def infer_with_ratio(
    model,
    processor,
    image_path: str,
    ingredient_names: Optional[List[str]] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.2
) -> Dict:
    """
    推理并获取 ratio 信息（两段式）
    
    1) 先用 LM 生成食材列表文本
    2) 从文本解析出食材名，再用 ratio head 计算比例
    
    Args:
        model: MultiTaskVLM 模型
        processor: 处理器
        image_path: 图片路径
        ingredient_names: 可选。若为空则从生成文本中解析
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度
        
    Returns:
        {
            'text': 生成的文本,
            'ratio_logits': ratio logits (num_ingredients,),
            'ratio_probs': ratio 概率分布 (softmax后),
            'ingredients': 食材名称列表
        }
    """
    # 加载图片
    image = Image.open(image_path).convert("RGB")
    
    # 构造输入
    prompt = USER_PROMPT_TEXT
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        padding=True
    )

    attn_len = int(inputs["attention_mask"].sum().item()) if "attention_mask" in inputs else inputs["input_ids"].shape[1]
    
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 1) 生成文本
    # 用 transformers GenerationConfig 保证参数与训练/验证一致
    from transformers import GenerationConfig
    import os
    processor_dir = getattr(processor, "pretrained_model_name_or_path", None)
    if processor_dir is None:
        processor_dir = os.environ.get("PROCESSOR_BASE", "./")
    try:
        gen_config = GenerationConfig.from_pretrained(processor_dir)
        gen_config.max_new_tokens = max_new_tokens
    except Exception:
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            temperature=temperature,
            do_sample=temperature > 0,
        )
    with torch.no_grad():
        output_ids = model.model.generate(**inputs, generation_config=gen_config)

    seq = output_ids[0]
    gen_tokens = seq[attn_len:]
    generated_text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    
    # 2) 解析食材列表（若未提供）
    ingredient_items: List[Dict[str, str]] = []
    if ingredient_names:
        ingredient_items = [{"name": n, "note": ""} for n in ingredient_names]
    else:
        ingredient_items = _parse_ingredients(generated_text)
    
    result = {
        'text': generated_text,
        'ingredients': ingredient_items
    }
    
    if not ingredient_items:
        result['ratio_logits'] = None
        result['ratio_probs'] = None
        return result
    
    # 3) Tokenize ingredients
    ingredient_texts = [f"{i.get('name', '')} {i.get('note', '')}".strip() for i in ingredient_items]
    ing_tokens = tokenize_ingredients(ingredient_texts, processor)
    for k, v in ing_tokens.items():
        inputs[k] = v.to(device)
    
    # 4) 获取 ratio logits (通过 forward pass)
    with torch.no_grad():
        outputs = model.forward(**inputs)
        ratio_logits = outputs.get('ratio_logits', None)
    
    if ratio_logits is not None:
        ratio_logits = ratio_logits[0].to(torch.float32).cpu()  # (num_ingredients,)
        valid_count = len(ingredient_items)
        ratio_logits = ratio_logits[:valid_count]
        ratio_probs = torch.softmax(ratio_logits, dim=0)
        result['ratio_logits'] = ratio_logits.numpy()
        result['ratio_probs'] = ratio_probs.numpy()
    else:
        result['ratio_logits'] = None
        result['ratio_probs'] = None
    
    return result


# ============ 使用示例 ============
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MultiTaskVLM 推理（含 Ratio）")
    parser.add_argument("checkpoint", type=str, help="分类头权重目录（如 v5.4-2）")
    parser.add_argument("image", type=str, help="图片路径")
    parser.add_argument("--base_model", type=str, required=True, help="主模型权重目录（合并后的 Qwen3-VL-8B-Instruct）")
    parser.add_argument("--processor_base", type=str, default=None, help="Processor 路径（默认使用 checkpoint）")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备，如 cuda:0 或 cpu")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度（默认关闭采样）")
    parser.add_argument("--use_4bit", action="store_true", help="使用 4bit 量化")
    parser.add_argument("--ingredients", type=str, nargs="*", default=None, help="可选：手动指定食材列表")
    parser.add_argument("--output_json", type=str, default=None, help="将结果写入 JSON 文件")
    args = parser.parse_args()

    CHECKPOINT = args.checkpoint
    BASE_MODEL = args.base_model
    IMAGE_PATH = args.image
    INGREDIENTS = args.ingredients

    # 默认 processor_base 为 checkpoint 路径
    PROCESSOR_BASE = args.processor_base if args.processor_base is not None else CHECKPOINT

    print("=" * 70)
    print("MultiTaskVLM 推理（含 Ratio）")
    print("=" * 70)

    # 加载模型
    print("\n[1/2] 加载模型...")
    model, processor = load_multitask_model(
        checkpoint_path=CHECKPOINT,
        base_model_path=BASE_MODEL,
        processor_base=PROCESSOR_BASE,
        device=args.device,
        use_4bit=args.use_4bit
    )
    print("✓ 模型加载完成")

    # 推理
    print("\n[2/2] 推理...")
    try:
        result = infer_with_ratio(
            model,
            processor,
            IMAGE_PATH,
            INGREDIENTS,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        print("\nGenerated JSON:")
        print("-" * 70)
        print(json.dumps(_extract_json(result["text"]) or {"text": result["text"]}, ensure_ascii=False, indent=2))
        print("-" * 70)

        if result['ratio_probs'] is not None:
            print("\nIngredients with Ratios (JSON):")
            print("-" * 70)
            items = []
            for item, prob in zip(result['ingredients'], result['ratio_probs']):
                name = item.get("name", "") if isinstance(item, dict) else str(item)
                note = item.get("note", "") if isinstance(item, dict) else ""
                items.append({"name": name, "note": note, "ratio": float(prob)})
            print(json.dumps({"ingredients": items}, ensure_ascii=False, indent=2))
            print("-" * 70)
            if args.output_json:
                out_obj = {
                    "title": (_extract_json(result["text"]) or {}).get("title", ""),
                    "ingredients": items,
                }
                Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
                with open(args.output_json, "w", encoding="utf-8") as f:
                    json.dump(out_obj, f, ensure_ascii=False, indent=2)
        else:
            print("\n⚠️  Ratio information unavailable")

    except FileNotFoundError:
        print(f"✗ 图片不存在: {IMAGE_PATH}")
    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
