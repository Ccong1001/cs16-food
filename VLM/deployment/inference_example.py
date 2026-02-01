#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型推理示例代码 - 基础版本（仅 LM head）

用途：演示如何加载模型并进行文本生成推理

推理模式：
1. 【本文件】简化推理：只使用 LM head 生成文本
   - 适用场景：只需要菜名和食材列表
   - 优点：简单、快速、易于部署
   - 需要：merged 模型

2. 【inference_with_ratio.py】完整推理：使用 LM head + ratio head
   - 适用场景：需要食材比例信息
   - 优点：可获取 ratio 概率分布
   - 需要：checkpoint + MultiTaskVLM 代码

对于 App 部署，推荐使用本文件的简化方式。
"""

import torch
import sys
import os
import importlib
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = CURRENT_DIR.parent / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

_dataset_impl = os.environ.get("DATASET_IMPL", "dataset")
try:
    _dataset_module = importlib.import_module(_dataset_impl)
except Exception:
    _dataset_module = importlib.import_module("dataset")

USER_PROMPT_TEXT = getattr(
    _dataset_module,
    "USER_PROMPT_TEXT",
    "Given a food image, return only the recipe title and ingredient list in plain text.\n"
    "Use the format:\n"
    "Title: <title>\n"
    "Ingredients:\n"
    "- <name> | <note>\n"
    "Do not output cuisine/meal/dish labels, amounts, ratios, or JSON.",
)


def load_model(
    model_path: str, 
    device: str = "cuda:0",
    use_4bit: bool = False,
    is_merged: bool = True
):
    """
    加载模型和处理器
    
    Args:
        model_path: 模型路径
            - 如果是 merged 模型：直接加载完整模型
            - 如果是 LoRA checkpoint：需要同时指定 base_model
        device: 设备 (cuda:0, cuda:1, cpu, auto)
        use_4bit: 是否使用4bit量化以节省显存
        is_merged: 是否为已合并的完整模型
    
    Returns:
        model, processor
    """
    # 检查是否为训练checkpoint (包含adapter_config.json)
    model_path = Path(model_path)
    adapter_config = model_path / "adapter_config.json"
    has_adapter = adapter_config.exists()
    
    if has_adapter and is_merged:
        print(f"警告: 检测到 adapter_config.json，这似乎是 LoRA checkpoint 而非 merged 模型")
        print(f"如需推理，请先运行 merge_lora 脚本")
    
    # 加载 processor
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    
    # 配置量化
    quant_config = None
    if use_4bit and device != "cpu":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    
    # 加载模型
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        dtype=dtype,
        quantization_config=quant_config,
        device_map=device if device != "cpu" else {"": "cpu"},
        trust_remote_code=True
    )
    
    model.eval()
    
    # 禁用cache以避免潜在问题
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    
    return model, processor


def infer_single_image(
    model,
    processor,
    image_path: str,
    prompt: str = None,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9
) -> str:
    """
    单图推理 - 生成食物分析结果
    
    Args:
        model: 模型实例
        processor: 处理器
        image_path: 图片路径
        prompt: 提示词，默认使用训练时的标准prompt
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度 (0 = greedy, >0 = sampling)
        top_p: nucleus sampling 参数
    
    Returns:
        生成的文本结果，格式如：
        Title: <菜名>
        Ingredients:
        - <食材1> | <备注>
        - <食材2> | <备注>
        ...
    """
    # 1. 加载图片
    image = Image.open(image_path).convert("RGB")
    
    # 2. 使用训练时的标准 prompt (如果未指定)
    if prompt is None:
        prompt = USER_PROMPT_TEXT
    
    # 3. 构造多模态消息 (与训练格式一致)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # 4. 应用 chat template
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 5. 处理输入
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        padding=True
    )
    
    # 移动到模型设备
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 6. 生成配置
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "do_sample": temperature > 0,
    }
    
    if temperature > 0:
        gen_kwargs.update({
            "temperature": temperature,
            "top_p": top_p
        })
    
    # 7. 推理
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    
    # 8. 解码 (只取新生成的部分)
    generated_text = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0].strip()
    
    return generated_text


def infer_batch(
    model,
    processor,
    image_paths: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    batch_size: int = 8
) -> list[str]:
    """
    批量推理 - 提高吞吐量
    
    Args:
        model: 模型实例
        processor: 处理器
        image_paths: 图片路径列表
        max_new_tokens: 最大生成token数
        temperature: 采样温度
        batch_size: 批大小
    
    Returns:
        生成结果列表
    """
    results = []
    
    # 标准 prompt
    prompt = (
        "Given a food image, return only the recipe title and ingredient list in plain text.\n"
        "Use the format:\n"
        "Title: <title>\n"
        "Ingredients:\n"
        "- <name> | <note>\n"
        "Do not output cuisine/meal/dish labels, amounts, ratios, or JSON."
    )
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        
        # 构造 messages
        messages_batch = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            for img in images
        ]
        
        # Apply chat template
        text_prompts = [
            processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_batch
        ]
        
        # Process inputs
        inputs = processor(
            text=text_prompts,
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
            "eos_token_id": processor.tokenizer.eos_token_id,
            "pad_token_id": processor.tokenizer.pad_token_id,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        
        # Decode
        batch_results = processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        results.extend([r.strip() for r in batch_results])
    
    return results


# ============ 使用示例 ============
if __name__ == "__main__":
    import sys
    
    # 配置
    MODEL_PATH = "/scratch/li96/zl9731/cs16/Model/output/VLM/v5.4-3/checkpoint-8000-merged"
    IMAGE_PATH = "/path/to/test_food.jpg"
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        IMAGE_PATH = sys.argv[2]
    
    print(f"Model: {MODEL_PATH}")
    print(f"Image: {IMAGE_PATH}")
    
    # 加载模型（只需加载一次）
    print("\n[1/3] Loading model...")
    model, processor = load_model(
        MODEL_PATH, 
        device="cuda:0",
        use_4bit=False,  # 设为 True 可节省显存
        is_merged=True
    )
    print("✓ Model loaded!")
    
    # 单图推理
    print("\n[2/3] Running inference...")
    try:
        result = infer_single_image(
            model, 
            processor, 
            IMAGE_PATH,
            temperature=0.2
        )
        print("✓ Inference complete!")
        
        print("\n[3/3] Result:")
        print("=" * 60)
        print(result)
        print("=" * 60)
    except FileNotFoundError:
        print(f"✗ 图片文件不存在: {IMAGE_PATH}")
        print("\n使用方法:")
        print(f"  python {sys.argv[0]} [模型路径] [图片路径]")
        sys.exit(1)
    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
