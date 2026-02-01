#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 LoRA checkpoint 到 base model

用途：
1. 将训练的 LoRA adapter 合并到基座模型
2. 生成可直接用于推理的完整模型
3. 只保留语言生成功能，移除训练时的分类头

使用方法:
    python merge_lora.py \\
        --base_model /path/to/base_model \\
        --checkpoint /path/to/checkpoint-XXXX \\
        --output /path/to/output/merged_model

注意：
- 此脚本只合并 LoRA 权重，不包含 MultiTaskVLM 的分类头
- 输出的模型可直接用于推理，无需自定义代码
"""

import argparse
import shutil
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
import json


def parse_args():
    parser = argparse.ArgumentParser(description="合并 LoRA checkpoint 到基座模型")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="基座模型路径"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="LoRA checkpoint 路径 (包含 adapter_model.safetensors)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出路径 (merged 模型保存位置)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="设备 (cuda:0, cpu)"
    )
    return parser.parse_args()


def merge_lora(base_model_path: str, checkpoint_path: str, output_path: str, device: str = "cuda:0"):
    """
    合并 LoRA 权重到基座模型
    
    Args:
        base_model_path: 基座模型路径
        checkpoint_path: LoRA checkpoint 路径
        output_path: 输出路径
        device: 设备
    """
    print("=" * 70)
    print("LoRA 模型合并工具")
    print("=" * 70)
    
    # 1. 检查路径
    base_model_path = Path(base_model_path)
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    
    print(f"\n基座模型: {base_model_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"输出路径: {output_path}")
    
    if not base_model_path.exists():
        raise FileNotFoundError(f"基座模型不存在: {base_model_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")
    
    # 检查是否为 LoRA checkpoint
    adapter_config = checkpoint_path / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(
            f"未找到 adapter_config.json，这可能不是 LoRA checkpoint: {checkpoint_path}"
        )
    
    # 2. 加载基座模型
    print("\n[1/4] 加载基座模型...")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    
    base_model = AutoModelForImageTextToText.from_pretrained(
        str(base_model_path),
        dtype=dtype,
        device_map=device if device != "cpu" else {"": "cpu"},
        trust_remote_code=True
    )
    print("✓ 基座模型加载完成")
    
    # 3. 加载 LoRA adapter
    print("\n[2/4] 加载 LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
    print("✓ LoRA adapter 加载完成")
    
    # 4. 合并权重
    print("\n[3/4] 合并 LoRA 权重到基座模型...")
    merged_model = model.merge_and_unload()
    print("✓ 权重合并完成")
    
    # 5. 保存模型
    print("\n[4/4] 保存 merged 模型...")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存模型权重
    merged_model.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size="5GB"
    )
    
    # 复制 processor 配置
    print("  - 保存 processor 配置...")
    processor = AutoProcessor.from_pretrained(str(base_model_path), trust_remote_code=True)
    processor.save_pretrained(str(output_path))
    
    # 复制必要的配置文件
    for config_file in ["generation_config.json", "preprocessor_config.json"]:
        src = base_model_path / config_file
        dst = output_path / config_file
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"  - 复制 {config_file}")
    
    print(f"✓ 模型保存至: {output_path}")
    
    # 6. 验证
    print("\n[验证] 检查输出文件...")
    required_files = [
        "config.json",
        "tokenizer_config.json",
    ]
    
    missing_files = [f for f in required_files if not (output_path / f).exists()]
    if missing_files:
        print(f"⚠️  警告: 缺少文件: {missing_files}")
    else:
        print("✓ 所有必需文件已生成")
    
    # 统计模型大小
    total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    print(f"\n模型总大小: {total_size / (1024**3):.2f} GB")
    
    print("\n" + "=" * 70)
    print("合并完成!")
    print("=" * 70)
    print(f"\n✓ 可以使用以下命令测试:")
    print(f"  python inference_example.py {output_path} /path/to/test_image.jpg")
    

if __name__ == "__main__":
    args = parse_args()
    
    try:
        merge_lora(
            base_model_path=args.base_model,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            device=args.device
        )
    except Exception as e:
        print(f"\n❌ 合并失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
