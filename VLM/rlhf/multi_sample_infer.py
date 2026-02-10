#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-sample inference for a single image.
Generates multiple candidates from the same input and outputs JSON/JSONL.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import GenerationConfig

CURRENT_DIR = Path(__file__).resolve().parent
VLM_DIR = CURRENT_DIR.parent
if str(VLM_DIR) not in sys.path:
    sys.path.insert(0, str(VLM_DIR))

from deployment.inference_with_ratio import (  # type: ignore
    USER_PROMPT_TEXT,
    _extract_json,
    _parse_ingredients,
    load_multitask_model,
    tokenize_ingredients,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_inputs(processor, image: Image.Image, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    attn_len = int(inputs["attention_mask"].sum().item()) if "attention_mask" in inputs else inputs["input_ids"].shape[1]
    return inputs, attn_len


def _build_gen_config(
    processor,
    max_new_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    do_sample: Optional[bool],
) -> GenerationConfig:
    processor_dir = getattr(processor, "pretrained_model_name_or_path", None)
    if processor_dir is None:
        processor_dir = os.environ.get("PROCESSOR_BASE", "./")
    try:
        gen_config = GenerationConfig.from_pretrained(processor_dir)
    except Exception:
        gen_config = GenerationConfig(
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    gen_config.max_new_tokens = max_new_tokens

    if temperature is not None:
        gen_config.temperature = temperature
    if top_p is not None:
        gen_config.top_p = top_p
    if top_k is not None:
        gen_config.top_k = top_k

    if do_sample is None:
        do_sample = False
        if temperature is not None and temperature > 0:
            do_sample = True
        if top_p is not None and top_p < 1.0:
            do_sample = True
        if top_k is not None and top_k > 0:
            do_sample = True
    gen_config.do_sample = do_sample

    if gen_config.do_sample and (temperature is None or temperature <= 0):
        gen_config.temperature = 1.0

    return gen_config


def _predict_total_weight(model, inputs: Dict[str, torch.Tensor]) -> Optional[float]:
    with torch.no_grad():
        outputs = model.forward(**inputs)
        total_weight_logits = outputs.get("total_weight_logits")
    if total_weight_logits is None:
        return None
    total_weight_pred = torch.expm1(total_weight_logits).clamp(min=0).detach().cpu()
    if total_weight_pred.numel() == 0:
        return None
    return float(total_weight_pred.view(-1)[0].item())


def _predict_ratios(
    model,
    processor,
    inputs: Dict[str, torch.Tensor],
    ingredient_items: List[Dict[str, str]],
) -> List[Dict[str, Optional[float]]]:
    if not ingredient_items:
        return []

    ingredient_texts = [
        f"{i.get('name', '')} {i.get('note', '')}".strip() for i in ingredient_items
    ]
    ing_tokens = tokenize_ingredients(ingredient_texts, processor)
    inputs_ratio = dict(inputs)
    for k, v in ing_tokens.items():
        inputs_ratio[k] = v.to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model.forward(**inputs_ratio)
        ratio_logits = outputs.get("ratio_logits")

    if ratio_logits is None:
        return [
            {
                "name": item.get("name", ""),
                "note": item.get("note", ""),
                "ratio": None,
            }
            for item in ingredient_items
        ]

    ratio_logits = ratio_logits[0].to(torch.float32).cpu()
    valid_count = len(ingredient_items)
    ratio_logits = ratio_logits[:valid_count]
    ratio_probs = torch.softmax(ratio_logits, dim=0).numpy().tolist()

    items = []
    for item, prob in zip(ingredient_items, ratio_probs):
        items.append(
            {
                "name": item.get("name", ""),
                "note": item.get("note", ""),
                "ratio": float(prob),
            }
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-sample inference for one image")
    parser.add_argument("checkpoint", type=str, help="multitask heads checkpoint dir")
    parser.add_argument("image", type=str, help="image path")
    parser.add_argument("--base_model", type=str, required=True, help="base model dir")
    parser.add_argument("--processor_base", type=str, default=None, help="processor dir")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--do_sample", action="store_true", help="force sampling")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--ingredients", type=str, nargs="*", default=None, help="override ingredient names")
    parser.add_argument("--output", type=str, default=None, help="output path (.json or .jsonl)")
    parser.add_argument("--sample_id", type=str, default=None, help="base sample id")
    args = parser.parse_args()

    image_path = args.image
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    processor_base = args.processor_base if args.processor_base is not None else args.checkpoint

    model, processor = load_multitask_model(
        checkpoint_path=args.checkpoint,
        base_model_path=args.base_model,
        processor_base=processor_base,
        device=args.device,
        use_4bit=args.use_4bit,
    )

    image = Image.open(image_path).convert("RGB")
    prompt = USER_PROMPT_TEXT

    outputs: List[Dict] = []
    for i in range(args.num_samples):
        if args.seed is not None:
            _set_seed(args.seed + i)

        inputs, attn_len = _build_inputs(processor, image, prompt)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_config = _build_gen_config(
            processor=processor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample if args.do_sample else None,
        )

        with torch.no_grad():
            output_ids = model.model.generate(**inputs, generation_config=gen_config)

        seq = output_ids[0]
        gen_tokens = seq[attn_len:]
        generated_text = processor.tokenizer.decode(
            gen_tokens, skip_special_tokens=True
        ).strip()

        ingredient_items: List[Dict[str, str]] = []
        if args.ingredients:
            ingredient_items = [{"name": n, "note": ""} for n in args.ingredients]
        else:
            ingredient_items = _parse_ingredients(generated_text)

        total_weight = _predict_total_weight(model, inputs)
        ratio_items = _predict_ratios(model, processor, inputs, ingredient_items)

        parsed = _extract_json(generated_text)
        title = parsed.get("title", "") if isinstance(parsed, dict) else ""

        base_id = args.sample_id or Path(image_path).stem
        record = {
            "sample_id": f"{base_id}_{i}",
            "image": image_path,
            "title": title,
            "ingredients": ratio_items if ratio_items else ingredient_items,
            "total_weight": total_weight,
            "generated_text": generated_text,
            "meta": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "do_sample": bool(gen_config.do_sample),
                "max_new_tokens": args.max_new_tokens,
                "seed": (args.seed + i) if args.seed is not None else None,
            },
        }
        outputs.append(record)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".jsonl":
            with out_path.open("w", encoding="utf-8") as f:
                for r in outputs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(outputs, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
