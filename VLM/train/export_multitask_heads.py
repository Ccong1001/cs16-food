#!/usr/bin/env python3
import torch
from safetensors.torch import load_file

HEAD_SRC = "/scratch/li96/zl9731/cs16/Model/output/VLM/v5.4-3/model.safetensors"
HEAD_DST = "/scratch/li96/zl9731/cs16/Model/output/VLM/v5.4-3-lora/multitask_heads.bin"

state = load_file(HEAD_SRC)
head_prefixes = ("cuisine_head", "meal_head", "dish_head", "amount_head", "ratio_head")
heads = {k: v for k, v in state.items() if k.startswith(head_prefixes)}
if not heads:
    raise SystemExit("No multitask heads found in source checkpoint")
torch.save(heads, HEAD_DST)
print(f"Saved multitask heads to: {HEAD_DST}")
