#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import sys

import torch

CURRENT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = CURRENT_DIR.parent / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.append(str(TRAIN_DIR))

from dataset import CUISINE_LABELS, MEAL_LABELS, DISH_LABELS  # type: ignore  # noqa: E402
from model import build_model  # type: ignore  # noqa: E402


def _resolve_checkpoint_file(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    for name in ("model.safetensors", "pytorch_model.bin"):
        cand = path / name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"no model.safetensors/pytorch_model.bin under: {path}")


def _load_state_dict(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file  # type: ignore

        return load_file(str(path), device="cpu")
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"unsupported checkpoint format: {path}")


def _copy_extras(src: Path, dst: Path) -> None:
    extras = (
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "video_preprocessor_config.json",
    )
    for name in extras:
        cand = src / name
        if cand.exists():
            target = dst / name
            if not target.exists():
                try:
                    shutil.copy2(cand, target)
                except Exception:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base checkpoint.")
    parser.add_argument(
        "--base_model",
        default="/mnt/hdd_1/home/cs16/Model/output/VLM/v3-20251222-211237/checkpoint-7633-merged",
    )
    parser.add_argument(
        "--checkpoint",
        default="/mnt/hdd_1/home/cs16/Model/output/VLM/v5.2-2",
        help="Path to checkpoint dir or model file (model.safetensors/pytorch_model.bin).",
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/hdd_1/home/cs16/Model/output/VLM/v5.2-merged",
    )
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--enable_text_lora", dest="enable_text_lora", action="store_true")
    parser.add_argument("--disable_text_lora", dest="enable_text_lora", action="store_false")
    parser.add_argument("--enable_vision_lora", dest="enable_vision_lora", action="store_true")
    parser.add_argument("--disable_vision_lora", dest="enable_vision_lora", action="store_false")
    parser.add_argument("--use_qlora", dest="use_qlora", action="store_true")
    parser.add_argument("--no_qlora", dest="use_qlora", action="store_false")
    parser.add_argument(
        "--save_safetensors",
        action="store_true",
        help="Save model.safetensors (moves tensors to CPU).",
    )
    parser.set_defaults(enable_text_lora=True, enable_vision_lora=False, use_qlora=True)
    args = parser.parse_args()

    model = build_model(
        base_model=args.base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        enable_text_lora=args.enable_text_lora,
        enable_vision_lora=args.enable_vision_lora,
        use_qlora=args.use_qlora,
        gradient_checkpointing=False,
        device_map=None,
        num_cuisine=len(CUISINE_LABELS),
        num_meal=len(MEAL_LABELS),
        num_dish=len(DISH_LABELS),
    )

    ckpt_path = _resolve_checkpoint_file(Path(args.checkpoint))
    state_dict = _load_state_dict(ckpt_path)
    model_state = model.state_dict()
    
    def _infer_base_prefix(keys):
        # Use a known submodule token to locate the base prefix.
        for k in keys:
            for token in (".visual.", ".language_model."):
                idx = k.find(token)
                if idx > 0:
                    return k[:idx]
        return None
    
    filtered = {k: v for k, v in state_dict.items() if k in model_state}
    # If almost nothing matched, try remapping merged checkpoints (model.model.* -> model.base_model.model.*).
    if len(filtered) < 50:
        base_prefix = _infer_base_prefix(model_state.keys())
        if base_prefix and any(k.startswith("model.model.") for k in state_dict):
            remapped = {}
            src_prefix = "model.model"
            for k, v in state_dict.items():
                if k.startswith(src_prefix):
                    remapped[base_prefix + k[len(src_prefix):]] = v
                else:
                    remapped[k] = v
            remapped_filtered = {k: v for k, v in remapped.items() if k in model_state}
            if len(remapped_filtered) > len(filtered):
                state_dict = remapped
                filtered = remapped_filtered
                print(f"Applied checkpoint key remap: {src_prefix} -> {base_prefix}")
    
    missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)
    print(
        f"Loaded checkpoint: {ckpt_path} (loaded={len(filtered)}, "
        f"missing={len(missing_keys) if missing_keys else 0}, "
        f"unexpected={len(unexpected_keys) if unexpected_keys else 0})"
    )

    base = model.model
    merge_fn = getattr(base, "merge_and_unload", None)
    if not callable(merge_fn):
        raise RuntimeError("Base model does not support merge_and_unload (is this a PEFT model?)")
    merged = merge_fn()
    model.model = merged

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_safetensors:
        from safetensors.torch import save_file  # type: ignore

        cpu_state = {k: v.detach().cpu() for k, v in model.model.state_dict().items()}
        save_file(cpu_state, str(output_dir / "model.safetensors"))
    else:
        torch.save(model.model.state_dict(), output_dir / "pytorch_model.bin")

    _copy_extras(Path(args.checkpoint), output_dir)
    _copy_extras(Path(args.base_model), output_dir)
    with (output_dir / "merge_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Merged checkpoint saved to: {output_dir}")


if __name__ == "__main__":
    main()
