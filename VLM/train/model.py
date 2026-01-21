import logging

import torch
import torch.nn as nn
from typing import Optional, Union, Dict, List

from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


logger = logging.getLogger(__name__)


class MultiTaskVLM(nn.Module):
    """Qwen3-VL with extra heads for cuisine/meal/dish, amount ordinal, ratio distribution."""

    def __init__(
        self,
        base_model,
        hidden_size: int,
        num_cuisine: int,
        num_meal: int,
        num_dish: int,
        num_amount_levels: int = 5,
    ):
        super().__init__()
        self.model = base_model
        self.hidden_size = hidden_size
        # HF Trainer expects these attributes for checkpoint warnings.
        self._keys_to_ignore_on_save = None

        self.cuisine_head = nn.Linear(hidden_size, num_cuisine)
        self.meal_head = nn.Linear(hidden_size, num_meal)
        self.dish_head = nn.Linear(hidden_size, num_dish)

        # Ordinal head: K-1 thresholds with sigmoid, target is cumulative.
        self.amount_head = nn.Linear(hidden_size, num_amount_levels - 1)

        # Ratio head: per-ingredient logits from ingredient embeddings.
        self.ratio_head = nn.Linear(hidden_size, 1)

    # 兼容 transformers Trainer 的 gradient_checkpointing_enable 调用
    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            return self.model.gradient_checkpointing_enable(**kwargs)
        return None

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            return self.model.gradient_checkpointing_disable()
        return None

    def forward(
        self,
        input_ids,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        labels=None,
        loss_weights=None,
        ingredient_token_ids=None,
        ingredient_token_mask=None,
        ingredient_mask=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Always request hidden states for pooling.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            use_cache=False,
        )

        logits = outputs.logits
        hidden = outputs.hidden_states[-1]  # (B, T, H)
        pooled = hidden[:, -1, :]  # use last token as global pooled

        cuisine_logits = self.cuisine_head(pooled)
        meal_logits = self.meal_head(pooled)
        dish_logits = self.dish_head(pooled)
        ratio_logits = None
        amount_logits = None
        if ingredient_token_ids is not None and ingredient_token_mask is not None:
            # ingredient_token_ids: (B, max_ing, max_len)
            B, M, L = ingredient_token_ids.shape
            flat_ids = ingredient_token_ids.view(B * M, L)
            flat_mask = ingredient_token_mask.view(B * M, L)
            embed = self.model.get_input_embeddings()(flat_ids)
            flat_mask = flat_mask.unsqueeze(-1)
            embed = embed * flat_mask
            denom = flat_mask.sum(dim=1).clamp(min=1e-6)
            pooled_ing = embed.sum(dim=1) / denom  # (B*M, H)
            ing_logits = self.ratio_head(pooled_ing).view(B, M)
            ratio_logits = ing_logits

            amount_logits = (
                self.amount_head(pooled_ing)
                .view(B, M, -1)
            )

        return {
            "logits": logits,
            "cuisine_logits": cuisine_logits,
            "meal_logits": meal_logits,
            "dish_logits": dish_logits,
            "amount_logits": amount_logits,
            "ratio_logits": ratio_logits,
            "hidden_states": hidden,
        }


def build_model(
    base_model: str,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    enable_text_lora: bool = True,
    enable_vision_lora: bool = True,
    use_qlora: bool = True,
    gradient_checkpointing: bool = True,
    device_map: Optional[Union[str, dict]] = None,
    num_cuisine: int = 22,
    num_meal: int = 4,
    num_dish: int = 20,
):
    """Load Qwen3-VL base model, attach LoRA, and wrap with multi-task heads."""
    quant_config = None
    if use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    base = AutoModelForImageTextToText.from_pretrained(
        base_model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quant_config,
        device_map=device_map,
        trust_remote_code=True,
    )

    def _enable_nonreentrant_gc(m: torch.nn.Module) -> bool:
        fn = getattr(m, "gradient_checkpointing_enable", None)
        if not callable(fn):
            return False
        # transformers>=4.36 supports gradient_checkpointing_kwargs
        try:
            fn(gradient_checkpointing_kwargs={"use_reentrant": False})
            return True
        except TypeError:
            try:
                fn()
            except Exception:
                return False
        except Exception:
            return False
        return False

    def _ensure_input_requires_grads(m: torch.nn.Module) -> None:
        """Ensure checkpointed blocks can backprop to trainable params.

        With (Q)LoRA + gradient checkpointing, it's common for embedding weights to be frozen,
        which makes the hidden states not require grads. Reentrant checkpointing then warns
        and may drop grads. The canonical fix is enable_input_require_grads() or a forward hook.
        """

        # Preferred: transformers provides this helper.
        enable_hook = getattr(m, "enable_input_require_grads", None)
        if callable(enable_hook):
            try:
                enable_hook()
                logger.info("Enabled input require grads via enable_input_require_grads()")
                return
            except Exception:
                pass

        getter = getattr(m, "get_input_embeddings", None)
        emb = None
        if callable(getter):
            try:
                emb = getter()
            except Exception:
                emb = None
        if emb is None:
            lm = getattr(m, "language_model", None)
            if lm is not None:
                getter = getattr(lm, "get_input_embeddings", None)
                if callable(getter):
                    try:
                        emb = getter()
                    except Exception:
                        emb = None
        if emb is None:
            logger.warning("Could not locate input embeddings to enable input grads")
            return

        def _set_require_grads(_module, _inputs, output):
            try:
                output.requires_grad_(True)
            except Exception:
                pass
            return output

        emb.register_forward_hook(_set_require_grads)
        logger.info("Enabled input require grads via embedding forward hook")

    if gradient_checkpointing:
        nonreentrant_ok = _enable_nonreentrant_gc(base)
        # Many multimodal models keep the text/vision towers under nested modules.
        for attr in ("language_model", "vision_model", "model"):
            child = getattr(base, attr, None)
            if isinstance(child, torch.nn.Module):
                nonreentrant_ok = _enable_nonreentrant_gc(child) or nonreentrant_ok
        if nonreentrant_ok:
            logger.info("Enabled gradient checkpointing (use_reentrant=False when supported)")
        else:
            logger.info("Enabled gradient checkpointing (default; non-reentrant not supported)")
        _ensure_input_requires_grads(base)

    def _available_module_suffixes(m: torch.nn.Module) -> set[str]:
        suffixes: set[str] = set()
        for name, _ in m.named_modules():
            if not name:
                continue
            suffixes.add(name.split(".")[-1])
        return suffixes

    suffixes = _available_module_suffixes(base)

    target_modules: list[str] = []
    if enable_text_lora:
        # Prefer attention projections; fall back to MLP projections if needed.
        for cand in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            if cand in suffixes:
                target_modules.append(cand)
        # Some architectures use fused projections.
        for cand in ("qkv_proj", "c_attn", "c_proj"):
            if cand in suffixes:
                target_modules.append(cand)
    if enable_vision_lora:
        # Vision naming differs across backbones; keep original patterns if present.
        for cand in ("qkv", "proj", "linear_fc1", "linear_fc2"):
            if cand in suffixes:
                # Note: these are suffixes; PEFT matches by module name, not path.
                target_modules.append(cand)

    # 防御：如果两个都关了，至少保持文本 LoRA 以避免空列表
    if not target_modules:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    base = get_peft_model(base, lora_config)

    # After PEFT wrapping, re-apply input-requires-grad hook if available.
    if gradient_checkpointing:
        # PEFT may wrap the base model; try again for nested modules.
        for attr in ("language_model", "vision_model", "model", "base_model"):
            child = getattr(base, attr, None)
            if isinstance(child, torch.nn.Module):
                _enable_nonreentrant_gc(child)
        _ensure_input_requires_grads(base)

    # Sanity check: ensure some LoRA params exist when requested.
    trainable_lora = [
        n
        for n, p in base.named_parameters()
        if p.requires_grad and ("lora_" in n.lower() or "lora" in n.lower())
    ]
    if (enable_text_lora or enable_vision_lora) and len(trainable_lora) == 0:
        raise RuntimeError(
            "LoRA was enabled but no trainable LoRA parameters were found. "
            "This usually means target_modules did not match the model's module names. "
            f"Requested enable_text_lora={enable_text_lora}, enable_vision_lora={enable_vision_lora}, "
            f"target_modules={target_modules}."
        )
    logger.info(
        "LoRA targets=%s, trainable_lora_params=%d",
        target_modules,
        len(trainable_lora),
    )

    hidden_size = getattr(base.config, "hidden_size", None)
    if hidden_size is None and hasattr(base.config, "text_config"):
        hidden_size = getattr(base.config.text_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Cannot determine hidden size from base model config")
    model = MultiTaskVLM(
        base_model=base,
        hidden_size=hidden_size,
        num_cuisine=num_cuisine,
        num_meal=num_meal,
        num_dish=num_dish,
    )

    # Disable KV cache during training to avoid autograd issues.
    model.model.config.use_cache = False
    return model


__all__ = ["build_model", "MultiTaskVLM"]
