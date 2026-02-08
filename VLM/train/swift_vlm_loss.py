"""
Custom VLM multitask loss for Swift.

Usage:
- Ensure this file is on PYTHONPATH (e.g., export PYTHONPATH=/mnt/hdd_1/home/cs16/vri-food/VLM/train:$PYTHONPATH).
- Import the module *before* Swift resolves loss_type so it can patch swift.plugin.loss.loss_mapping:
    import swift_vlm_loss  # registers 'vlm_multitask'
- Then run Swift with `--loss_type vlm_multitask`; labels must be a dict matching the keys below.
"""

from typing import Any, Dict

import torch
import torch.nn as nn


def vlm_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    num_items_in_batch=None,
    trainer=None,
    attention_mask=None,
    **kwargs,
) -> torch.Tensor:
    """
    Multi-task loss: LM + cuisine/meal/dish + amount (ordinal) + ratio (regression) + total_weight (regression).

    Expected outputs keys:
      - logits: (B, T, V)
      - cuisine_logits: (B, C)
      - meal_logits: (B, M)
      - dish_logits: (B, D)
      - amount_logits: (B, max_ing, K-1) or None
      - ratio_logits: (B, max_ing) or None
      - total_weight_logits: (B,) or None

    Expected labels keys:
      - lm: (B, T) token ids with -100 mask
      - lm_weights: (B, T) optional weights; 2->ingredient span, 1->title span, else default
      - cuisine: (B, C) multi-hot
      - meal: (B, M) multi-hot
      - dish: (B, D) multi-hot
      - ingredient_amount: (B, max_ing) ordinal targets 0..K-1
      - ingredient_ratio: (B, max_ing) float ratios
      - ingredient_mask: (B, max_ing) 1/0 validity mask
      - total_weight: (B,) float targets
      - total_weight_mask: (B,) 1/0 validity mask
    """
    if not isinstance(outputs, dict) or not isinstance(labels, dict):
        raise ValueError("vlm_multitask_loss expects dict outputs and dict labels")

    device = outputs["logits"].device

    def _get_lambda(name: str, default: float = 1.0) -> float:
        if trainer is not None and hasattr(trainer, "args"):
            return getattr(trainer.args, f"loss_lambda_{name}", default)
        return default

    lambda_lm = _get_lambda("lm", 1.0)
    lambda_lm_title = _get_lambda("lm_title", 1.0)
    lambda_lm_ing = _get_lambda("lm_ing", 1.0)
    lambda_cuisine = _get_lambda("cuisine", 1.0)
    lambda_meal = _get_lambda("meal", 1.0)
    lambda_dish = _get_lambda("dish", 1.0)
    lambda_amount = _get_lambda("amount", 1.0)
    lambda_ratio = _get_lambda("ratio", 1.0)
    lambda_total_weight = _get_lambda("total_weight", 1.0)

    # LM loss with optional span weights
    logits = outputs["logits"]
    lm_labels = labels["lm"].to(device)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = lm_labels[..., 1:].contiguous()
    lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    lm_ce = lm_loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.shape)
    lm_weights = labels.get("lm_weights")
    if lm_weights is not None:
        lm_weights = lm_weights.to(device)[..., 1:].contiguous()
        w = torch.ones_like(lm_weights)
        w = torch.where(lm_weights == 2.0, lambda_lm_ing, w)
        w = torch.where(lm_weights == 1.0, lambda_lm_title, w)
        lm_loss = (lm_ce * w * (shift_labels != -100)).sum() / (
            (shift_labels != -100).sum().clamp(min=1)
        )
    else:
        lm_loss = lm_ce.mean()
    lm_loss = lm_loss * lambda_lm

    bce = nn.BCEWithLogitsLoss()
    cuisine_loss = bce(outputs["cuisine_logits"], labels["cuisine"].to(device)) * lambda_cuisine
    meal_loss = bce(outputs["meal_logits"], labels["meal"].to(device)) * lambda_meal
    dish_loss = bce(outputs["dish_logits"], labels["dish"].to(device)) * lambda_dish

    total_loss = lm_loss + cuisine_loss + meal_loss + dish_loss

    # Amount ordinal loss (cumulative targets)
    amount_logits = outputs.get("amount_logits")
    if amount_logits is not None and lambda_amount != 0:
        amount_logits = amount_logits.to(device)
        amount_targets = labels["ingredient_amount"].to(device)
        amount_mask = labels["ingredient_mask"].to(device)
        k_minus_1 = amount_logits.shape[-1]
        ordinal_targets = torch.arange(k_minus_1, device=device).unsqueeze(0).unsqueeze(0)
        ordinal_labels = (amount_targets.unsqueeze(-1) > ordinal_targets).float()
        bce_ord = nn.BCEWithLogitsLoss(reduction="none")
        ord_loss = bce_ord(amount_logits, ordinal_labels)
        ord_loss = (ord_loss * amount_mask.unsqueeze(-1)).sum() / amount_mask.sum().clamp(min=1)
        total_loss = total_loss + lambda_amount * ord_loss

    # Ratio regression loss
    ratio_logits = outputs.get("ratio_logits")
    if ratio_logits is not None and lambda_ratio != 0:
        ratio_targets = labels["ingredient_ratio"].to(device)
        ratio_mask = labels["ingredient_mask"].to(device)
        mse = nn.MSELoss(reduction="none")
        ratio_loss = mse(torch.sigmoid(ratio_logits), ratio_targets)
        ratio_loss = (ratio_loss * ratio_mask).sum() / ratio_mask.sum().clamp(min=1)
        total_loss = total_loss + lambda_ratio * ratio_loss

    # Total weight regression loss (log1p space)
    total_weight_logits = outputs.get("total_weight_logits")
    if total_weight_logits is not None and lambda_total_weight != 0:
        total_weight = labels.get("total_weight")
        total_weight_mask = labels.get("total_weight_mask")
        if total_weight is not None and total_weight_mask is not None:
            tw_target = total_weight.to(device).clamp(min=0)
            tw_mask = total_weight_mask.to(device)
            tw_target_log = torch.log1p(tw_target)
            mse = nn.MSELoss(reduction="none")
            tw_loss = mse(total_weight_logits, tw_target_log)
            tw_loss = (tw_loss * tw_mask).sum() / tw_mask.sum().clamp(min=1)
            total_loss = total_loss + lambda_total_weight * tw_loss

    return total_loss


def register_swift_loss() -> bool:
    """
    Register the loss into Swift's loss_mapping at runtime.

    Returns:
        bool: True if registration succeeded, False otherwise.
    """
    try:
        from swift.plugin import loss as swift_loss  # type: ignore
    except Exception:
        return False
    swift_loss.loss_mapping["vlm_multitask"] = vlm_multitask_loss
    return True


# Try to register automatically on import; silently ignore if Swift not installed.
try:
    register_swift_loss()
except Exception:
    pass
