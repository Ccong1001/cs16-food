import logging
import os
import sys
import warnings
from pathlib import Path
import json
import shutil
import importlib

import torch
from transformers import AutoProcessor, Trainer, TrainingArguments, TrainerCallback
from transformers.utils import logging as hf_logging

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

_dataset_impl = os.environ.get("DATASET_IMPL", "dataset")
try:
    _dataset_module = importlib.import_module(_dataset_impl)
except Exception:
    _dataset_module = importlib.import_module("dataset")

VLMJsonlDataset = _dataset_module.VLMJsonlDataset
VisionConfig = _dataset_module.VisionConfig
MultiTaskCollator = _dataset_module.MultiTaskCollator
AMOUNT_LOWER = _dataset_module.AMOUNT_LOWER
AMOUNT_UPPER = _dataset_module.AMOUNT_UPPER
CUISINE_LABELS = _dataset_module.CUISINE_LABELS
MEAL_LABELS = _dataset_module.MEAL_LABELS
DISH_LABELS = _dataset_module.DISH_LABELS
from model import build_model  # type: ignore  # noqa: E402
from arguments import parse_args  # type: ignore  # noqa: E402

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
# 压低 transformers/peft 的冗余日志，避免输出大段参数名
hf_logging.set_verbosity_error()
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)
# 过滤掉 tokenizer FutureWarning
warnings.filterwarnings(
    "ignore",
    message="`tokenizer` is deprecated and will be removed.*WeightedTrainer.__init__",
    category=FutureWarning,
)

DEFAULT_BASE_MODEL = "/mnt/hdd_1/home/cs16/Model/Qwen3-VL-8B-Instruct"
DEFAULT_DATA_PATH = "/mnt/hdd_1/home/cs16/Data/dataAB_v5/vlm_train_AB_v5.jsonl"
DEFAULT_OUTPUT_DIR = "/mnt/hdd_1/home/cs16/Model/output/VLM"
DEFAULT_DEEPSPEED = "/mnt/hdd_1/home/cs16/vri-food/VLM/train/deepspeed_zero2.json"

LABEL_HEAD_TOKENS = ("cuisine_head", "meal_head", "dish_head")
WEIGHT_HEAD_TOKENS = ("amount_head", "ratio_head", "total_weight_head")
ALL_HEAD_TOKENS = LABEL_HEAD_TOKENS + WEIGHT_HEAD_TOKENS


def _log_trainable_params(model: torch.nn.Module) -> None:
    total = 0
    trainable = 0
    lora_trainable = 0
    head_trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            lname = name.lower()
            if "lora" in lname:
                lora_trainable += n
            if lname.startswith(ALL_HEAD_TOKENS):
                head_trainable += n
    logger.info(
        "Params: total=%d, trainable=%d (lora=%d, heads=%d)",
        total,
        trainable,
        lora_trainable,
        head_trainable,
    )


def _set_head_group_trainable(
    model: torch.nn.Module, head_tokens: tuple[str, ...], trainable: bool
) -> None:
    for name, p in model.named_parameters():
        if any(tok in name for tok in head_tokens):
            p.requires_grad = bool(trainable)


def _freeze_base_model(model: torch.nn.Module) -> None:
    # Base model (including LoRA) lives under the "model." prefix in MultiTaskVLM.
    for name, p in model.named_parameters():
        if name.startswith("model."):
            p.requires_grad = False


def _log_state_dict_prefix(model: torch.nn.Module, max_keys: int = 8) -> None:
    keys = list(model.state_dict().keys())
    sample = keys[:max_keys]
    base_prefix = None
    for k in keys:
        for token in (".visual.", ".language_model."):
            idx = k.find(token)
            if idx > 0:
                base_prefix = k[:idx]
                break
        if base_prefix:
            break
    logger.info("State dict sample keys: %s", sample)
    logger.info("State dict inferred base prefix: %s", base_prefix)


def _resolve_checkpoint_file(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if not p.exists():
        raise FileNotFoundError(f"init_from_checkpoint not found: {p}")
    for name in (
        "model.safetensors",
        "pytorch_model.bin",
        "adapter_model.safetensors",
        "adapter_model.bin",
    ):
        cand = p / name
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"init_from_checkpoint directory has no model.safetensors/pytorch_model.bin: {p}"
    )


def _load_state_dict_from_file(path: Path) -> dict:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file  # type: ignore

            return load_file(str(path), device="cpu")
        except Exception as exc:
            raise RuntimeError(f"Failed to load safetensors: {path} ({exc})")
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def init_model_from_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt_file = _resolve_checkpoint_file(ckpt_path)
    state_dict = _load_state_dict_from_file(ckpt_file)
    model_state = model.state_dict()

    # Explicit prefix remap for LoRA-only adapters saved with base_model.model.model.*
    if any(k.startswith("base_model.model.model.") for k in state_dict):
        state_dict = {
            ("model.base_model.model.model." + k[len("base_model.model.model."):])
            if k.startswith("base_model.model.model.")
            else k: v
            for k, v in state_dict.items()
        }
        logger.info("Applied checkpoint prefix remap: base_model.model.model. -> model.base_model.model.model.")

    # PEFT v0.10+ stores LoRA weights under lora_A.default / lora_B.default.
    # Older checkpoints may lack the '.default' suffix. Remap if needed.
    if any(".lora_A." in k or ".lora_B." in k for k in state_dict.keys()):
        def _remap_lora_default(k: str) -> str:
            k = k.replace(".lora_A.weight", ".lora_A.default.weight")
            k = k.replace(".lora_B.weight", ".lora_B.default.weight")
            return k

        remapped = { _remap_lora_default(k): v for k, v in state_dict.items() }
        if remapped.keys() != state_dict.keys():
            state_dict = remapped
            logger.info("Applied LoRA key remap: add '.default' suffix")

    def _infer_base_prefix(keys) -> str | None:
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
                logger.info("Applied checkpoint key remap: %s -> %s", src_prefix, base_prefix)
    # If still nothing matched, try adding common wrapper prefixes (LoRA adapter keys often miss top-level wrappers).
    if len(filtered) < 50:
        model_keys = set(model_state.keys())
        best_state = state_dict
        best_filtered = filtered
        for prefix in ("model.", "model.base_model.", "model.base_model.model."):
            remapped = {prefix + k: v for k, v in state_dict.items()}
            remapped_filtered = {k: v for k, v in remapped.items() if k in model_keys}
            if len(remapped_filtered) > len(best_filtered):
                best_state = remapped
                best_filtered = remapped_filtered
                logger.info("Applied checkpoint key prefix: +%s", prefix)
        state_dict = best_state
        filtered = best_filtered
    # If still nothing matched, replace everything before the core submodule token with base_prefix.
    if len(filtered) < 50:
        base_prefix = _infer_base_prefix(model_state.keys())
        if base_prefix:
            remapped = {}
            for k, v in state_dict.items():
                for token in (".language_model.", ".visual."):
                    idx = k.find(token)
                    if idx > 0:
                        remapped[base_prefix + k[idx:]] = v
                        break
                else:
                    remapped[k] = v
            remapped_filtered = {k: v for k, v in remapped.items() if k in model_state}
            if len(remapped_filtered) > len(filtered):
                state_dict = remapped
                filtered = remapped_filtered
                logger.info("Applied checkpoint base prefix remap to: %s", base_prefix)
    if len(filtered) == 0:
        ckpt_lora = [k for k in state_dict.keys() if "lora_" in k.lower()][:8]
        model_lora = [k for k in model_state.keys() if "lora_" in k.lower()][:8]
        logger.info("Checkpoint LoRA key sample: %s", ckpt_lora)
        logger.info("Model LoRA key sample: %s", model_lora)

    missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)
    logger.info(
        "Initialized from %s (loaded=%d, missing=%d, unexpected=%d)",
        ckpt_file,
        len(filtered),
        len(missing_keys) if missing_keys else 0,
        len(unexpected_keys) if unexpected_keys else 0,
    )
    if missing_keys:
        logger.info("Missing keys sample from load_state_dict(adapter): %s", missing_keys[:20])
    if unexpected_keys:
        logger.info("Unexpected keys sample from load_state_dict(adapter): %s", unexpected_keys[:20])


def init_heads_from_file(model: torch.nn.Module, heads_path: str) -> None:
    p = Path(heads_path)
    if not p.exists():
        raise FileNotFoundError(f"init_heads not found: {p}")
    state = torch.load(str(p), map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported heads checkpoint format: {p}")
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state}
    missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)
    logger.info(
        "Initialized heads from %s (loaded=%d, missing=%d, unexpected=%d)",
        p,
        len(filtered),
        len(missing_keys) if missing_keys else 0,
        len(unexpected_keys) if unexpected_keys else 0,
    )
    if missing_keys:
        logger.info("Missing keys sample from load_state_dict(heads): %s", missing_keys[:20])
    if unexpected_keys:
        logger.info("Unexpected keys sample from load_state_dict(heads): %s", unexpected_keys[:20])


class WeightedTrainer(Trainer):
    """Trainer with multitask losses and eval component logging."""

    COMPONENT_KEYS = (
        "loss_lm",
        "loss_title",
        "loss_ing",
        "loss_cuisine",
        "loss_meal",
        "loss_dish",
        "loss_amount",
        "loss_ratio",
        "loss_hinge",
        "loss_total_weight",
    )

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        if optimizer is None:
            return optimizer
        non_empty = [g for g in optimizer.param_groups if len(g.get("params", [])) > 0]
        if len(non_empty) != len(optimizer.param_groups):
            logger.warning(
                "Dropping %d empty optimizer param groups.",
                len(optimizer.param_groups) - len(non_empty),
            )
            optimizer.param_groups = non_empty
        return optimizer

    def _reset_eval_component_metrics(self):
        self._eval_component_sums = {k: 0.0 for k in self.COMPONENT_KEYS}
        self._eval_component_counts = {k: 0 for k in self.COMPONENT_KEYS}

    def _gather_eval_component_metrics(self):
        device = getattr(self.args, "device", None) or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        sums = torch.tensor(
            [self._eval_component_sums.get(k, 0.0) for k in self.COMPONENT_KEYS],
            device=device,
        )
        counts = torch.tensor(
            [self._eval_component_counts.get(k, 0) for k in self.COMPONENT_KEYS],
            device=device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums, torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(counts, torch.distributed.ReduceOp.SUM)

        metrics = {}
        for idx, key in enumerate(self.COMPONENT_KEYS):
            denom = max(int(counts[idx].item()), 1)
            metrics[f"eval_{key}"] = sums[idx].item() / denom
        return metrics

    def _current_lambdas(self):
        lambdas = {
            "lm": self.args.loss_lambda_lm,
            "lm_title": self.args.loss_lambda_lm_title,
            "lm_ing": self.args.loss_lambda_lm_ing,
            "cuisine": self.args.loss_lambda_cuisine,
            "meal": self.args.loss_lambda_meal,
            "dish": self.args.loss_lambda_dish,
            "amount": self.args.loss_lambda_amount,
            "ratio": self.args.loss_lambda_ratio,
            "hinge": self.args.loss_lambda_hinge,
            "total_weight": getattr(self.args, "loss_lambda_total_weight", 0.0),
        }
        schedule = getattr(self.args, "loss_schedule", None) or []
        if schedule:
            step = self.state.global_step
            epoch = getattr(self.state, "epoch", None)
            for phase in schedule:
                # Epoch-based schedule takes precedence if provided.
                start_epoch = phase.get("start_epoch", None)
                end_epoch = phase.get("end_epoch", None)
                if start_epoch is not None and epoch is not None:
                    if epoch < start_epoch or (end_epoch is not None and epoch >= end_epoch):
                        continue
                else:
                    start = phase.get("start", 0)
                    end = phase.get("end", None)
                    if step < start or (end is not None and step >= end):
                        continue
                for k in lambdas.keys():
                    if f"lambda_{k}" in phase:
                        lambdas[k] = phase[f"lambda_{k}"]
        return lambdas

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        lm_weights = inputs.pop("lm_weights", None)
        cuisine_labels = inputs.pop("cuisine_labels")
        meal_labels = inputs.pop("meal_labels")
        dish_labels = inputs.pop("dish_labels")
        ingredient_ratio = inputs.pop("ingredient_ratio")
        ingredient_amount = inputs.pop("ingredient_amount")
        ingredient_mask = inputs.pop("ingredient_mask")
        total_weight = inputs.pop("total_weight", None)
        total_weight_mask = inputs.pop("total_weight_mask", None)

        outputs = model(**inputs)
        lm_logits = outputs["logits"]

        # LM loss (shifted)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        lm_ce = lm_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)
        loss_title = torch.tensor(0.0, device=lm_logits.device)
        loss_ing = torch.tensor(0.0, device=lm_logits.device)
        if lm_weights is not None:
            # shift weights
            shift_w = lm_weights[..., 1:].contiguous()
            lambdas = self._current_lambdas()
            # mark: 2.0 -> ingredient span, 1.0 -> title span, else -> default
            w = torch.ones_like(shift_w)
            w = torch.where(shift_w == 2.0, lambdas["lm_ing"], w)
            w = torch.where(shift_w == 1.0, lambdas["lm_title"], w)
            valid_mask = (shift_labels != -100)
            lm_loss = (lm_ce * w * valid_mask).sum() / valid_mask.sum().clamp(min=1)

            # 分开统计 title / ingredient 段的平均 loss（未乘 lambda）
            title_mask = (shift_w == 1.0) * valid_mask
            ing_mask = (shift_w == 2.0) * valid_mask
            if title_mask.any():
                loss_title = (lm_ce * title_mask).sum() / title_mask.sum().clamp(min=1)
            if ing_mask.any():
                loss_ing = (lm_ce * ing_mask).sum() / ing_mask.sum().clamp(min=1)
        else:
            lm_loss = lm_ce.mean()

        bce = torch.nn.BCEWithLogitsLoss()
        cuisine_loss = bce(outputs["cuisine_logits"], cuisine_labels)
        meal_loss = bce(outputs["meal_logits"], meal_labels)
        dish_loss = bce(outputs["dish_logits"], dish_labels)

        # Ordinal amount loss
        amount_logits = outputs["amount_logits"]  # (B, M, 4)
        amount_targets = ingredient_amount  # (B, M)
        valid_ing = (ingredient_mask > 0).float()
        ordinal_targets = torch.arange(
            amount_logits.size(-1), device=amount_logits.device
        ).view(1, 1, -1)
        ord_labels = (ordinal_targets < amount_targets.unsqueeze(-1)).float()
        amount_loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(
            amount_logits, ord_labels, reduction="none"
        )
        amount_loss = (
            amount_loss_raw.sum(dim=2) * valid_ing
        ).sum() / valid_ing.sum().clamp(min=1)

        ratio_logits = outputs["ratio_logits"]  # (B, M) or None
        ratio_loss = torch.tensor(0.0, device=lm_loss.device)
        hinge_loss = torch.tensor(0.0, device=lm_loss.device)

        if ratio_logits is not None:
            # Mask padding
            mask = ingredient_mask
            ratio_logits = ratio_logits.masked_fill(mask == 0, -1e4)
            ratio_prob = torch.softmax(ratio_logits, dim=1)

            # ratio CE/KL over amount_level > 1
            high_mask = (ingredient_amount > 1).float() * mask
            target = ingredient_ratio * high_mask
            denom = target.sum(dim=1, keepdim=True).clamp(min=1e-8)
            target = target / denom
            log_prob = torch.log(ratio_prob + 1e-8)
            per_sample_ratio = -(target * log_prob).sum(dim=1)
            ratio_supervised = high_mask.sum(dim=1) > 0
            if ratio_supervised.any():
                ratio_loss = per_sample_ratio[ratio_supervised].mean()

            # Hinge per ingredient
            lower = torch.zeros_like(ratio_prob)
            upper = torch.zeros_like(ratio_prob)
            for level, lo in AMOUNT_LOWER.items():
                lower = lower + (ingredient_amount == level).float() * lo
            for level, up in AMOUNT_UPPER.items():
                upper = upper + (ingredient_amount == level).float() * up

            high_mask_hinge = (ingredient_amount > 1).float() * mask
            per_ing_hinge = (
                torch.relu(lower - ratio_prob) + torch.relu(ratio_prob - upper)
            ) * high_mask_hinge

            # For level 0/1: only upper cap and total cap
            low_mask = ((ingredient_amount == 0) | (ingredient_amount == 1)).float()
            upper_low = torch.zeros_like(ratio_prob)
            for level, up in AMOUNT_UPPER.items():
                upper_low = upper_low + (ingredient_amount == level).float() * up
            cap_low = torch.relu(ratio_prob - upper_low) * low_mask * mask

            # Additional constraints for level 0/1 totals
            low_mask = ((ingredient_amount == 0) | (ingredient_amount == 1)).float()
            low_sum = (ratio_prob * low_mask).sum(dim=1)
            hinge_low_sum = torch.relu(low_sum - 0.15)

            valid_supervised = mask.sum(dim=1) > 0
            if valid_supervised.any():
                per_sample_hinge = (
                    (per_ing_hinge.sum(dim=1) + cap_low.sum(dim=1))
                    / mask.sum(dim=1).clamp(min=1)
                ) + hinge_low_sum
                hinge_loss = per_sample_hinge[valid_supervised].mean()

        total_weight_loss = torch.tensor(0.0, device=lm_loss.device)
        if total_weight is not None and total_weight_mask is not None:
            tw_logits = outputs.get("total_weight_logits")
            if tw_logits is not None:
                tw_target = total_weight.to(lm_loss.device).clamp(min=0)
                tw_mask = total_weight_mask.to(lm_loss.device)
                tw_target_log = torch.log1p(tw_target)
                mse = torch.nn.MSELoss(reduction="none")
                tw_loss_raw = mse(tw_logits, tw_target_log)
                total_weight_loss = (tw_loss_raw * tw_mask).sum() / tw_mask.sum().clamp(min=1)

        lambdas = self._current_lambdas()
        total_loss = (
            lambdas["lm"] * lm_loss
            + lambdas["cuisine"] * cuisine_loss
            + lambdas["meal"] * meal_loss
            + lambdas["dish"] * dish_loss
            + lambdas["amount"] * amount_loss
            + lambdas["ratio"] * ratio_loss
            + lambdas["hinge"] * hinge_loss
            + lambdas["total_weight"] * total_weight_loss
        )
        # 保存各项 loss 便于日志输出
        try:
            self._last_component_logs = {
                "loss_lm": lm_loss.detach().float().item(),
                "loss_title": loss_title.detach().float().item(),
                "loss_ing": loss_ing.detach().float().item(),
                "loss_cuisine": cuisine_loss.detach().float().item(),
                "loss_meal": meal_loss.detach().float().item(),
                "loss_dish": dish_loss.detach().float().item(),
                "loss_amount": amount_loss.detach().float().item(),
                "loss_ratio": ratio_loss.detach().float().item(),
                "loss_hinge": hinge_loss.detach().float().item(),
                "loss_total_weight": total_weight_loss.detach().float().item(),
            }
        except Exception:
            self._last_component_logs = {}
        if not model.training and getattr(self, "_track_eval_components", False):
            for k in self.COMPONENT_KEYS:
                if k in self._last_component_logs:
                    self._eval_component_sums[k] += float(self._last_component_logs[k])
                    self._eval_component_counts[k] += 1
        return (total_loss, outputs) if return_outputs else total_loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        self._track_eval_components = True
        self._reset_eval_component_metrics()
        output = self.evaluation_loop(
            self.get_eval_dataloader(eval_dataset),
            description=f"{metric_key_prefix.capitalize()}ation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self._track_eval_components = False

        metrics = output.metrics
        component_metrics = self._gather_eval_component_metrics()
        metrics.update(component_metrics)
        self._last_eval_component_metrics = component_metrics

        if f"{metric_key_prefix}_samples_per_second" not in metrics and "eval_runtime" in metrics:
            total_batch_size = self.args.eval_batch_size * max(1, self.args.world_size)
            metrics[f"{metric_key_prefix}_samples_per_second"] = (
                total_batch_size / metrics["eval_runtime"]
            )

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    # HF 对 PeftModel 的默认加载会要求 state_dict 只包含 LoRA 权重，包含底座/量化权重时会报大量 unexpected keys
    #（就是之前刷屏的 absmax/quant_map 列表），导致 resume 直接失败。这里放宽为只加载当前模型里存在的键。
    def _load_state_dict_in_model(self, state_dict):
        model_state = self.model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_state}
        missing_keys, unexpected_keys = self.model.load_state_dict(filtered, strict=False)
        if missing_keys:
            logger.warning("Ignoring %d missing keys when loading checkpoint (e.g. %s)", len(missing_keys), missing_keys[:3])
        if unexpected_keys:
            logger.warning("Ignoring %d unexpected keys when loading checkpoint (e.g. %s)", len(unexpected_keys), unexpected_keys[:3])
        return missing_keys, unexpected_keys

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        if not getattr(self.args, "save_lora_only", False):
            return super().save_model(output_dir=output_dir, _internal_call=_internal_call)

        save_dir = Path(output_dir or self.args.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        model = self.model
        if hasattr(model, "module"):
            model = model.module

        peft_model = getattr(model, "model", None)
        if peft_model is not None and hasattr(peft_model, "save_pretrained"):
            try:
                peft_model.save_pretrained(save_dir)
            except Exception as exc:
                logger.warning("Failed to save LoRA adapters: %s", exc)

        # Save multitask heads separately
        head_prefixes = ALL_HEAD_TOKENS
        try:
            heads_state = {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
                if k.startswith(head_prefixes)
            }
            if heads_state:
                torch.save(heads_state, save_dir / "multitask_heads.bin")
        except Exception as exc:
            logger.warning("Failed to save multitask heads: %s", exc)


class JsonlLoggerCallback(TrainerCallback):
    """Append training/eval logs to output_dir/logging.jsonl (rank 0 only)."""

    def __init__(self, output_dir: Path, trainer_ref: WeightedTrainer | None = None):
        self.log_path = output_dir / "logging.jsonl"
        self.trainer_ref = trainer_ref

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        trainer = self.trainer_ref
        if trainer is not None and hasattr(trainer, "is_world_process_zero") and not trainer.is_world_process_zero():
            return
        # 合并最近一次各项 loss：训练步用 _last_component_logs，评估步用 eval 聚合指标
        if trainer is not None and hasattr(trainer, "_last_eval_component_metrics") and "eval_loss" in logs:
            logs = {**logs, **getattr(trainer, "_last_eval_component_metrics", {})}
        elif trainer is not None and hasattr(trainer, "_last_component_logs"):
            logs = {**logs, **getattr(trainer, "_last_component_logs", {})}
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(logs) + "\n")
        except Exception:
            pass


class ProcessorSaveCallback(TrainerCallback):
    """Save processor/tokenizer assets into output_dir and checkpoints (rank 0 only)."""

    def __init__(self, processor, base_model_path: str):
        self.processor = processor
        self.base_model_path = Path(base_model_path)
        self._extra_files = (
            "config.json",
            "generation_config.json",
            "video_preprocessor_config.json",
        )

    def _copy_extras(self, dst: Path):
        for name in self._extra_files:
            src = self.base_model_path / name
            if src.exists():
                target = dst / name
                if not target.exists():
                    try:
                        shutil.copy2(src, target)
                    except Exception:
                        pass

    def _save_to(self, dst: Path):
        dst.mkdir(parents=True, exist_ok=True)
        try:
            self.processor.save_pretrained(dst)
        except Exception:
            pass
        self._copy_extras(dst)

    def on_train_begin(self, args, state, control, **kwargs):
        if hasattr(args, "local_rank") and args.local_rank not in (-1, 0):
            return
        self._save_to(Path(args.output_dir))

    def on_save(self, args, state, control, **kwargs):
        if hasattr(args, "local_rank") and args.local_rank not in (-1, 0):
            return
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        self._save_to(checkpoint_dir)


def dump_eval_predictions(
    trainer: WeightedTrainer,
    processor,
    eval_dataset,
    output_path: Path,
    max_new_tokens: int = 256,
    batch_size: int = 1,
):
    """Run generation on eval dataset and dump predictions to JSONL."""
    if eval_dataset is None or len(eval_dataset) == 0:
        logger.warning("Skip dumping eval predictions: no eval dataset.")
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if not trainer.is_world_process_zero():
            return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Dumping eval predictions to %s", output_path)

    model = trainer.model
    if hasattr(model, "module"):
        model = model.module
    base_model = getattr(model, "model", model)
    device = getattr(trainer.args, "device", None) or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    old_cache = getattr(base_model.config, "use_cache", None)
    try:
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = True
        base_model.eval()
        collator = getattr(trainer, "data_collator", None)
        if collator is None or not isinstance(collator, MultiTaskCollator):
            collator = MultiTaskCollator(
                processor, lambda_lm_ing=getattr(trainer.args, "loss_lambda_lm_ing", 0.0)
            )
        with output_path.open("w", encoding="utf-8") as f:
            for start in range(0, len(eval_dataset), batch_size):
                batch = [eval_dataset[i] for i in range(start, min(start + batch_size, len(eval_dataset)))]
                prompts = []
                images = []
                targets = []
                for ex in batch:
                    user_msg = ex["messages"][0]
                    prompt = processor.apply_chat_template(
                        [user_msg],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    prompts.append(prompt)
                    images.append(ex["image"])
                    targets.append(ex.get("labels", ""))

                inputs = processor(
                    text=prompts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
                attn_lens = (
                    inputs["attention_mask"].sum(dim=1).tolist()
                    if "attention_mask" in inputs
                    else [inputs["input_ids"].shape[1]] * len(batch)
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    gen = base_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                    # 分类/回归头预测
                    collated = collator(batch)
                    model_inputs = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in collated.items()
                        if k
                        not in (
                            "ingredient_texts",
                        )
                    }
                    outputs = model(**model_inputs)
                    cuisine_pred = torch.sigmoid(outputs["cuisine_logits"]).cpu()
                    meal_pred = torch.sigmoid(outputs["meal_logits"]).cpu()
                    dish_pred = torch.sigmoid(outputs["dish_logits"]).cpu()
                    amount_logits = outputs.get("amount_logits")
                    ratio_logits = outputs.get("ratio_logits")
                    total_weight_logits = outputs.get("total_weight_logits")
                    amount_pred = None
                    ratio_pred = None
                    total_weight_pred = None
                    if amount_logits is not None:
                        amount_probs = torch.sigmoid(amount_logits).cpu()
                        amount_pred = (amount_probs > 0.5).sum(dim=-1)
                    if ratio_logits is not None:
                        mask = collated["ingredient_mask"].to(device)
                        ratio_logits_m = ratio_logits.masked_fill(mask == 0, -1e4)
                        ratio_pred = torch.softmax(ratio_logits_m, dim=1).cpu()
                    if total_weight_logits is not None:
                        total_weight_pred = torch.expm1(total_weight_logits).clamp(min=0).cpu()
                for j, seq in enumerate(gen):
                    prompt_len = int(attn_lens[j])
                    gen_tokens = seq[prompt_len:]
                    text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                    # decode targets/preds for logging
                    label_threshold = getattr(trainer.args, "label_threshold", 0.5)

                    def _decode_multi(vec, names):
                        return [
                            names[idx]
                            for idx, val in enumerate(vec.tolist())
                            if val > label_threshold
                        ]

                    record = {
                        "index": start + j,
                        "prediction": text,
                        "target": targets[j],
                        "cuisine_target": _decode_multi(
                            collated["cuisine_labels"][j], CUISINE_LABELS
                        ),
                        "meal_target": _decode_multi(
                            collated["meal_labels"][j], MEAL_LABELS
                        ),
                        "dish_target": _decode_multi(
                            collated["dish_labels"][j], DISH_LABELS
                        ),
                        "cuisine_pred": _decode_multi(cuisine_pred[j], CUISINE_LABELS),
                        "meal_pred": _decode_multi(meal_pred[j], MEAL_LABELS),
                        "dish_pred": _decode_multi(dish_pred[j], DISH_LABELS),
                    }
                    ing_mask = collated["ingredient_mask"][j].cpu()
                    if amount_pred is not None:
                        record["amount_target"] = [
                            int(x)
                            for x, m in zip(
                                collated["ingredient_amount"][j].tolist(),
                                ing_mask.tolist(),
                            )
                            if m > 0
                        ]
                        record["amount_pred"] = [
                            int(x)
                            for x, m in zip(amount_pred[j].tolist(), ing_mask.tolist())
                            if m > 0
                        ]
                    if ratio_pred is not None:
                        record["ratio_target"] = [
                            float(x)
                            for x, m in zip(
                                collated["ingredient_ratio"][j].tolist(),
                                ing_mask.tolist(),
                            )
                            if m > 0
                        ]
                        record["ratio_pred"] = [
                            float(x)
                            for x, m in zip(
                                ratio_pred[j].tolist(), ing_mask.tolist()
                            )
                            if m > 0
                        ]
                    if total_weight_pred is not None:
                        record["total_weight_pred"] = float(total_weight_pred[j].item())
                    if "total_weight" in collated and "total_weight_mask" in collated:
                        if float(collated["total_weight_mask"][j].item()) > 0:
                            record["total_weight_target"] = float(collated["total_weight"][j].item())
                    record = {
                        **record,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if old_cache is not None and hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = old_cache


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 保存运行参数
    try:
        with (output_dir / "args.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Failed to save args.json: %s", exc)

    logger.info("Base model: %s", args.base_model)
    logger.info("Dataset: %s", args.dataset)
    logger.info("Output dir: %s", output_dir)

    processor_base = args.processor_base or args.base_model
    processor = AutoProcessor.from_pretrained(processor_base, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = "right"

    vision_cfg = VisionConfig()
    dataset = VLMJsonlDataset(
        data_path=Path(args.dataset),
        processor=processor,
        vision_cfg=vision_cfg,
        max_samples=args.max_samples,
    )
    collator = MultiTaskCollator(processor, lambda_lm_ing=args.lambda_lm_ing)
    eval_dataset = None
    if args.val_dataset:
        eval_dataset = VLMJsonlDataset(
            data_path=Path(args.val_dataset),
            processor=processor,
            vision_cfg=vision_cfg,
            max_samples=None,
        )

    device_map = None
    if not args.no_qlora and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_map = {"": local_rank}

    model = build_model(
        base_model=args.base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        enable_text_lora=args.enable_text_lora,
        enable_vision_lora=args.enable_vision_lora,
        use_qlora=not args.no_qlora,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        device_map=device_map,
        num_cuisine=len(CUISINE_LABELS),
        num_meal=len(MEAL_LABELS),
        num_dish=len(DISH_LABELS),
    )

    _log_state_dict_prefix(model)

    if getattr(args, "init_from_checkpoint", None):
        init_model_from_checkpoint(model, args.init_from_checkpoint)
        try:
            ckpt_path = Path(args.init_from_checkpoint)
            heads_path = (
                ckpt_path / "multitask_heads.bin"
                if ckpt_path.is_dir()
                else ckpt_path.parent / "multitask_heads.bin"
            )
            if heads_path.exists():
                init_heads_from_file(model, str(heads_path))
        except Exception as exc:
            logger.warning("Failed to load multitask heads: %s", exc)

    if not getattr(args, "train_lm", True):
        _freeze_base_model(model)
    if not getattr(args, "train_labels", True):
        _set_head_group_trainable(model, LABEL_HEAD_TOKENS, False)
    if not getattr(args, "train_weight", True):
        _set_head_group_trainable(model, WEIGHT_HEAD_TOKENS, False)

    _log_trainable_params(model)

    eval_strategy = "steps" if eval_dataset is not None else "no"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=torch.cuda.is_available(),
        dataloader_pin_memory=True,
        dataloader_num_workers=args.dataloader_num_workers,
        eval_strategy=eval_strategy,
        eval_steps=getattr(args, "eval_steps", None),
        deepspeed=None if not args.deepspeed else args.deepspeed,
        max_steps=args.max_steps,
        seed=args.seed,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        disable_tqdm=False,
    )
    # attach loss weights
    training_args.loss_lambda_lm = args.lambda_lm
    training_args.loss_lambda_lm_title = args.lambda_lm_title
    training_args.loss_lambda_lm_ing = args.lambda_lm_ing
    training_args.loss_lambda_cuisine = args.lambda_cuisine
    training_args.loss_lambda_meal = args.lambda_meal
    training_args.loss_lambda_dish = args.lambda_dish
    training_args.loss_lambda_amount = args.lambda_amount
    training_args.loss_lambda_ratio = args.lambda_ratio
    training_args.loss_lambda_hinge = args.lambda_hinge
    training_args.loss_lambda_total_weight = args.lambda_total_weight
    training_args.save_lora_only = args.save_lora_only
    training_args.train_lm = args.train_lm
    training_args.train_labels = args.train_labels
    training_args.train_weight = args.train_weight
    # optional phased schedule
    try:
        training_args.loss_schedule = json.loads(args.loss_schedule) if args.loss_schedule else []
    except Exception as exc:
        logger.warning("Failed to parse loss_schedule JSON: %s", exc)
        training_args.loss_schedule = []

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )
    # 追加 jsonl logger（仅主进程）
    trainer.add_callback(JsonlLoggerCallback(output_dir=output_dir, trainer_ref=trainer))
    trainer.add_callback(ProcessorSaveCallback(processor=processor, base_model_path=processor_base))

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if not getattr(args, "skip_save", False):
        trainer.save_state()
        trainer.save_model()
        logger.info("Training complete. Final checkpoint saved to %s", output_dir)
    else:
        logger.info("Skip saving model/state (--skip_save enabled).")

    if eval_dataset is not None and getattr(args, "eval_dump_predictions", None):
        dump_eval_predictions(
            trainer=trainer,
            processor=processor,
            eval_dataset=eval_dataset,
            output_path=Path(args.eval_dump_predictions),
            max_new_tokens=args.eval_max_new_tokens,
            batch_size=args.eval_generate_batch_size,
        )


if __name__ == "__main__":
    main()
