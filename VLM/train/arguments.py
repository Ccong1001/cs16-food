import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 默认路径集中在此，便于统一修改
DEFAULT_BASE_MODEL = "/mnt/hdd_1/home/cs16/Model/output/VLM/v3-20251222-211237/checkpoint-7633-merged"
DEFAULT_DATA_PATH = "/mnt/hdd_1/home/cs16/Data/dataAB_v5/vlm_train_AB_v5.jsonl"
DEFAULT_OUTPUT_DIR = "/mnt/hdd_1/home/cs16/Model/output/VLM"
DEFAULT_DEEPSPEED = "/mnt/hdd_1/home/cs16/vri-food/VLM/train/deepspeed_zero2.json"


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("yes", "true", "t", "1", "y"):
        return True
    if str(v).lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


@dataclass
class ModelArgs:
    base_model: str = DEFAULT_BASE_MODEL
    processor_base: str | None = None
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    enable_text_lora: bool = True
    enable_vision_lora: bool = True
    no_qlora: bool = False
    no_gradient_checkpointing: bool = False


@dataclass
class DataArgs:
    dataset: str = DEFAULT_DATA_PATH
    val_dataset: str | None = None
    output_dir: str = DEFAULT_OUTPUT_DIR
    cache_dir: str | None = None
    max_samples: int | None = None
    deepspeed: str | None = DEFAULT_DEEPSPEED
    eval_dump_predictions: str | None = None
    eval_max_new_tokens: int = 256
    eval_generate_batch_size: int = 1


@dataclass
class TrainArgs:
    epochs: float = 2.0
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05
    logging_steps: int = 50
    eval_steps: int = 1000
    save_steps: int = 1000
    save_total_limit: int = 10
    max_steps: int = -1
    resume_from_checkpoint: str | None = None
    init_from_checkpoint: str | None = None
    seed: int = 42
    skip_save: bool = False
    save_lora_only: bool = False
    label_threshold: float = 0.5
    dataloader_num_workers: int = 4
    train_lm: bool = True
    train_labels: bool = True
    train_weight: bool = True


@dataclass
class LossArgs:
    lambda_lm: float = 1.0
    lambda_lm_title: float = 1.0
    lambda_lm_ing: float = 0
    lambda_cuisine: float = 0.3
    lambda_meal: float = 0.3
    lambda_dish: float = 0.5
    lambda_amount: float = 0
    lambda_ratio: float = 0
    lambda_hinge: float = 0
    lambda_total_weight: float = 0
    # loss_schedule: JSON list，可按 step 切换权重，例如：
    # [{"start":0,"end":2000,"lambda_ratio":0.5},{"start":2000,"lambda_ratio":1.0}]
    loss_schedule: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-task fine-tune Qwen3-VL on food data with ingredient ratio heads."
    )
    # Model
    parser.add_argument("--base_model", type=str, default=ModelArgs.base_model)
    parser.add_argument(
        "--processor_base",
        type=str,
        default=ModelArgs.processor_base,
        help="Optional processor/tokenizer base path; defaults to --base_model if not set.",
    )
    parser.add_argument("--lora_r", type=int, default=ModelArgs.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=ModelArgs.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=ModelArgs.lora_dropout)
    parser.add_argument(
        "--enable_text_lora",
        type=str2bool,
        default=ModelArgs.enable_text_lora,
        help="Enable LoRA on text q/k/v/o projections (true/false).",
    )
    parser.add_argument(
        "--enable_vision_lora",
        type=str2bool,
        default=ModelArgs.enable_vision_lora,
        help="Enable LoRA on vision blocks (attn.qkv/proj, mlp.fc) (true/false).",
    )
    parser.add_argument("--no_qlora", action="store_true", default=ModelArgs.no_qlora)
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        default=ModelArgs.no_gradient_checkpointing,
    )

    # Data
    parser.add_argument("--dataset", type=str, default=DataArgs.dataset)
    parser.add_argument("--val_dataset", type=str, default=DataArgs.val_dataset)
    parser.add_argument("--output_dir", type=str, default=DataArgs.output_dir)
    parser.add_argument("--cache_dir", type=str, default=DataArgs.cache_dir)
    parser.add_argument("--deepspeed", type=str, default=DataArgs.deepspeed)
    parser.add_argument("--max_samples", type=int, default=DataArgs.max_samples)
    parser.add_argument(
        "--eval_dump_predictions",
        type=str,
        default=DataArgs.eval_dump_predictions,
        help="If set, will run generation on eval_dataset after training and dump to this JSONL path.",
    )
    parser.add_argument(
        "--eval_max_new_tokens",
        type=int,
        default=DataArgs.eval_max_new_tokens,
        help="Max new tokens for eval generation dump.",
    )
    parser.add_argument(
        "--eval_generate_batch_size",
        type=int,
        default=DataArgs.eval_generate_batch_size,
        help="Batch size for eval generation dump.",
    )

    # Training
    parser.add_argument("--epochs", type=float, default=TrainArgs.epochs)
    parser.add_argument("--train_batch_size", type=int, default=TrainArgs.train_batch_size)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=TrainArgs.gradient_accumulation_steps,
    )
    parser.add_argument("--learning_rate", type=float, default=TrainArgs.learning_rate)
    parser.add_argument("--warmup_ratio", type=float, default=TrainArgs.warmup_ratio)
    parser.add_argument("--logging_steps", type=int, default=TrainArgs.logging_steps)
    parser.add_argument("--eval_steps", type=int, default=TrainArgs.eval_steps)
    parser.add_argument("--save_steps", type=int, default=TrainArgs.save_steps)
    parser.add_argument("--save_total_limit", type=int, default=TrainArgs.save_total_limit)
    parser.add_argument("--max_steps", type=int, default=TrainArgs.max_steps)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=TrainArgs.dataloader_num_workers,
        help="Number of workers for the PyTorch DataLoader.",
    )
    parser.add_argument(
        "--train_lm",
        type=str2bool,
        default=TrainArgs.train_lm,
        help="Whether to train the base LM/vision model (true/false).",
    )
    parser.add_argument(
        "--train_labels",
        type=str2bool,
        default=TrainArgs.train_labels,
        help="Whether to train label heads (cuisine/meal/dish) (true/false).",
    )
    parser.add_argument(
        "--train_weight",
        type=str2bool,
        default=TrainArgs.train_weight,
        help="Whether to train weight heads (amount/ratio/total_weight) (true/false).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=TrainArgs.resume_from_checkpoint,
        help="Path to resume checkpoint.",
    )
    parser.add_argument(
        "--init_from_checkpoint",
        type=str,
        default=TrainArgs.init_from_checkpoint,
        help=(
            "Initialize model weights from a checkpoint state_dict (directory or file), "
            "but do NOT resume optimizer/scheduler/global_step. Useful for changing lambdas (scheme B)."
        ),
    )
    parser.add_argument("--seed", type=int, default=TrainArgs.seed)
    parser.add_argument(
        "--skip_save",
        action="store_true",
        default=TrainArgs.skip_save,
        help="Skip saving model/state at the end (useful for pure inference).",
    )
    parser.add_argument(
        "--save_lora_only",
        action="store_true",
        default=TrainArgs.save_lora_only,
        help="Save only LoRA adapters (and multitask heads) instead of full model.",
    )
    parser.add_argument(
        "--label_threshold",
        type=float,
        default=TrainArgs.label_threshold,
        help="Sigmoid threshold for decoding label predictions in eval dumps.",
    )

    # Loss weights
    parser.add_argument("--lambda_lm", type=float, default=LossArgs.lambda_lm)
    parser.add_argument("--lambda_lm_title", type=float, default=LossArgs.lambda_lm_title)
    parser.add_argument("--lambda_lm_ing", type=float, default=LossArgs.lambda_lm_ing)
    parser.add_argument("--lambda_cuisine", type=float, default=LossArgs.lambda_cuisine)
    parser.add_argument("--lambda_meal", type=float, default=LossArgs.lambda_meal)
    parser.add_argument("--lambda_dish", type=float, default=LossArgs.lambda_dish)
    parser.add_argument("--lambda_amount", type=float, default=LossArgs.lambda_amount)
    parser.add_argument("--lambda_ratio", type=float, default=LossArgs.lambda_ratio)
    parser.add_argument("--lambda_hinge", type=float, default=LossArgs.lambda_hinge)
    parser.add_argument("--lambda_total_weight", type=float, default=LossArgs.lambda_total_weight)
    parser.add_argument(
        "--loss_schedule",
        type=str,
        default=LossArgs.loss_schedule,
        help="JSON 列表，按 global_step 切换各 lambda，例如: "
        '[{"start":0,"end":2000,"lambda_ratio":0.5},{"start":2000,"lambda_ratio":1.0}]',
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    # 规范化路径
    args.output_dir = str(Path(args.output_dir))
    if args.cache_dir:
        args.cache_dir = str(Path(args.cache_dir))
    if args.resume_from_checkpoint:
        args.resume_from_checkpoint = str(Path(args.resume_from_checkpoint))
    if args.init_from_checkpoint:
        args.init_from_checkpoint = str(Path(args.init_from_checkpoint))
    if args.eval_dump_predictions:
        args.eval_dump_predictions = str(Path(args.eval_dump_predictions))
    return args
