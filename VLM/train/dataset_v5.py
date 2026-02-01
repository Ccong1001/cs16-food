import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = -100
MAX_INGREDIENTS = 20
MAX_ING_TOKEN_LEN = 16
USER_PROMPT_TEXT = """
Given a food image, output only the recipe title and ingredient list in JSON format.
The JSON should look like this:
{"title": "<title>", "ingredients": [{"name": "<name>", "note": "<note>"}, ...]}
Do not output cuisine/meal/dish labels, amounts, or ratios.
"""

# 固定多标签 vocab
CUISINE_LABELS: Sequence[str] = (
    "american",
    "asian",
    "british",
    "caribbean",
    "central europe",
    "chinese",
    "eastern europe",
    "french",
    "greek",
    "indian",
    "italian",
    "japanese",
    "jewish",
    "korean",
    "kosher",
    "mediterranean",
    "mexican",
    "middle eastern",
    "nordic",
    "south american",
    "south east asian",
    "world",
)
MEAL_LABELS: Sequence[str] = (
    "breakfast",
    "brunch",
    "lunch/dinner",
    "snack",
)
DISH_LABELS: Sequence[str] = (
    "alcohol cocktail",
    "biscuits and cookies",
    "bread",
    "cereals",
    "cleaning",
    "condiments and sauces",
    "desserts",
    "drinks",
    "egg",
    "main course",
    "pancake",
    "preps",
    "preserve",
    "salad",
    "sandwiches",
    "snack",
    "soup",
    "special occasions",
    "starter",
    "unknown",
)

CUISINE2ID = {k: i for i, k in enumerate(CUISINE_LABELS)}
MEAL2ID = {k: i for i, k in enumerate(MEAL_LABELS)}
DISH2ID = {k: i for i, k in enumerate(DISH_LABELS)}

# amount_level 直接使用 4->large ... 0->trace
# Hinge/Cap 区间（可按数据分布再调）
AMOUNT_LOWER = {4: 0.25, 3: 0.12, 2: 0.03}
AMOUNT_UPPER = {4: 1.0, 3: 0.35, 2: 0.12, 1: 0.05, 0: 0.02}


@dataclass
class VisionConfig:
    """预留视觉尺寸配置（暂未使用，可按需扩展）"""

    min_pixels: int = 256
    max_pixels: int = 1280


def _safe_open_image(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        # 避免崩溃，给个黑图占位
        return Image.new("RGB", (224, 224), (0, 0, 0))


def _extract_assistant_json(raw: str) -> Dict[str, Any]:
    """从 assistant 文本中取出 JSON。"""
    try:
        match = re.search(r"```json\s*(.*?)```", raw, flags=re.S | re.I)
        content = match.group(1) if match else raw
        return json.loads(content)
    except Exception:
        return {}


def _build_assistant_text(title: str, ingredients: List[Dict[str, Any]]) -> str:
    return json.dumps({
        "title": title,
        "ingredients": [
            {"name": ing.get("name", ""), "note": ing.get("note", "")} for ing in ingredients
        ]
    }, ensure_ascii=False)


class VLMJsonlDataset(Dataset):
    """读取 jsonl，拆出多任务标签与文本序列。"""

    def __init__(
        self,
        data_path: Path,
        processor,
        vision_cfg: Optional[VisionConfig] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.processor = processor
        self.vision_cfg = vision_cfg or VisionConfig()
        self.samples: List[Dict[str, Any]] = []

        with self.data_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    self.samples.append(json.loads(line))
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw = self.samples[idx]
        messages = raw.get("messages", [])
        # user 消息文本，移除 <image>
        user_text = USER_PROMPT_TEXT

        assistant_raw = messages[-1]["content"] if messages else ""
        assistant_obj = _extract_assistant_json(assistant_raw)

        title = assistant_obj.get("title", "")
        labels = assistant_obj.get("labels", {}) or {}
        ing_list = assistant_obj.get("ingredients", []) or []

        # Multi-hot
        cuisine_vec = torch.zeros(len(CUISINE_LABELS), dtype=torch.float)
        for name in labels.get("cuisine_type", []) or []:
            if name in CUISINE2ID:
                cuisine_vec[CUISINE2ID[name]] = 1.0

        meal_vec = torch.zeros(len(MEAL_LABELS), dtype=torch.float)
        for name in labels.get("meal_type", []) or []:
            if name in MEAL2ID:
                meal_vec[MEAL2ID[name]] = 1.0

        dish_vec = torch.zeros(len(DISH_LABELS), dtype=torch.float)
        for name in labels.get("dish_type", []) or []:
            if name in DISH2ID:
                dish_vec[DISH2ID[name]] = 1.0

        # Ingredients
        ing_amount = torch.full((MAX_INGREDIENTS,), -1, dtype=torch.long)
        ing_ratio = torch.zeros(MAX_INGREDIENTS, dtype=torch.float)
        ing_mask = torch.zeros(MAX_INGREDIENTS, dtype=torch.float)
        ing_texts: List[str] = []

        for i, ing in enumerate(ing_list[:MAX_INGREDIENTS]):
            name = ing.get("name", "")
            note = ing.get("note", "")
            level = ing.get("amount_level", None)
            ratio = ing.get("ratio", 0.0)
            # 只在 level 合法时启用该食材监督
            if isinstance(level, int) and 0 <= level <= 4:
                ing_amount[i] = level
                ing_ratio[i] = float(ratio)
                ing_mask[i] = 1.0
            ing_texts.append(f"{name} {note}".strip())

        assistant_text = _build_assistant_text(title, ing_list[:MAX_INGREDIENTS])

        image_path = raw.get("images", [None])[0]
        image = _safe_open_image(image_path) if image_path else Image.new("RGB", (224, 224))

        chat_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text or "Classify this dish type."},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]

        return {
            "messages": chat_messages,
            "image": image,
            "labels": assistant_text,
            "cuisine_labels": cuisine_vec,
            "meal_labels": meal_vec,
            "dish_labels": dish_vec,
            "ingredient_amount": ing_amount,
            "ingredient_ratio": ing_ratio,
            "ingredient_mask": ing_mask,
            "ingredient_texts": ing_texts,
        }


class MultiTaskCollator:
    """将样本批量化并构造所需张量。"""

    def __init__(self, processor, lambda_lm_ing: float = 0.0):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.lambda_lm_ing = lambda_lm_ing
        # Tokens used to locate ingredient段；用于提升 loss_ing 覆盖范围
        self._ingredient_tokens = self.tokenizer.encode("Ingredients:", add_special_tokens=False)
        # 兼容可能的前置空格/换行，增加备用匹配模式
        self._ingredient_patterns = []
        for s in ("Ingredients:", "\nIngredients:", " Ingredients:"):
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if ids:
                self._ingredient_patterns.append(ids)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts: List[str] = []
        images: List[Image.Image] = []
        assistant_texts: List[str] = []
        for ex in batch:
            prompt = self.processor.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            texts.append(prompt)
            images.append(ex["image"])
            assistant_texts.append(ex.get("labels", ""))

        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        lm_weights = torch.zeros_like(labels, dtype=torch.float)

        # 只训练 assistant 段
        header_tokens = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        newline_token = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        for i in range(len(labels)):
            labels[i, :] = IGNORE_INDEX
            seq = input_ids[i]
            seq_len = seq.size(0)
            w = len(header_tokens)
            start_pos = -1
            for j in range(seq_len - w):
                if torch.equal(seq[j : j + w], torch.tensor(header_tokens, device=seq.device)):
                    start_pos = j + w
                    if start_pos < seq_len and seq[start_pos] == newline_token:
                        start_pos += 1
                    break
            if start_pos != -1:
                labels[i, start_pos:] = seq[start_pos:]
                lm_weights[i, start_pos:] = 1.0  # 简单区分：assistant 段全权重 1
                # 如果启用 lm_ing，则将 Ingredients 段单独标记为 2.0
                if self.lambda_lm_ing != 0:
                    ing_start = -1
                    # 先尝试 token 模式匹配
                    for pat_ids in self._ingredient_patterns:
                        pat = torch.tensor(pat_ids, device=seq.device)
                        pat_len = pat.size(0)
                        for k in range(start_pos, seq_len - pat_len + 1):
                            if torch.equal(seq[k : k + pat_len], pat):
                                ing_start = k
                                break
                        if ing_start != -1:
                            break
                    # 若未匹配上，按文本长度回退：分割 Title / Ingredients 段
                    if ing_start == -1:
                        text = assistant_texts[i] if i < len(assistant_texts) else ""
                        if text and "Ingredients:" in text:
                            title_prefix = text.split("Ingredients:", 1)[0]
                            title_ids = self.tokenizer.encode(title_prefix, add_special_tokens=False)
                            ing_start = start_pos + min(len(title_ids), seq_len - start_pos)
                    if ing_start != -1:
                        lm_weights[i, ing_start:] = 2.0

        # Stack classification labels
        def stack(key):
            return torch.stack([ex[key] for ex in batch], dim=0)

        cuisine = stack("cuisine_labels")
        meal = stack("meal_labels")
        dish = stack("dish_labels")
        amount = stack("ingredient_amount")
        ratio = stack("ingredient_ratio")
        ing_mask = stack("ingredient_mask")

        # Ingredient tokens for ratio/amount heads
        ing_tokens = torch.zeros(
            (len(batch), MAX_INGREDIENTS, MAX_ING_TOKEN_LEN), dtype=torch.long
        )
        ing_token_mask = torch.zeros_like(ing_tokens, dtype=torch.float)
        for b_idx, ex in enumerate(batch):
            for i, text in enumerate(ex["ingredient_texts"][:MAX_INGREDIENTS]):
                if not text:
                    continue
                tok = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=MAX_ING_TOKEN_LEN,
                )
                ids = tok["input_ids"]
                ing_tokens[b_idx, i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                ing_token_mask[b_idx, i, : len(ids)] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": inputs.get("attention_mask"),
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "labels": labels,
            "lm_weights": lm_weights,
            "cuisine_labels": cuisine,
            "meal_labels": meal,
            "dish_labels": dish,
            "ingredient_amount": amount,
            "ingredient_ratio": ratio,
            "ingredient_mask": ing_mask,
            "ingredient_token_ids": ing_tokens,
            "ingredient_token_mask": ing_token_mask,
        }


__all__ = [
    "VLMJsonlDataset",
    "MultiTaskCollator",
    "VisionConfig",
    "CUISINE_LABELS",
    "MEAL_LABELS",
    "DISH_LABELS",
    "AMOUNT_LOWER",
    "AMOUNT_UPPER",
]
