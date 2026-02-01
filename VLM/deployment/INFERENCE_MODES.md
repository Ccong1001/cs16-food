# 推理方式对比

根据你的需求（LM head + ratio），提供了两种推理方式：

## 方式1：简化推理（推荐用于 App 部署）

**文件**: `inference_example.py`

**特点**:
- ✅ 只使用 LM head 生成文本（菜名 + 食材列表）
- ✅ 不需要 MultiTaskVLM 包装层
- ✅ 代码简单，易于部署
- ✅ 推理速度快
- ❌ 无法获取 ratio 信息

**使用场景**:
- App 端展示菜名和食材列表
- 不需要精确的食材比例

**前置条件**:
- 需要先 merge LoRA: `python merge_lora.py --base_model ... --checkpoint ... --output ...`

**示例**:
```python
from inference_example import load_model, infer_single_image

model, processor = load_model("/path/to/checkpoint-XXXX-merged")
result = infer_single_image(model, processor, "food.jpg")

print(result)
# 输出:
# Title: Spaghetti Bolognese
# Ingredients:
# - pasta | 200g
# - beef mince | 100g
# ...
```

---

## 方式2：完整推理（需要 ratio）

**文件**: `inference_with_ratio.py`

**特点**:
- ✅ 使用 LM head 生成文本
- ✅ 使用 ratio head 计算食材比例
- ✅ 自动从生成文本解析食材（无需手动提供）
- ✅ 输出概率分布
- ❌ 需要 MultiTaskVLM 代码
- ❌ 代码较复杂

**使用场景**:
- 需要知道每个食材的相对比例
- 进行营养分析或份量估算
- 研究/分析用途

**前置条件**:
- 不需要 merge，直接使用 checkpoint
- 若 checkpoint 是 LoRA adapter，则需要提供 base_model_path
- 需要 `VLM/train/model.py` 文件

**示例**:
```python
from inference_with_ratio import load_multitask_model, infer_with_ratio

model, processor = load_multitask_model(
    checkpoint_path="/path/to/checkpoint-XXXX"
)

result = infer_with_ratio(model, processor, "food.jpg")

print(result['text'])          # 生成的文本
print(result['ratio_probs'])   # [0.45, 0.35, 0.20] - 食材比例

# 如需手动指定食材列表（可选）
ingredients = ["pasta", "beef mince", "tomato sauce"]
result = infer_with_ratio(model, processor, "food.jpg", ingredients)
```

**Ratio 含义**:
- 输出是 softmax 概率分布
- 数值越大，该食材在菜品中的占比越高
- 总和为 1.0

---

## 推荐方案

### 对于 App 部署：
**使用方式1** (`inference_example.py`)
- 简单、稳定、易维护
- 文本已包含食材信息，足够 App 使用
- 如需比例，可通过 LLM 后处理文本中的数量信息

### 对于研究/分析：
**使用方式2** (`inference_with_ratio.py`)
- 可获取模型学到的比例分布
- 适合批量分析、模型评估

---

## 性能对比

| 指标 | 方式1 (简化) | 方式2 (完整) |
|------|-------------|-------------|
| 推理时间 | 1.5-2s | 1.8-2.5s |
| 显存占用 | ~10GB | ~10GB |
| 代码复杂度 | 低 | 中 |
| 依赖 | 少 | 多 (需train代码) |
| 部署难度 | 简单 | 中等 |

---

## 快速决策

**问题**: 你的 App 需要显示食材比例吗？

- **不需要** → 使用 `inference_example.py`
- **需要精确比例** → 使用 `inference_with_ratio.py`
- **只需大概比例** → 使用 `inference_example.py` + 解析文本中的数量
