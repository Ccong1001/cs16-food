# 模型部署包

这个文件夹包含了将 VLM 模型部署到生产环境所需的所有文件。

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `inference_example.py` | **【推荐】** 简化推理（仅LM head），适合App部署 |
| `inference_with_ratio.py` | 完整推理（LM + ratio head），需要食材比例时使用 |
| `merge_lora.py` | LoRA 合并脚本，将 checkpoint 合并为完整模型 |
| `test_inference.py` | 测试脚本，验证模型推理是否正常 |
| `requirements_inference.txt` | Python 依赖包列表 |
| `INFERENCE_MODES.md` | **【必读】** 两种推理方式的详细对比 |
| `API_SPEC.md` | API 接口规范文档 |
| `DELIVERY_CHECKLIST.md` | 交付清单 |

## 🎯 快速选择

根据你的需求选择推理方式：

```
需要食材比例吗？
├─ 不需要 → 使用 inference_example.py ✅ (推荐)
└─ 需要   → 使用 inference_with_ratio.py
```

详细对比请查看 [INFERENCE_MODES.md](INFERENCE_MODES.md)

---

## 🚀 快速开始 - 方式1（推荐）

### 步骤1: 合并 LoRA 模型

```bash
python merge_lora.py \
    --base_model /scratch/li96/zl9731/cs16/Model/output/VLM/v3-20251222-211237/checkpoint-7633-merged \
    --checkpoint /scratch/li96/zl9731/cs16/Model/output/VLM/v5.4-3/checkpoint-6000 \
    --output /scratch/li96/zl9731/cs16/Model/output/VLM/v5.4-3/checkpoint-6000-merged
```

### 步骤2: 安装依赖

```bash
pip install -r requirements_inference.txt
```

### 步骤3: 测试推理

```bash
python test_inference.py \
    /scratch/li96/zl9731/cs16/Model/output/VLM/v5.4-3/checkpoint-6000-merged \
    /path/to/test_image.jpg
```

### 步骤4: 集成到代码

```python
from inference_example import load_model, infer_single_image

# 加载模型 (只需一次)
model, processor = load_model("/path/to/checkpoint-XXXX-merged")

# 推理
result = infer_single_image(model, processor, "food.jpg")
print(result)
```

**输出示例**:
```
Title: Spaghetti Bolognese
Ingredients:
- pasta | 200g, dry
- beef mince | 100g
- tomato sauce | 50g
- onion | 1 medium, diced
- garlic | 2 cloves
```

---

## 🔬 高级用法 - 方式2（含 Ratio）

如果需要食材比例信息，使用 `inference_with_ratio.py`（两段式：先生成文本，再从文本解析食材并计算 ratio）：

```python
from inference_with_ratio import load_multitask_model, infer_with_ratio

# 加载完整模型（不需要 merge）
model, processor = load_multitask_model(
    checkpoint_path="/path/to/checkpoint-6000"
)

# 推理
result = infer_with_ratio(model, processor, "food.jpg")

print(result['text'])
# Title: Spaghetti Bolognese
# Ingredients: ...

print(result['ratio_probs'])
# [0.45, 0.25, 0.20, 0.10]  # 各食材的比例

# 如需手动指定食材列表（可选）
ingredients = ["pasta", "beef mince", "tomato sauce", "onion"]
result = infer_with_ratio(model, processor, "food.jpg", ingredients)
```

---

## 📚 文档

- [INFERENCE_MODES.md](INFERENCE_MODES.md) - **推理方式详细对比**
- [API_SPEC.md](API_SPEC.md) - API 接口规范
- [DELIVERY_CHECKLIST.md](DELIVERY_CHECKLIST.md) - 完整交付清单

---

## ⚙️ 模型信息

- **基座模型**: Qwen3-VL-8B-Instruct
- **训练方法**: LoRA (r=32, alpha=64)
- **参数量**: ~8B
- **输入**: 食物图片 (RGB, 建议 640×640 ~ 1280×1280)
- **输出**: 结构化文本 (菜名 + 食材列表)

---

## 🔧 推理配置

```python
# 基础配置
result = infer_single_image(
    model, processor, image_path,
    max_new_tokens=512,     # 生成长度
    temperature=0.2,        # 采样温度 (0=贪婪, >0=随机)
    top_p=0.9              # nucleus sampling
)

# 节省显存（使用4bit量化）
model, processor = load_model(
    model_path, 
    use_4bit=True,          # 启用4bit量化
    device="cuda:0"
)

# 批量推理（提高吞吐量）
from inference_example import infer_batch
results = infer_batch(
    model, processor,
    image_paths=["img1.jpg", "img2.jpg", ...],
    batch_size=8
)
```

---

## 💡 常见问题

### Q1: 推理时出现 "MultiTaskVLM" 相关错误？
A: 如果使用 `inference_example.py`（方式1），确保你使用的是 **merged 模型**，不是 checkpoint。先运行 `merge_lora.py`。

### Q2: 我需要食材比例信息，怎么办？
A: 使用 `inference_with_ratio.py`（方式2）。脚本会先生成食材列表，再自动解析并计算 ratio；也支持手动传入食材列表。需要 `VLM/train/model.py`。

### Q3: 部署到 App 该用哪种方式？
A: **推荐方式1** (`inference_example.py`)。简单、稳定，生成的文本已包含食材和数量信息。

### Q4: Ratio 数值的含义是什么？
A: Ratio 是 softmax 概率分布，表示各食材的相对占比，总和为 1.0。数值越大表示该食材占比越高。

---

## 📞 技术支持

详细问题请查看：
- [INFERENCE_MODES.md](INFERENCE_MODES.md) - 推理模式对比
- [API_SPEC.md](API_SPEC.md) - API 文档
