# Task 03: 训练核心超参升级与多模态正则化

## 1. 任务目标
在 [`train_modal.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/train_modal.py) 与 [`aicomp_grounding/training_state.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/aicomp_grounding/training_state.py) 中，落地已达成共识的训练端算法升级：
1. **LoRA 容量扩容**：`lora_rank = 32`, `lora_alpha = 64`（参数量增至 1.08 亿）。
2. **训练轮数与调度**：`NUM_EPOCHS = 3.5`，引入 **余弦退火（Cosine Annealing）**，并在末期保留 `1e-5` 底噪学习率。
3. **多模态正则化**：在 Dataset 加载时引入 `5%` 概率的 **Modality Dropout**（随机将单模态置黑），强迫跨模态协同。
4. **保持无损分辨率**：维持 `3072 * 28 * 28` 原图无损输入。

---

## 2. 目标文件
- **核心文件**：[`train_modal.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/train_modal.py)
- **元数据与校验**：[`aicomp_grounding/training_state.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/aicomp_grounding/training_state.py)

---

## 3. 具体修改规范

### 3.1 超参数字典更新 ([`train_modal.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/train_modal.py))
更新顶层常量与 `HYPERPARAMETERS` 字典：

```python
NUM_EPOCHS = 3.5  # 提升至 3.5 轮（约 630 步迭代）

HYPERPARAMETERS = {
    "batch_size": 1,
    "gradient_accumulation_steps": 16,     # 有效 batch size = 16
    "learning_rate": 1e-4,
    "epochs": NUM_EPOCHS,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",         # 切换为余弦退火
    "min_lr_ratio": 0.1,                   # 最终退火至 1e-5
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "lora_rank": 32,                       # 从 16 升至 32
    "lora_alpha": 64,                      # 保持 alpha = 2 * rank
    "lora_dropout": 0.05,
    "compute_dtype": "bfloat16",
    "autocast": True,
    "modality_dropout_prob": 0.05,         # 5% 模态丢弃概率
    "lora_targets": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "min_pixels": MIN_PIXELS,
    "max_pixels": MAX_PIXELS,              # 保持 3072 * 28 * 28
}
```

### 3.2 优化器学习率调度器构建
在 `train()` 函数中，使用 `get_cosine_schedule_with_warmup` 替换原调度器：
```python
from transformers import get_cosine_schedule_with_warmup

total_optimizer_steps = int(math.ceil((len(train_dataset) * NUM_EPOCHS) / GRAD_ACCUM_STEPS))
warmup_steps = int(total_optimizer_steps * HYPERPARAMETERS["warmup_ratio"])

scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_optimizer_steps,
    min_lr=1e-5,  # 保留底噪学习率
)
```

### 3.3 Dataset 中的多模态随机丢弃（Modality Dropout）
在 `RGBDTGroundingDataset.__getitem__` 中（仅在训练模式激活）：
```python
if self.is_train and random.random() < HYPERPARAMETERS.get("modality_dropout_prob", 0.0):
    drop_modality = random.choice(["rgb", "ir", "depth"])
    if drop_modality == "rgb":
        vis_image = Image.new("RGB", vis_image.size, (0, 0, 0))
    elif drop_modality == "ir":
        ir_image = Image.new("RGB", ir_image.size, (0, 0, 0))
    elif drop_modality == "depth":
        depth_image = Image.new("RGB", depth_image.size, (0, 0, 0))
```

---

## 4. 验证命令与验收标准
在本地运行训练状态构建单元测试，确保超参校验与步数计算 100% 正确：

```bash
/home/fang0/miniconda3/envs/qwen_vg/bin/python -c "
from aicomp_grounding.training_state import expected_global_steps, optimizer_steps_per_epoch

train_samples = 2875
grad_accum = 16
epochs = 3.5

steps_per_ep = optimizer_steps_per_epoch(train_samples, batch_size=1, grad_accum=grad_accum)
total_steps = expected_global_steps(train_samples, batch_size=1, grad_accum=grad_accum, epochs=epochs)

print(f'Steps per epoch: {steps_per_ep}')
print(f'Total global steps for 3.5 epochs: {total_steps}')
assert total_steps > 600, 'Global steps should be around 630'
print('Task 03 Hyperparameter Verification Passed successfully!')
"
```
