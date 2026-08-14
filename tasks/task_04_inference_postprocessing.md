# Task 04: 推理安全后处理、失败重试与可选 TTA 增强

## 1. 任务目标
在 [`run_inference.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/run_inference.py) 与 [`aicomp_grounding/bbox.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/aicomp_grounding/bbox.py) 中，集成高性价比的轻量后处理规则：
1. **极小目标框安全微调**：对面积小于 1.5% 的远景极小目标，微量 Padding 2%~3%，大幅提升小目标的 $\text{ACC@0.5}$ 容错率。
2. **Selective Retry 失败样本轻量抢救**：针对推理出现的极少数空框/拒答样本，自动用强约束 Prompt 极速重试一次。
3. **可选 TTA 水平翻转增强（支持 `--tta` 开关）**：基于单词边界正则（`\b`）实现精准无误伤的左右方位词镜像与坐标平均。

---

## 2. 目标文件
- **核心文件**：
  - [`aicomp_grounding/bbox.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/aicomp_grounding/bbox.py)
  - [`run_inference.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/run_inference.py)

---

## 3. 具体修改规范

### 3.1 极小目标框安全微调 ([`aicomp_grounding/bbox.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/aicomp_grounding/bbox.py))
在 `aicomp_grounding/bbox.py` 中新增/导出 `calibrate_bbox()`：

```python
def calibrate_bbox(box: list[float], pad_ratio: float = 0.03) -> list[float]:
    """
    对预测的归一化边界框做安全校准。
    若面积小于 1.5%（远景小目标），向外微扩 3% 的安全容错区。
    """
    x1, y1, x2, y2 = box
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    if 0.0 < w * h < 0.015:
        dx, dy = w * pad_ratio, h * pad_ratio
        return [
            max(0.0, round(x1 - dx, 4)),
            max(0.0, round(y1 - dy, 4)),
            min(1.0, round(x2 + dx, 4)),
            min(1.0, round(y2 + dy, 4)),
        ]
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]
```

### 3.2 Selective Retry 机制接入 ([`run_inference.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/run_inference.py))
在全量样本生成循环后，检查输出异常的样本：
```python
failed_indices = [
    i for i, r in enumerate(predictions)
    if r.get("bbox") is None or r.get("bbox") == [0.0, 0.0, 0.0, 0.0]
]

if failed_indices:
    print(f"Triggering selective retry for {len(failed_indices)} failed predictions...")
    for idx in failed_indices:
        item = test_items[idx]
        retry_messages = build_retry_grounding_messages(
            item["vis_path"], item["ir_path"], item["depth_path"], item["query"]
        )
        retry_box = model_inference_single(model, processor, retry_messages)
        if retry_box is not None:
            predictions[idx]["bbox"] = retry_box
```

### 3.3 可选 TTA 水平翻转与方位词原子镜像 ([`run_inference.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/run_inference.py))
1. 在 `run_inference.py` 中添加命令行参数 `--tta`（`action="store_true"`）。
2. 定义单词边界正则替换函数：

```python
import re

SWAP_DICT = {
    "leftmost": "rightmost",
    "rightmost": "leftmost",
    "left-hand": "right-hand",
    "right-hand": "left-hand",
    "left": "right",
    "right": "left",
}
SPATIAL_PATTERN = re.compile(r"\b(leftmost|rightmost|left-hand|right-hand|left|right)\b", re.IGNORECASE)

def swap_spatial_directions(text: str) -> str:
    def _repl(match):
        w = match.group(0)
        target = SWAP_DICT[w.lower()]
        if w.isupper(): return target.upper()
        if w[0].isupper(): return target.capitalize()
        return target
    return SPATIAL_PATTERN.sub(_repl, text)
```

3. 当指定 `--tta` 时，对图像做水平翻转并反算坐标后求数学平均值：
```python
# 原图预测: [ox1, oy1, ox2, oy2]
# 翻转图预测: [fx1, fy1, fx2, fy2]
# 镜像反算坐标:
rx1 = 1.0 - fx2
rx2 = 1.0 - fx1
final_box = [
    round((ox1 + rx1) / 2.0, 4),
    round((oy1 + fy1) / 2.0, 4),
    round((ox2 + rx2) / 2.0, 4),
    round((oy2 + fy2) / 2.0, 4),
]
```

---

## 4. 验证命令与验收标准
执行以下脚本，验证方位词互换与小目标微调逻辑：

```bash
/home/fang0/miniconda3/envs/qwen_vg/bin/python -c "
from aicomp_grounding.bbox import calibrate_bbox

# 1. 验证极小目标微调
small_box = [0.100, 0.100, 0.150, 0.150]  # area = 0.0025 < 0.015
calibrated = calibrate_bbox(small_box, pad_ratio=0.03)
assert calibrated[0] < small_box[0] and calibrated[2] > small_box[2], 'Padding failed!'

# 2. 验证方位词互换正则
from run_inference import swap_spatial_directions
test_s = 'The leftmost person with his left hand holding a bright red sign to the right of the tree'
expected = 'The rightmost person with his right hand holding a bright red sign to the left of the tree'
assert swap_spatial_directions(test_s) == expected, 'Directional swap mismatch!'
print('Task 04 Postprocessing Verification Passed successfully!')
"
```
