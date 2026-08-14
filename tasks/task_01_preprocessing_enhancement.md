# Task 01: 传感器预处理增强（深度截断收紧至 13m 与红外 CLAHE 增强）

## 1. 任务目标
优化 [`scripts/prepare_rgbdt.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/scripts/prepare_rgbdt.py) 中的传感器处理流水线：
1. **深度图**：将最大截断距离 `max_depth_mm` 从 `20000` (20m) 收紧至 **`13000` (13m)**，将 256 个色彩阶梯聚焦于 0.3m~13.0m 真实目标活跃区。
2. **红外图**：引入 **限制对比度自适应直方图均衡化（CLAHE）**，增强暗区发热活体与车辆的边缘热辐射轮廓。

---

## 2. 目标文件
- **核心文件**：[`scripts/prepare_rgbdt.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/scripts/prepare_rgbdt.py)

---

## 3. 具体修改规范

### 3.1 深度图处理优化：`render_depth_to_jet`
在 `render_depth_to_jet` 函数中：
- 默认参数更新为 `min_depth_mm: int = 300`, `max_depth_mm: int = 13000`。
- 将 `[300, 13000]` 的毫米深度线性映射到 `[0, 255]` 并应用 `cv2.COLORMAP_JET`。
- 无效深度像素（`depth == 0` 或 `depth > 13000`）在 BGR 空间赋值为 `[128, 0, 0]`（深蓝底色）。

```python
def render_depth_to_jet(
    depth_raw: np.ndarray,
    min_depth_mm: int = 300,
    max_depth_mm: int = 13000,
) -> np.ndarray:
    valid_mask = (depth_raw >= min_depth_mm) & (depth_raw <= max_depth_mm)
    depth_clipped = np.clip(depth_raw, min_depth_mm, max_depth_mm).astype(np.float32)
    normalized = ((depth_clipped - min_depth_mm) / (max_depth_mm - min_depth_mm) * 255.0).astype(np.uint8)
    
    jet_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    jet_bgr[~valid_mask] = [128, 0, 0]
    jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)
    return jet_rgb
```

### 3.2 红外图处理优化：`render_infrared_enhanced`
- 新增/优化函数 `render_infrared_enhanced(ir_raw: np.ndarray) -> np.ndarray`。
- 对单通道红外应用 `clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))`。
- 增强后将单通道转换为标准 3 通道 RGB 图像并保存。

```python
def render_infrared_enhanced(ir_raw: np.ndarray) -> np.ndarray:
    if len(ir_raw.shape) == 3:
        ir_gray = cv2.cvtColor(ir_raw, cv2.COLOR_RGB2GRAY)
    else:
        ir_gray = ir_raw.copy()
        
    if ir_gray.dtype != np.uint8:
        ir_gray = cv2.normalize(ir_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(ir_gray)
    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    return enhanced_rgb
```

### 3.3 主流程管道集成
确保在遍历帧保存深度和红外图时，统一调用上述优化后的函数进行保存。

---

## 4. 验证命令与验收标准
执行以下单样本测试命令，确保生成的深度图与红外图为合法的 1920×1080 3 通道 uint8 数组：

```bash
/home/fang0/miniconda3/envs/qwen_vg/bin/python -c "
import cv2, numpy as np
from scripts.prepare_rgbdt import render_depth_to_jet, render_infrared_enhanced

# Mock 16-bit depth & 8-bit IR
fake_depth = np.random.randint(0, 15000, (1080, 1920), dtype=np.uint16)
fake_ir = np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)

depth_rgb = render_depth_to_jet(fake_depth, max_depth_mm=13000)
ir_rgb = render_infrared_enhanced(fake_ir)

assert depth_rgb.shape == (1080, 1920, 3) and depth_rgb.dtype == np.uint8, 'Depth shape/dtype mismatch'
assert ir_rgb.shape == (1080, 1920, 3) and ir_rgb.dtype == np.uint8, 'IR shape/dtype mismatch'
print('Task 01 Verification Passed successfully!')
"
```
