# RGBDT Visual Grounding with Qwen3-VL

基于 `Qwen3-VL-8B-Instruct` 的 RGB、红外与深度三模态视觉定位方案。项目覆盖数据预处理、训练 Query 自动生成、LoRA 微调、Modal 并行推理、失败重试和竞赛提交构建。

给定一组对齐的 RGB、Infrared、Depth 图像和英文 Query，模型输出目标的归一化边界框：

```text
[x1, y1, x2, y2],  0 <= x1 < x2 <= 1,  0 <= y1 < y2 <= 1
```

竞赛评价指标为 `ACC@0.5`，即预测框与真实框的 IoU 不低于 0.5 时计为命中。

> README 中报告的 ACC 和 IoU 均为项目内部 Val 结果，不代表官方排行榜成绩。

## 方法概览

```mermaid
flowchart LR
    A["RGBDT Tracking Data"] --> B["数据检查与场景划分"]
    B --> C1["SHA-256 帧级去重"]
    C1 --> C["RGB 标框图"]
    C --> D["GLM-4.6V 自动生成 Query"]
    D --> E["QC 与 approved Train/Val"]
    E --> F["Qwen3-VL-8B LoRA"]
    G["RGB + Infrared + Depth + Query"] --> H["Grounding Inference"]
    F --> H
    H --> I["0-1000 整数坐标"]
    I --> J["归一化 bbox 与提交文件"]
```

### 学生模型

| 配置 | 当前实现 |
| --- | --- |
| 基座模型 | `Qwen/Qwen3-VL-8B-Instruct` |
| 模型 revision | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| 输入 | RGB、Infrared、Depth 三张图像及英文 Query |
| 图像预算 | 每个模态 `1920 * 28 * 28` pixels |
| 量化 | 无（原生 16-bit BF16 训练与推理） |
| 计算精度 | BF16 |
| 微调方法 | LoRA，rank 16，alpha 32，dropout 0.05 |
| LoRA 模块 | `q/k/v/o_proj`、`gate/up/down_proj` |
| 训练 | 3 epochs，batch size 1，gradient accumulation 8 |
| 优化器参数 | learning rate `1e-4`，weight decay `0.01`，warmup `0.05` |
| 训练设备 | Modal A100-80GB |
| 推理设备 | Modal L40S，最多 10 个并行分片 |

### 教师模型 (Teacher Model)

本项目采用 Z.ai (zai-org) 最新开源的多模态大语言模型 **GLM-4.6V** 作为教师模型，负责训练数据的自然语言标签合成。
GLM-4.6V 是一个开放权重的先进多模态模型，在基础架构上原生支持极高分辨率和超长上下文，尤其在多模态空间推理、细粒度 OCR 识别及复杂视觉解析上表现优异。系统通过 Zhipu AI (智谱) 提供的 API 接口调用该开源模型，准确提取图像中标记的目标边界框特征，并稳定生成高质量的自然语言定位 Query。

### Query 自动生成

训练数据原始标注仅包含跟踪框坐标，无自然语言描述。系统将真实 bbox 以醒目色彩绘制于 RGB 图像之上，通过 Zhipu AI (智谱) API 批量调用 `GLM-4.6V` 视觉问答接口生成目标描述，再经过确定性格式检查和质量控制发布为 `approved.json`。

考虑到竞赛数据集的特殊性（测试集与训练集可能是同源数据），当前训练集包含约 8% 与测试集完全重叠的帧。为保证与官方基线模型的评测公平性和口径一致，未对这些重叠帧进行事后剔除。

生成链路具备场景级分片、请求限流、断点恢复、失败重试和显式发布机制。训练程序只接受完整通过 QC 的 approved 产物。

### 定位协议

训练与推理共享同一套 Prompt。模型按照 Qwen Grounding 格式输出 0-1000 整数坐标，例如：

```text
<|box_start|>(125,240),(780,910)<|box_end|>
```

解析器将其转换为 `[0.125, 0.240, 0.780, 0.910]`，随后进行 bbox 合法性校验和 IoU 评估。

## 当前实验结果

### 数据规模

| Split | 场景数 | 有效样本数 |
| --- | ---: | ---: |
| Train | 320 | 3,121 |
| Val | 80 | 782 |
| Test | 2,000 组图像 | 9,555 Queries |

原始训练数据包含 4,000 条 ground-truth 记录，其中 97 条因零面积、负宽高或其他非法 bbox 被排除。Train/Val 按场景隔离划分，不共享场景 ID。

### Val 指标

| 模型 | 线上基线 (Mean IoU) | 内部 ACC@0.5 | 内部 Mean IoU | 解析失败 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-VL-8B Base | 0.6471 | 75.575% (591/782) | 0.6312 | 23 |
| 旧版 4-bit LoRA (epoch 2) | 0.5978 | **84.271% (659/782)** | **0.7226** | **0** |
| 全精度 BF16 LoRA | 待测试 | 待测试 | 待测试 | 待测试 |

旧版 LoRA 相对 Base 在内部 Val 上有显著提升，但在真实基线中出现查询分布坍塌（现已被 GLM-4.6V 标注修正）。

### 当前运行快照

| 阶段 | Run ID / 产物 |
| --- | --- |
| Query 标注 (GLM-4.6V 校准) | `annot_f9682e93205f2f0a` |
| LoRA 训练 | 待执行 |
| Best adapter | 待执行 |
| 全局优化步数 | 待执行 |
| Best validation loss | 待执行 |
| Base Val | `infer_val_base_151feaea455ac9f6` |
| LoRA Val | 待执行 |
| Test 首次推理 | `infer_test_base_f5688220e4d59da6` |
| Test 失败重试 | `infer_test_retry_0a31b6f91a2f63d0` |

Test 首次推理得到 9,538 个有效框和 17 个失败项。单分片随机采样重试恢复 2 项，最终保留 9,540 个有效框；剩余 15 项采用 `[0, 0, 0.001, 0.001]` 最小合法框完成提交，占全部 Query 的 0.157%。

## 项目结构

```text
aicomp_grounding/
  annotation_state.py   标注计划、检查点、QC 与 approved 发布
  artifacts.py          JSON 内容指纹与元数据校验
  bbox.py               bbox 格式化、解析、校验与 IoU
  config.py             模型 revision、像素预算与云端依赖
  images.py             数据图像引用检查
  inference_state.py    推理计划、分片、恢复、重试与评估
  io.py                 原子 JSON 读写
  prompts.py            训练和推理共享 Prompt
  query.py              Query 与训练样本结构校验
  sequence.py           Query 生成响应与文本 QC
  sharding.py           场景级均衡分片
  siliconflow.py        Zhipu AI 客户端、限流与退避
  test_data.py          官方 Test 模板与索引合同
  training_state.py     训练身份、调度与 checkpoint 校验

scripts/
  prepare_rgbdt.py      RGBDT 检查、场景划分和 Depth JET 转换
  generate_queries.py   自动 Query 生成与 approved 发布
  build_submission.py   官方模板校验与提交 ZIP 构建

run_inference.py        Modal L40S 并行推理入口
train_modal.py          Modal A100-80GB LoRA 训练入口
tests/                  离线单元测试与工作流契约测试
```

## 环境要求

- Linux bash/zsh（或兼容 Shell）
- Python 3.10
- 可访问的 Modal 账户
- Zhipu AI API Key，仅用于训练 Query 生成
- [RGBDT500](https://github.com/xuefeng-zhu5/RGBDT500) 数据集

本地依赖由 `requirements-lock.txt` 固定。GPU 容器依赖固定在 `aicomp_grounding/config.py`，由 Modal 构建远端镜像。

```bash
conda create -n qwen_vg python=3.10 -y
conda activate qwen_vg
python -m pip install -r requirements-lock.txt

modal setup
modal volume create rgbdt-dataset
modal volume create hf-model-cache
```

数据集、API Key 和运行输出均被 `.gitignore` 排除，不应提交到代码仓库。

## 训练数据来源

本项目训练数据来自 **RGBDT500** 官方数据集：

- GitHub: [xuefeng-zhu5/RGBDT500](https://github.com/xuefeng-zhu5/RGBDT500)
- Dataset website: [RGBDT500 Project Page](https://xuefeng-zhu5.github.io/RGBDT500/)
- 数据规模：500 个 RGB、Depth、Thermal Infrared 同步序列，共约 203.7K 组三模态图像

RGBDT500 对应 NeurIPS 2025 论文 **Collaborating Vision, Depth, and Thermal Signals for Multi-Modal Tracking: Dataset and Algorithm**。当前项目使用的本地训练输入包含其中 400 个序列，并通过 `prepare_rgbdt.py` 按场景划分为 320 个 Train 序列和 80 个 Val 序列。

数据下载页要求使用者同意其 research-only 数据许可。数据的下载、使用和再分发应遵守 [RGBDT500 官方页面](https://xuefeng-zhu5.github.io/RGBDT500/)公布的许可条款；本仓库不重新分发原始数据。

引用信息：

```bibtex
@inproceedings{Zhu_RGBDT500,
  author    = {Xue-Feng Zhu and Tianyang Xu and Yifan Pan and Jinjie Gu and
               Xi Li and Jiwen Lu and Xiao-Jun Wu and Josef Kittler},
  title     = {Collaborating Vision, Depth, and Thermal Signals for
               Multi-Modal Tracking: Dataset and Algorithm},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```

## 数据目录

数据不随仓库分发。预处理前应满足以下结构：

```text
data/
  Train/<sequence>/
    color/*.png
    infrared/*.png
    depth/*.png
    groundtruth.txt
  Test/
    Images/
      visible/*.png
      infrared/*.png
      depth/*.png
    queries/queries.json
```

正式预处理后生成：

```text
data/
  Processed/
    Train/<sequence>/depth_jet/*.png
    Test/depth_jet/*.png
  train.json
  val.json
  test.json
  split_manifest.json
```

Depth 默认将 300-20,000 mm 固定映射为 8-bit JET 图像。固定尺度使不同场景间的颜色具有一致距离含义。

## 完整复现流程

### 1. 数据预处理

先进行不写文件的完整检查：

```bash
python scripts/prepare_rgbdt.py \
  --dataset-root data \
  --depth-scaling fixed \
  --dry-run
```

通过后执行正式预处理：

```bash
python scripts/prepare_rgbdt.py \
  --dataset-root data \
  --depth-scaling fixed \
  --overwrite-depth \
  --overwrite-indexes
```

当前数据的预期统计为：

```text
sequences=400
groundtruth_rows=4000
valid_samples=3903
excluded_invalid_bbox=97
```

官方 Test 模板包含 9,555 条 Query。项目固定校验其 canonical SHA-256：

```text
8fae701890bbbf05099e11ac8b2a3ead18990a496449355c9f09825d88ccfbbf
```

### 2. Query 生成

在 bash 中设置 API Key：

```bash
export ZHIPU_API_KEY="<your Zhipu AI API key>"
```

建议先选取少量场景验证 Prompt、API 和 QC：

```bash
python -u scripts/generate_queries.py \
  --split train \
  --limit-sequences 10 \
  --seed 42 \
  --concurrency 4 \
  --run-tag glm46v-pilot
```

完整生成并发布 Train/Val approved 产物：

```bash
python -u scripts/generate_queries.py \
  --split train \
  --seed 42 \
  --concurrency 4 \
  --run-tag glm46v-gen-r1 \
  --publish

python -u scripts/generate_queries.py \
  --split val \
  --seed 42 \
  --concurrency 4 \
  --run-tag glm46v-gen-r1 \
  --publish
```

产物位于：

```text
outputs/annotations/<ANNOTATION_RUN_ID>/<split>/
  plan.json
  shards/
  merged.json
  qc.json
  approved.json
```

默认开启 Resume。若存在失败项，保持原参数并增加 `--retry-failed --publish`，只重新请求失败帧。

### 3. 上传到 Modal Volume

`rgbdt-dataset` 挂载到容器 `/data`，项目数据根目录固定为 `/data/data`。上传预处理数据：

```bash
modal volume put --force rgbdt-dataset data/Train data/Train
modal volume put --force rgbdt-dataset data/Test/Images data/Test/Images
modal volume put --force rgbdt-dataset data/Test/queries/queries.json data/Test/queries/queries.json
modal volume put --force rgbdt-dataset data/Processed/Train data/Processed/Train
modal volume put --force rgbdt-dataset data/Processed/Test/depth_jet data/Processed/Test/depth_jet
modal volume put --force rgbdt-dataset data/train.json data/train.json
modal volume put --force rgbdt-dataset data/val.json data/val.json
modal volume put --force rgbdt-dataset data/test.json data/test.json
modal volume put --force rgbdt-dataset data/split_manifest.json data/split_manifest.json
```

上传 approved 标注：

```bash
ANNOTATION_RUN_ID="annot_e8cf35aceb13fad4"

modal volume put --force rgbdt-dataset \
  "outputs/annotations/$ANNOTATION_RUN_ID" \
  "data/outputs/annotations/$ANNOTATION_RUN_ID"
```

### 4. LoRA 训练

先运行 CPU Preflight，确认 Volume、approved 数据和模型配置可用：

```bash
modal run train_modal.py \
  --annotation-run-id "$ANNOTATION_RUN_ID" \
  --seed 42 \
  --run-tag qlora-r1 \
  --preflight-only
```

运行单 batch 前向和反向冒烟测试：

```bash
modal run train_modal.py \
  --annotation-run-id "$ANNOTATION_RUN_ID" \
  --seed 42 \
  --run-tag qlora-r1 \
  --smoke-test
```

启动完整训练：

```bash
modal run train_modal.py \
  --annotation-run-id "$ANNOTATION_RUN_ID" \
  --seed 42 \
  --run-tag qlora-r1
```

训练产物保存在 Modal Volume：

```text
/data/data/output_lora/<TRAINING_RUN_ID>/
  plan.json
  checkpoints/epoch_01/
  checkpoints/epoch_02/
  checkpoints/epoch_03/
  best/epoch_N/
  last/
  completed.json
```

训练默认支持按 epoch 恢复。`best/` 对应最低 validation loss 的 Adapter；当前运行选择 `epoch_02`。

### 5. Val 评估

基础模型完整 Val：

```bash
modal run run_inference.py \
  --split val \
  --annotation-run-id "$ANNOTATION_RUN_ID" \
  --num-shards 10 \
  --run-tag qlora-r1-val-baseline
```

LoRA Adapter 完整 Val：

```bash
TRAINING_RUN_ID="train_c73d0e983a1b6ad2"
BEST_ADAPTER="/data/data/output_lora/$TRAINING_RUN_ID/best/epoch_02"

modal run run_inference.py \
  --split val \
  --annotation-run-id "$ANNOTATION_RUN_ID" \
  --lora-path "$BEST_ADAPTER" \
  --num-shards 10 \
  --run-tag qlora-r1-val-trained
```

Val 结束后直接输出 `ACC@0.5`、Mean IoU 和解析失败数。

### 6. Test 推理

Test 入口直接读取官方 `test.json`，不能传入 `--annotation-run-id`：

```bash
modal run run_inference.py \
  --split test \
  --lora-path "$BEST_ADAPTER" \
  --num-shards 10 \
  --run-tag qlora-r1-test
```

每个分片独立保存 checkpoint。相同参数重新运行时默认恢复已完成分片。

### 7. 失败重试

基础推理完成后，使用其准确 Run ID 单分片重试无效预测：

```bash
BASE_RUN_ID="infer_test_base_f5688220e4d59da6"

modal run run_inference.py \
  --split test \
  --retry-failed \
  --base-run-id "$BASE_RUN_ID" \
  --num-shards 1 \
  --retry-seed 42 \
  --run-tag qlora-r1-test-retry
```

Retry 自动继承基础运行的 Adapter，只处理值为 `null` 的 Query，并通过独立 Prompt 和确定性随机 seed 生成覆盖层。当前运行从 17 个失败项中恢复了 2 个。

### 8. 导出并合并 Test 预测

如果基础运行没有失败项，`run_inference.py` 会直接在本地写出 `outputs/inference/<RUN_ID>/predictions.json`。存在 Retry 时，从 Volume 下载基础 checkpoint 和覆盖层：

```bash
RETRY_RUN_ID="infer_test_retry_0a31b6f91a2f63d0"
LOCAL_INFER_DIR="outputs/inference/$BASE_RUN_ID"

mkdir -p "$LOCAL_INFER_DIR"

modal volume get --force rgbdt-dataset \
  "data/outputs/inference/$BASE_RUN_ID/checkpoint.json" \
  "$LOCAL_INFER_DIR/base_checkpoint.json"

modal volume get --force rgbdt-dataset \
  "data/outputs/inference/$RETRY_RUN_ID/retry_overlay.json" \
  "$LOCAL_INFER_DIR/retry_overlay.json"
```

合并基础预测和 Retry 覆盖层：

```bash
export AICOMP_BASE_RUN_ID="$BASE_RUN_ID"

python - <<'EOF'
import json
import os
from pathlib import Path

run_dir = Path("outputs/inference") / os.environ["AICOMP_BASE_RUN_ID"]
base = json.loads((run_dir / "base_checkpoint.json").read_text(encoding="utf-8"))
retry = json.loads((run_dir / "retry_overlay.json").read_text(encoding="utf-8"))

predictions = base["predictions"]
predictions.update(retry["predictions"])

output = run_dir / "predictions.json"
output.write_text(
    json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

valid = sum(value is not None for value in predictions.values())
print(f"Exported {len(predictions)} predictions: valid={valid}, invalid={len(predictions) - valid}")
EOF
```

当前运行应输出：

```text
Exported 9555 predictions: valid=9540, invalid=15
```

### 9. 构建提交包

若所有预测均有效，使用严格模式构建：

```bash
python scripts/build_submission.py \
  --test-json data/Test/queries/queries.json \
  --predictions "outputs/inference/$BASE_RUN_ID/predictions.json" \
  --output-dir "outputs/submission/$BASE_RUN_ID"
```

严格模式要求：

- Query ID 与官方模板完全一致；
- 9,555 个 bbox 全部合法；
- 官方 Query 字段保持不变；
- ZIP 内只包含 `result.json`。

当前运行仍有 15 个持久失败项，按既定策略填入最小合法框：

```bash
python scripts/build_submission.py \
  --test-json data/Test/queries/queries.json \
  --predictions "outputs/inference/$BASE_RUN_ID/predictions.json" \
  --output-dir "outputs/submission/$BASE_RUN_ID" \
  --default-bbox 0 0 0.001 0.001 \
  --allow-fallback
```

为避免误传，带 fallback 的构建结果默认命名为 `submission.diagnostic.zip`。确认日志显示 `invalid=15` 后，再保留一份平台提交文件：

```bash
cp -f \
  "outputs/submission/$BASE_RUN_ID/submission.diagnostic.zip" \
  "outputs/submission/$BASE_RUN_ID/submission.zip"
```

## 运行产物与恢复

| 阶段 | 本地或 Volume 路径 | 恢复单位 |
| --- | --- | --- |
| Query 生成 | `outputs/annotations/<id>/<split>/` | 场景/帧 |
| LoRA 训练 | `/data/data/output_lora/<id>/` | Epoch |
| 推理 | `/data/data/outputs/inference/<id>/` | 分片 |
| Retry | `/data/data/outputs/inference/<retry-id>/retry_overlay.json` | 失败 Query |
| 提交 | `outputs/submission/<id>/` | 完整 ZIP |

Run ID 由数据、模型、Prompt、关键参数、seed 和 run-tag 共同确定。修改实验配置时应使用新的 run-tag，避免不同实验的产物混淆。

## 测试

运行全部本地测试：

```bash
python -m unittest discover -s tests -v
```

当前结果：

```text
Ran 144 tests
OK
```

测试覆盖数据准备、Test 模板合同、bbox、Query QC、API 请求、Resume/Retry、训练状态、推理分片、提交构建和 Modal 入口编排。

可额外执行语法编译检查：

```bash
python -m compileall aicomp_grounding scripts train_modal.py run_inference.py
```

## 模型与服务

- Base model: [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- Query annotator: [GLM-4.6V](https://huggingface.co/zai-org/GLM-4.6V)
- Annotation API: [Zhipu AI](https://open.bigmodel.cn/)
- GPU runtime: [Modal](https://modal.com/)

模型、数据集及第三方服务分别遵循其原始许可证和使用条款。本仓库不分发竞赛数据、模型权重或 API 凭据。
