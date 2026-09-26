# RGB-Grounding

第八届全球校园人工智能算法精英大赛 · 算法挑战赛道 · 基于大模型的多模态视觉理解与推理

赛题链接：<https://www.aicomp.cn>

## 方法概述

输入一张可见光图像与一句英文目标描述（query），输出目标在图像中的归一化边界框
`[x1, y1, x2, y2]`。指标为 ACC@0.5（预测框与真值框 IoU ≥ 0.5 计为命中）。

训练与推理仅使用可见光（RGB）图像。红外与深度数据在原始数据集中提供，不进模型输入。

三款视觉大模型均采用 LoRA 微调。推理后可通过加权框融合（WBF）集成多模型预测。

| `--model` | 底座权重 | 微调方式 |
| --- | --- | --- |
| `qwen3vl` | `Qwen/Qwen3-VL-8B-Instruct` | LoRA |
| `qwen3_5` | `Qwen/Qwen3.5-9B` | LoRA |
| `glm46v` | `zai-org/GLM-4.6V-Flash` | LoRA |
| `mock` | 无 | 测试占位 |

## 项目结构

```text
aicomp_grounding/                核心库
  __init__.py                    bbox 工具导出
  bbox.py                       坐标转换、解析、IoU 计算
  query.py                      query 文本校验、空间词翻转增强
  contract.py                   approved.json 契约校验与指纹
  config.py                     跨模型常量与 YAML 配置加载
  artifacts.py                  SHA-256 确定性哈希
  images.py                     图像指纹绑定
  io.py                         原子 JSON 读写
  paths.py                      仓库级路径策略
  sharding.py                   场景级分片
  testset.py                    官方测试集模板校验
  annotator/                    标注工具（stdlib-only HTTP 服务器）
    server.py                   后端：会话构建、HTTP handler
    store.py                    JSONL 日志 + 原子快照存储
    bbox.py                     归一化坐标校验
    static/                     前端：canvas 标注界面
  grounding/                    模型侧
    models/                     模型适配器（qwen3vl / qwen3_5 / glm46v / mock）
      base.py                   适配器协议定义
    engine/                     训练与推理引擎
      training_core.py          LoRA 训练主循环
      training_state.py         训练计划与检查点校验
      inference_core.py         推理项加载与指标计算
      inference_state.py        可恢复分片推理状态管理
    fusion.py                   加权框融合（WBF）
    submission.py               提交包打包
    messages.py                 prompt 构建
    prompts/                    每模型 prompt 文本文件
tools/                          CLI 入口
  train.py                      训练
  infer.py                      推理与评测
  fusion.py                     多模型融合
  submission.py                 提交包生成
  prepare_split.py              train/val 划分与跨集去重
  package_approved.py           数据集封包与指纹签章
  annotator_server.py           标注服务器启动
  make_manifest.py              标注清单生成
  setup_cuda.sh                 CUDA 环境一键安装
configs/                        A10 24GB YAML 配置（训练 + 推理 × 3 模型）
tests/                          218 项 CPU 单元测试
docs/                           数据契约
```

## 硬件与环境

**测试环境**：NVIDIA A10 24GB, Ubuntu, CUDA 13.0, Python 3.12。

依赖在 `pyproject.toml` 声明，torch / torchvision 精确 pin。

```bash
# CUDA 环境一键安装（按驱动版本选 cu130 / cu128 轮子）
bash tools/setup_cuda.sh

# 或手动安装
pip install -e .             # 运行依赖
pip install -e ".[dev]"      # 开发：ruff + modelscope
```

核心依赖版本：

| 包 | 版本 |
| --- | --- |
| torch | 2.14.0 |
| torchvision | 0.29.0 |
| transformers | 5.17.0 |
| peft | 0.21.0 |
| accelerate | 1.15.0 |
| flash-linear-attention | 0.5.2 |

## 数据来源与准备

### 数据来源

训练图像来自公开数据集 [RGBDT500](https://xuefeng-zhu5.github.io/RGBDT500/)
（research-only 许可）。本仓库不重新分发原始数据。

下载链接（Google Drive）：<https://drive.google.com/drive/folders/1UAe_maNR_ukYgtBrmeqiv87WhW28yK9r>

### 目录布局

```text
data/
  Raw/                         原始训练数据（400 序列）
    <seq>/
      color/                   可见光图像
      infrared/                红外图像
      depth/                   深度图像
      groundtruth.txt          真值框
  Test/                        官方测试集
    Images/
      visible/
      infrared/
      depth/
    queries/
      queries.json             官方查询模板
```

### 预处理流程

1. **划分与去重**：解析 `groundtruth.txt`，按序列级 8:2 划分 train / val，
   SHA-256 去重与测试集图像重叠的训练样本。

```bash
python tools/prepare_split.py --raw-root data --out-dir data/indexes \
  --test-images-dir data/Test/Images/visible
```

2. **标注补齐**：原始数据不含 query 文本。标注服务器提供浏览器界面，
   用于为每个样本补写 query 并校验 bbox。

```bash
# 启动标注服务器（--split 加载 outputs/annotations/{split}.json）
python tools/annotator_server.py --split train --port 8788

# 或对外部预标注清单进行校验
python tools/annotator_server.py --manifest path/to/manifest.json \
  --data-root data/Test --port 8790
```

3. **封包**：将标注结果封装为带 4 重 SHA-256 指纹的 `approved.json`，
   供训练引擎消费。

```bash
python tools/package_approved.py --dataset outputs/annotations/train.json --split train
python tools/package_approved.py --dataset outputs/annotations/val.json --split val
```

数据格式详见 [`docs/data-contract.md`](docs/data-contract.md)。

## 训练

参数优先级：CLI > YAML（`--config`）> 适配器默认值。
每个模型各有一份 A10 24GB 配置：`configs/train_{qwen3vl,qwen3_5,glm46v}.yaml`。

```bash
# 冒烟测试：单步前向 + 反向
python tools/train.py --config configs/train_qwen3vl.yaml --smoke-test

# 完整训练
python tools/train.py --config configs/train_qwen3vl.yaml --run-tag qwen3vl-r1
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--batch-size` | 1 | micro-batch |
| `--gradient-accumulation-steps` | 16 | 等效批大小 = batch-size × 本值 |
| `--learning-rate` | 1e-4 | |
| `--epochs` | 3 | |
| `--best-metric` | `acc_at_0_5` | `acc_at_0_5` / `mean_iou` / `val_loss` |
| `--lora-rank` | 16 | |
| `--lora-alpha` | 32 | |
| `--warmup-ratio` | 0.05 | 线性 warmup 占总步数比例 |
| `--lr-scheduler-type` | `cosine` | `cosine` / `linear` / `constant` |
| `--max-pixels` | 3072×28×28 | 单帧视觉 token 预算 |
| `--gradient-checkpointing` | 开 | 省激活显存，慢约 20-30% |

产物路径：`outputs/training/<run_id>/best/<best_epoch>`。

## 推理与评测

```bash
# 验证集评测（带真值，打印 ACC@0.5 / mIoU）
python tools/infer.py --config configs/infer_qwen3vl.yaml

# 测试集推理（无真值，只写 predictions.json）
python tools/infer.py --model qwen3vl --model-path models/Qwen3-VL-8B-Instruct \
  --lora-path outputs/training/<run_id>/best/<best_epoch> \
  --test-json data/Test/queries/queries.json \
  --data-dir data --batch-size 2 --run-tag test-full
```

`--resume`（默认开启）支持中断续跑。`--limit N` 用于小样本试跑。

## 融合与提交

推理只产出 `predictions.json`。融合与打包为显式独立步骤。

```bash
# 加权框融合
python tools/fusion.py \
  --predictions outputs/inference/<run_a>/predictions.json \
                outputs/inference/<run_b>/predictions.json \
  --weights 1.0 1.0 --iou-threshold 0.55 \
  --output-dir outputs/fusion

# 打包提交（校验 ID 与 bbox 合法性）
python tools/submission.py \
  --test-json data/Test/queries/queries.json \
  --predictions outputs/fusion/<run_id>/predictions.json \
  --output-dir outputs/submission/<tag>
```

## 测试

```bash
python -m unittest discover -s tests     # 218 项，纯 CPU
ruff check .                             # 静态检查
```

覆盖：坐标解析、契约指纹、训练/推理状态机、断点恢复、融合与提交包、
标注存储与会话、mock 端到端链路。不覆盖：真实权重加载、生成质量、GPU 步时与显存。

## 准备权重

权重需自行下载到本地目录，用 `--model-path` 指定。训练与推理不自动下载。

| `--model` | 权重仓库 | revision |
| --- | --- | --- |
| `qwen3vl` | `Qwen/Qwen3-VL-8B-Instruct` | `5d854aab08710c16b980ec6d603d863b3821b915` |
| `qwen3_5` | `Qwen/Qwen3.5-9B` | `460979c3d11864dd16408d860ac930a360a2fac2` |
| `glm46v` | `zai-org/GLM-4.6V-Flash` | `a4ec61fcdfab32bbccdf26c5ca8cb5a437b7ca41` |

```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
  --revision 5d854aab08710c16b980ec6d603d863b3821b915 \
  --local-dir models/Qwen3-VL-8B-Instruct
```

## 限制

- 不含数据、权重与标注产物；无标注产物时训练无法启动。
- 训练与推理需 CUDA GPU；CPU 只能跑测试与 `mock` 链路。
- `qwen3_5` 的 GDN 线性注意力内核（`flash-linear-attention`）已随默认依赖安装；
  缺失时 transformers 回退到纯 torch 实现。

## 数据来源与许可

训练图像来自 [RGBDT500](https://xuefeng-zhu5.github.io/RGBDT500/)
（research-only 许可）。本仓库不重新分发原始数据，仅提供代码。
仓库代码以 MIT 许可发布。

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
