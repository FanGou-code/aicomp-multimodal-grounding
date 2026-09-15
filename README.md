# RGBDT-Grounding

RGB-D-T（可见光 / 热红外 / 深度）三模态目标视觉定位系统。输入同一场景的
可见光、红外、深度三张已对齐图像与一句英文目标描述（query），输出目标在可见光
图像中的归一化边界框 `[x1, y1, x2, y2]`。唯一评测指标 `ACC@0.5`：预测框与真值框
IoU ≥ 0.5 记为命中，命中数 / 查询总数。

支持 4 款开源视觉大模型（LoRA 微调）与 1 个零样本检测基线，统一 adapter 协议。
训练与推理核心平台无关，本地单机与 GPU 服务器共用同一套编排。

## 模型矩阵

| 名称 | 底座 | 规模 | 输入 | 坐标协议 | LoRA 可训练参数 | 能力 |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3vl` | `Qwen/Qwen3-VL-8B-Instruct` | 8B | RGB + 红外 + 深度 | `<|box_start|>(x1,y1),(x2,y2)<|box_end|>`，整数 0–1000 | 43,646,976 | 训练 / 推理 |
| `qwen3_5` | `Qwen/Qwen3.5-9B` | 9B | RGB + 红外 + 深度 | 同上 | 29,097,984 | 训练 / 推理 |
| `mimo_vl` | `XiaomiMiMo/MiMo-VL-7B-RL` | 7B | RGB + 红外 + 深度 | JSON `[{"bbox_2d": [x1,y1,x2,y2]}]`，像素坐标按帧尺寸归一 | 41,435,136 | 训练 / 推理 |
| `glm46v` | `zai-org/GLM-4.6V-Flash` | 底座 10,292,777,472 参数 | RGB + 红外 + 深度 | `|begin_of_box|x1,y1,x2,y2|end_of_box|`，整数 0–1000 | 27,443,200 | 训练 / 推理 |
| `groundingdino` | `IDEA-Research/grounding-dino-base` | Swin-B + BERT 文本编码器 | 仅 RGB | 归一化 XYXY + 原生置信度 | 不适用（零样本，无 LoRA） | 仅推理 |
| `mock` | — | — | 三图接口（不读像素） | 归一化 XYXY | — | CPU 契约测试 |

- LoRA 目标层一律由 `models/base.py` 的 `language_model_lora_targets()` 构造，
  锚定 `model.language_model` 的正则；视觉塔恒为冻结，只做前向。`glm46v` 的 MLP
  将 gate 与 up 融合为 `gate_up_proj`，只声明 `q/k/v/o/down_proj`。
- 底座权重按 commit id 钉死（`MODEL_REVISION`），执行阶段只读本地 `--model-path`，
  不自动下载。核对值见 `tests/test_models.py`。
- 表中的 LoRA 可训练参数量由「适配器声明的投影集合」与「语言模型隐层尺寸」决定，
  可在拿到对应 `config.json` 的机器上按 meta device 复算；训练启动时打印的
  `trainable params: <N>` 是同一口径的运行时校验值。

## 核心技术设计

- **LoRA 范围锚定语言模型**：PEFT 对字符串 `target_modules` 走 `re.fullmatch`，
  因此用 `model\.language_model\..*\.(?:q_proj|…)$` 这类正则表达「只训练语言模型」。
  裸后缀名单会按后缀命中复用同名投影的视觉塔（Qwen2.5-VL 视觉 MLP 的
  `gate_proj`/`up_proj`/`down_proj`），使视觉编码器进入反向与梯度检查点重算。
- **视觉塔梯度中和**：`gradient_checkpointing_enable()` 会给所有子模块的输入
  嵌入挂钩子，PEFT 再挂一次；训练前只中和非语言子树（摘钩子 + 关该子树检查点），
  语言侧检查点保留，LoRA 梯度与 loss 不变。
- **单张量形状复用**：验证损失的批大小与训练 micro-batch 保持一致，避免第二种张量
  形状在 AMD ROCm / MIOpen 上触发第二轮算子 JIT 编译（现场记录：把验证批大小由 1
  改为 4 后，`glm46v` 冒烟卡在该前向约 21 分钟）。
- **运行身份与断点续跑**：训练与推理的 run id 由模型 revision、提示词哈希、数据
  指纹、超参与参数共同哈希得出；超参改动自动分流到新 run 目录，不覆盖既有产物。
  推理支持多分片、分片断点与已校验结果的续跑，冲突结果拒绝合并。
- **坐标防护**：三模态大模型的解析器只接受各自协议的 0–1000 或像素坐标，越界即
  判为失败（不裁剪）；GroundingDINO 的浮点回归框先裁剪到 `[0,1]` 再校验。提交包
  保留官方模板除 `bbox` 外的全部字段。
- **WBF 融合**：以各模型的 `predictions.json` 为输入（`{query_id: bbox|None}`），
  按贪心 IoU 聚类做加权坐标平均，支持用检测器原生置信度做乘性加权。

## 环境安装

Python 3.12；依赖唯一声明源是根 `pyproject.toml`（`dependencies` + extras）。

```bash
pip install -e .             # 运行依赖（torch 声明为范围，平台已有版本自动跳过）
pip install -e ".[dev]"      # 本地开发：ruff + modelscope
pip install -e ".[kernels]"  # 仅混合线性注意力架构（qwen3_5）需要
```

AMD ROCm（MI300X）环境在训练/推理前 source 性能环境文件：

```bash
source offline/rocm_env.sh   # hipBLASLt、硬件队列上限、Triton/pip 缓存目录
```

## 数据准备（Data Preparation）

原始数据集是三模态视频帧 + `groundtruth.txt`。主仓的 `scripts/prepare_rgbdt.py`
负责把原始数据整理成模型与标注流水线都能直接读取的基础数据：

| 步骤 | 产物 | 说明 |
| --- | --- | --- |
| 三模态校验 | — | 逐帧检查 `color/`（RGB）、`infrared/`（热红外）、`depth/`（16 位深度）三路存在且尺寸对齐；解析 `groundtruth.txt`，把像素框归一化为 0–1 的 XYXY，越界或退化框计为无效并剔除 |
| 深度伪彩 | `Processed/Train/<seq>/depth_jet/`、`Processed/Test/depth_jet/` | 16 位毫米深度按固定标定（默认 `--min-depth-mm 300`、`--max-depth-mm 20000`）映射为 JET 伪彩三通道图，作为三模态中的深度输入 |
| Test 引用校验 | `split_manifest.json` 的 `test_contract`（需 `--generate-indexes`） | 校验官方 Test 模板与 `Processed/Test/depth_jet` 的逐条路径映射、深度文件集合指纹，不符即中止 |

```bash
python scripts/prepare_rgbdt.py --dataset-root data            # 三模态校验 + 深度伪彩
python scripts/prepare_rgbdt.py --dataset-root data --dry-run   # 仅校验，不写文件
```

本仓的 train/val 索引生成已移交副仓（`--generate-indexes` 默认关闭，且不做测试集
重叠过滤）；训练只读 `approved.json`，不读索引。

产出的基础数据（RGB / 红外 / Depth-Jet 三路对齐图 + 归一化框）**直接供给上游数据
工程流水线**：[query-foundry](https://github.com/FanGou-code/query-foundry) 在此之上
完成训练集与测试集同帧 SHA-256 去重（防污染）、镜头序列级隔离划分（防时序穿越）、
三模态事实普查与指向性文本质检，最后按 4 重 SHA-256 指纹契约封包为 `approved.json`
回交本仓。

两仓的接口就是这份自包含产物（`protocol_version=12`）：本仓训练与推理只消费
`outputs/annotations/<run_id>/{train,val}/approved.json`，不导入副仓代码。

## 数据布局

```text
data/
  Train/<sequence>/          color/  infrared/  depth/  groundtruth.txt
  Test/                      Images/{visible,infrared,depth}/ + queries/queries.json
  Processed/                 Train/<sequence>/depth_jet/ + Test/depth_jet/（JET 伪彩深度）
outputs/
  annotations/<run_id>/      train/approved.json  val/approved.json   ← 训练唯一输入
  output_lora/<run_id>/      训练记录与 LoRA（best/  last/  checkpoints/）
  inference/<run_id>/        预测与 checkpoint
```

标注产物协议 version 12：每个样本含 `visible`/`infrared`/`depth` 路径、`query`、
归一化 `bbox` 与记录尺寸。数据与产物不入库（`.gitignore` 忽略 `data/` 与 `outputs/`）。
字段级合同与坐标约束见 `docs/data-contract.md`。

## 快速开始

### 1. 微调训练

先跑冒烟（单步前向+反向，不起 DataLoader worker），再启动完整训练：

```bash
python offline/train.py \
  --annotation-run-id <run_id> \
  --model qwen3vl \
  --model-path /path/to/Qwen3-VL-8B-Instruct \
  --data-dir data \
  --batch-size 1 --eval-batch-size 1 --smoke-test

python offline/train.py \
  --annotation-run-id <run_id> \
  --model qwen3vl \
  --model-path /path/to/Qwen3-VL-8B-Instruct \
  --data-dir data \
  --batch-size 1 --gradient-accumulation-steps 16 --learning-rate 1e-4 --epochs 3 \
  --eval-batch-size 1 --num-workers 4 --checkpoint-interval 20 \
  --run-tag qwen3vl-r1
```

换 `--model mimo_vl` 与对应 `--model-path` 即可训练其他底座；四个可训练模型的
超参默认值相同，均可用 CLI 覆盖，覆盖值计入 run 身份。GPU 实操（下载命令、
逐模型参数、平台注意事项）见 `docs/sop.md`。

### 2. 推理与 ACC@0.5 评测

```bash
# 验证集评测（带真值，直接打印 ACC@0.5 / 平均 IoU / 解析失败数）
python offline/infer.py \
  --model qwen3vl --model-path /path/to/Qwen3-VL-8B-Instruct \
  --lora-path outputs/output_lora/<train_run_id>/best/epoch_03 \
  --test-json outputs/annotations/<run_id>/val/approved.json \
  --annotation-run-id <run_id> --data-dir data \
  --num-shards 1 --num-workers 2 --batch-size 2 --batch-save 100 --run-tag val-eval

# 测试集推理（无真值，完成后自动打包 submission.zip）
python offline/infer.py \
  --model qwen3vl --model-path /path/to/Qwen3-VL-8B-Instruct \
  --lora-path outputs/output_lora/<train_run_id>/best/epoch_03 \
  --test-json data/Test/queries/queries.json --data-dir data \
  --num-shards 1 --num-workers 2 --batch-size 2 --batch-save 100 --run-tag test-full
```

`--num-shards N` 在本机多卡上拆分推理，每个分片带独立 checkpoint，中断后加
`--resume`（默认开启）继续。GroundingDINO 为单图零样本，不传 `--lora-path`。

### 3. 多模型融合（WBF）

```bash
python -m aicomp_grounding.fusion.wbf \
  --predictions outputs/inference/<a>/predictions.json outputs/inference/<b>/predictions.json \
  --weights 1.0 1.0 --iou-threshold 0.55 \
  --test-json data/Test/queries/queries.json \
  --output-dir outputs/fusion
```

`--scores` 可传各模型的分数文件（检测器用原生置信度），空字符串表示该模型不计分数。

### 4. 提交包

```bash
python -m aicomp_grounding.submission \
  --test-json data/Test/queries/queries.json \
  --predictions outputs/inference/<run_id>/predictions.json \
  --output-dir outputs/submission/<run_id>
```

### 5. 数据检查与索引

```bash
python -m unittest discover -s tests                      # 全量单元测试（纯 CPU）
python -m compileall aicomp_grounding scripts offline     # 语法检查
```

## 目录结构

```text
aicomp_grounding/        核心库：图像/坐标/合同校验、运行身份、训练核心、
                         model adapter、融合与提交
aicomp_grounding/models/ 各模型 adapter（qwen3vl / qwen3_5 / mimo_vl /
                         glm46v / groundingdino / mock）
aicomp_grounding/fusion/ 加权框融合（WBF）
offline/                 训练与推理命令行入口、ROCm 环境脚本
scripts/                 数据预处理（深度伪彩）与数据集上传工具
tests/                   单元测试（torch-free，含端到端 mock 链路）
docs/                    架构、数据合同、GPU 实操、赛题说明
```

## 文档

| 文档 | 内容 |
| --- | --- |
| `docs/architecture.md` | 模块边界、依赖方向、adapter 约定与身份/恢复规则 |
| `docs/data-contract.md` | `data/` 布局、标注产物协议、提交格式 |
| `docs/sop.md` | GPU 实操：环境、模型下载、逐模型训练与推理命令 |
| `docs/research.md` | 赛题背景与官方评测口径 |
| [query-foundry](https://github.com/FanGou-code/query-foundry) | 上游数据工程仓：跨集去重、序列级划分、三模态事实普查、人审质检与 `approved.json` 契约封包 |

## 数据来源与许可

训练图像来自公开数据集 [RGBDT500](https://xuefeng-zhu5.github.io/RGBDT500/)
（research-only 许可），本仓库不重新分发原始数据，仅提供预处理与训练/推理代码。

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

本仓库自身代码以 MIT 许可发布，见 `LICENSE`。
