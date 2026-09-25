# RGB-Grounding

可见光（RGB）目标视觉定位的完整代码：**数据生产（标注流水线）与模型微调评测**
两条产线，合于一个仓库。视觉大模型只消费可见光图像；深度与红外不进模型输入，
只作为序数后处理的物理量（原始 16 位毫米深度切片、红外强度切片）。

- **标注侧**（`aicomp_grounding/annotation/` + `tools/`）：把原始图像加工成自包含的
  `approved.json`——划分、跨集去重、事实普查、教师组句、人审、封包与 4 重指纹自校验。
- **服务侧**（`aicomp_grounding/serving/` + `tools/`）：三款视觉大模型的 LoRA 微调、
  推理评测、序数后处理、融合与提交。

输入一张可见光图像与一句英文目标描述（query），输出目标在图像中的
归一化边界框 `[x1, y1, x2, y2]`。指标为 `ACC@0.5`：与真值框 IoU ≥ 0.5 记为命中。

仓库只提供代码，不含数据集、模型权重与标注产物。

## 模型支持

| `--model` | 底座 | 输入 | 训练 |
| --- | --- | --- | --- |
| `qwen3vl` | `Qwen/Qwen3-VL-8B-Instruct` | 可见光 RGB | LoRA |
| `qwen3_5` | `Qwen/Qwen3.5-9B` | 可见光 RGB | LoRA |
| `glm46v` | `zai-org/GLM-4.6V-Flash` | 可见光 RGB | LoRA |
| `mock` | — | 接口占位 | 仅测试 |

## 安装

Python 3.12；依赖在根 `pyproject.toml` 声明，torch/torchvision 精确 pin。

CUDA 环境一键安装——按驱动报告的 CUDA 版本选择 cu130 / cu128 轮子，装完做健康自检。
托管镜像已预装匹配的 torch 时（如 CUDA 13.0.3 / Python 3.12 / torch 2.13.0 基础镜像），
该步由 pip 判定已满足并跳过：

```bash
bash tools/setup_cuda.sh
```

其它环境先按平台 CUDA 版本装好 torch/torchvision 轮子，再装本仓库：

```bash
pip install -e .             # 其余运行依赖
pip install -e ".[kernels]"  # 仅 qwen3_5 需要：GDN 线性注意力加速内核
pip install -e ".[dev]"      # 开发用：ruff
```

## 数据生产（标注侧）

标注流水线把原始数据加工成 `approved.json`，阶段入口都在 `tools/`：

```text
prepare_split.py    划分 + 跨集去重 → data/indexes/{train,val}.json + split_manifest.json
run_census.py       普查：每帧一次枚举 + 属性报告
run_generation.py   生成：教师为每个目标写一句
review_server.py    人审（浏览器）
apply_review.py     合并烘焙
package_approved.py 封包 + 4 重指纹自校验 → outputs/annotations/<run_id>/{train,val}/approved.json
```

每轮手动注入一把 key（`--api-key` 或环境变量 `ANNOTATION_API_KEY`），不轮换、不落盘。
三个 API 阶段默认断点续跑与错误有限退避重试。

## 准备权重

权重需自行下载到本地目录，用 `--model-path` 指给入口；训练与推理不下载文件。下表是
各适配器记录的来源快照（`tests/test_models.py` 逐值钉死）；执行时不校验目录与
revision 是否对应，revision 只用于追溯与运行身份。

| `--model` | 权重仓库 | revision |
| --- | --- | --- |
| `qwen3vl` | `Qwen/Qwen3-VL-8B-Instruct` | `5d854aab08710c16b980ec6d603d863b3821b915` |
| `qwen3_5` | `Qwen/Qwen3.5-9B` | `460979c3d11864dd16408d860ac930a360a2fac2` |
| `glm46v` | `zai-org/GLM-4.6V-Flash`（镜像 `ZhipuAI/GLM-4.6V-Flash`） | `a4ec61fcdfab32bbccdf26c5ca8cb5a437b7ca41` |

```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
  --revision 5d854aab08710c16b980ec6d603d863b3821b915 \
  --local-dir models/Qwen3-VL-8B-Instruct
```

## 准备数据

图像按 `docs/data-contract.md` 的布局放入 `data/`（`Train/<seq>/{color,infrared,depth}`、
`Test/Images/{visible,infrared,depth}`）。视觉大模型只读可见光（`color` / `visible`）；
原始 16 位深度图与红外图由序数后处理消费，不进模型输入。

`tools/prepare_split.py` 解析 `groundtruth.txt`、归一化真值框、按序列划分 train/val、
跨集去重并落盘索引；训练只消费 `outputs/annotations/<run_id>/{train,val}/approved.json`
（协议 12）。

```bash
python tools/prepare_split.py --raw-root data --out-dir data/indexes \
  --test-images-dir data/Test/Images/visible
```

## 训练

三个可训练模型的超参默认值相同。参数优先级：CLI > YAML（`--config`）> 适配器默认值；
覆盖值计入运行身份。每个模型各有一份 A10 24GB 配置：
`configs/train_{qwen3vl,qwen3_5,glm46v}.yaml`。

```bash
# 冒烟：单步前向 + 反向，只跑 1 个 micro-batch
python tools/train.py --annotation-run-id <run_id> --model qwen3vl \
  --model-path models/Qwen3-VL-8B-Instruct --data-dir data \
  --batch-size 1 --eval-batch-size 1 --checkpoint-interval 20 --smoke-test

# 完整训练：用对应模型的配置，CLI 可覆盖单项（这里换 run-tag）
python tools/train.py --config configs/train_qwen3vl.yaml --run-tag qwen3vl-r2
```

训练其它模型时替换 `--model`（`qwen3vl` / `qwen3_5` / `glm46v`）与 `--model-path`。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--batch-size` | 1 | micro-batch |
| `--gradient-accumulation-steps` | 16 | 等效批大小 = `--batch-size` × 本值 |
| `--learning-rate` | 1e-4 | |
| `--epochs` | 3 | |
| `--eval-batch-size` | 1 | 与 `--batch-size` 保持一致：训练与验证共用一种张量形状 |
| `--best-metric` | `acc_at_0_5` | `acc_at_0_5` / `mean_iou` 越大越好，`val_loss` 越小越好 |
| `--lora-rank` | 16 | LoRA 秩 |
| `--lora-alpha` | 32 | LoRA 缩放 |
| `--lora-dropout` | 0.05 | LoRA dropout（0–1） |
| `--warmup-ratio` | 0.05 | 线性 warmup 占总步数比例（0–1） |
| `--lr-scheduler-type` | `cosine` | `cosine` / `linear` / `constant`；warmup 后衰减至峰值学习率的 10% |
| `--max-grad-norm` | 1.0 | 梯度裁剪范数 |
| `--weight-decay` | 0.01 | AdamW 权重衰减 |
| `--max-pixels` | 3072×28×28 | 单帧视觉 token 预算；显存不足时下调（须大于 min_pixels），会改变运行身份 |
| `--gradient-checkpointing` | 开 | 梯度检查点：省激活显存、慢约 20–30%；显存充足时用 `--no-gradient-checkpointing` 关闭 |

## 推理与评测

参数优先级：CLI > YAML（`--config`）> 默认值。每个模型各有一份 A10 24GB 配置
（`configs/infer_{qwen3vl,qwen3_5,glm46v}.yaml`，`lora_path` 指向对应训练 run 的最佳 epoch）。

```bash
# 验证集评测：带真值，直接打印 ACC@0.5 / 平均 IoU / 解析失败数
python tools/infer.py --model qwen3vl --model-path models/Qwen3-VL-8B-Instruct \
  --lora-path outputs/training/<train_run_id>/best/epoch_03 \
  --test-json outputs/annotations/<run_id>/val/approved.json --annotation-run-id <run_id> \
  --data-dir data --num-shards 1 --num-workers 2 --batch-size 2 --batch-save 100 \
  --run-tag val-eval

# 测试集推理：无真值，只写 predictions.json，不打包
python tools/infer.py --model qwen3vl --model-path models/Qwen3-VL-8B-Instruct \
  --lora-path outputs/training/<train_run_id>/best/epoch_03 \
  --test-json data/Test/queries/queries.json \
  --data-dir data --num-shards 1 --num-workers 2 --batch-size 2 --batch-save 100 \
  --run-tag test-full

# 或用 YAML 配置整体运行
python tools/infer.py --config configs/infer_qwen3vl.yaml
```

`--limit 100` 用于小样本试跑。中断后用 `--resume`（默认开启）续跑。

## 融合与提交

融合与打包都是显式步骤：推理只产出 `predictions.json`，融合只产出融合后的
`predictions.json`，提交包单独生成。训练与推理入口都不会自动打包。

```bash
python tools/fusion.py \
  --predictions outputs/inference/<run_a>/predictions.json \
                outputs/inference/<run_b>/predictions.json \
  --weights 1.0 1.0 --iou-threshold 0.55 \
  --output-dir outputs/fusion

python tools/submission.py \
  --test-json data/Test/queries/queries.json \
  --predictions outputs/fusion/<run_id>/predictions.json \
  --output-dir outputs/submission/<tag>
```

`submission` 默认要求预测 ID 与官方模板完全一致、bbox 全部合法，否则拒绝出包；
`--allow-fallback` 会把缺失或无效的框填成占位框，仅用于显式不完整的诊断包。

## 序数后处理（可选，显式步骤）

推理之后、融合之前可插入一步序数后处理：解析 query 意图，枚举同类别的全部实例，再按
坐标轴取第 k 个覆盖该模型的框。它只处理"在多个同类实例里按序选一个"这一类题，其余题
逐字节不改。由 `qwen3_5` 走完整套，产出它那一份修正后的预测，随后进 WBF。

```bash
# 解析（纯文本、贪心）→ 对排序题枚举一次（基座权重 + 仅可见光 + 开思考）
python tools/ordinal_enumerate.py \
  --model qwen3_5 --model-path models/Qwen3.5-9B \
  --test-json data/Test/queries/queries.json --data-dir data \
  --temperature 0.6 --top-p 0.95 --top-k 20 --presence-penalty 0.0 \
  --output-dir outputs/ordinal_enum --run-tag ord-full

# 用枚举清单修正 qwen3_5 的预测
python tools/ordinal_resolve.py \
  --enum-run outputs/ordinal_enum/<id> --predictions outputs/inference/<infer>/predictions.json \
  --test-json data/Test/queries/queries.json --data-dir data \
  --output-dir outputs/ordinal_resolve --run-tag ord-full
```

参数出处与正向/逆向共用的中间表示见 `docs/ordinal-contract.md`。

## 运行身份与产物

产物目录名即运行身份，由模型 revision、提示词哈希、数据与图像指纹、像素预算、超参、
种子与 `--run-tag` 共同哈希得出：

- 同一份输入与参数命中同一个 run id（默认 `--resume` 续跑）；
- 参数或数据变更分流到新目录，不覆盖既有产物；
- 训练产物在 `outputs/training/<run_id>/`，推理产物在 `outputs/inference/<run_id>/`。

## 扩展：新增一个模型

适配器协议在 `aicomp_grounding/serving/models/base.py`，最小完整示例是
`serving/models/mock.py`。需要实现：

| 类别 | 方法 |
| --- | --- |
| 标识 | `name`、`model_name`、`model_revision`、`supports_lora`、`prompt_hash()`、`identity()` |
| 推理 | `load()`、`predict()`；可选 `prepare_inputs()` + `predict_from_inputs()`；`generate_messages()`（调用方自备消息、返回原始文本） |
| 训练 | `training_hyperparameters()`、`lora_target_modules()`、`load_for_training()`、`build_training_batch()`、`collate_training_batch()`、`build_grounding_batch()`、`decode_grounding_outputs()`、`parse_grounding_text()` |

新增后需同步：注册表 `serving/models/__init__.py`、可训练白名单
`serving/engine/training_core.TRAINABLE_MODELS`（若支持训练）、像素预算判定
`tools/infer.py:PIXEL_BUDGET_MODELS`，以及 `tests/test_models.py` 的 revision 表、
投影名单与 LoRA 锚定断言。LoRA 目标层必须经 `language_model_lora_targets()` 构造。

## 测试

```bash
python -m unittest discover -s tests            # 434 项，纯 CPU
python -m compileall aicomp_grounding tools     # 语法检查
ruff check .                                    # 静态检查
```

覆盖坐标与解析、产物合同与指纹、训练/推理身份与状态机、断点恢复、融合与提交包、
标注流水线（划分/普查/生成/人审/封包）、mock 端到端链路；不覆盖真实权重加载、
生成质量、步时与显存（需 GPU）。

## 目录结构

```text
aicomp_grounding/            核心库
  bbox.py io.py artifacts.py contract.py query.py images.py sharding.py
  ordinal_kernel.py paths.py config.py testset.py    共享底座
  annotation/               标注侧：普查 / 生成 / 人审 / 封包 / 逆向通路
  serving/                  服务侧：models / engine / fusion / ordinal / submission / messages
tools/                      全部 CLI 入口
tests/                      CPU 单元测试
docs/                      架构、数据合同、序数契约、赛题说明
```

## 文档

| 文档 | 内容 |
| --- | --- |
| `docs/architecture.md` | 分层、模块地图、结构不变量、契约边界 |
| `docs/data-contract.md` | `data/` 布局、模态格式、`approved.json` 字段表、校验点、提交格式 |
| `docs/ordinal-contract.md` | 序数后处理的中间表示、解码参数表、门、run-id 语义 |
| `docs/research.md` | 赛题背景与官方评测口径 |

## 限制

- 不含数据、权重、标注产物与结果；无标注产物时训练无法启动。
- 训练与推理需自备 CUDA GPU；CPU 只能跑契约测试与 `mock` 链路。
- `qwen3_5` 依赖 GDN 加速内核（`pip install -e ".[kernels]"`），未安装时静默回退到
  较慢的 torch 实现。
- 序数后处理依赖深度/红外与可见光的空间对齐；本仓库不做配准与同步。
- 本仓库不校验 `--model-path` 目录与声明 revision 的对应关系，也不校验图像字节版本。

## 数据来源与许可

训练图像来自公开数据集 [RGBDT500](https://xuefeng-zhu5.github.io/RGBDT500/)
（research-only 许可），本仓库不重新分发原始数据，仅提供预处理与训练/推理代码。
本仓库自身代码以 MIT 许可发布，见 `LICENSE`。

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
