# RGBDT-Grounding

RGB-D-T（可见光、热红外、深度）三模态目标视觉定位的微调与评测代码。

输入同一场景已对齐的三张图像与一句英文目标描述（query），输出目标在可见光图像中的
归一化边界框 `[x1, y1, x2, y2]`。指标为 `ACC@0.5`：与真值框 IoU ≥ 0.5 记为命中，
命中数 / 查询总数。支持三款视觉大模型的 LoRA 微调，统一 adapter
协议。

仓库只提供代码，不含数据集、模型权重与标注产物。

## 模型支持

| `--model` | 底座 | 模态 | 训练 |
| --- | --- | --- | --- |
| `qwen3vl` | `Qwen/Qwen3-VL-8B-Instruct` | RGB + 红外 + 深度 | LoRA |
| `qwen3_5` | `Qwen/Qwen3.5-9B` | RGB + 红外 + 深度 | LoRA |
| `glm46v` | `zai-org/GLM-4.6V-Flash` | RGB + 红外 + 深度 | LoRA |
| `mock` | — | 接口占位 | 仅测试 |

## 安装

Python 3.12；依赖在根 `pyproject.toml` 声明。

```bash
pip install -e .             # 运行依赖（torch 声明为范围，平台已装版本自动跳过）
pip install -e ".[kernels]"  # 仅 qwen3_5 需要：GDN 线性注意力加速内核
pip install -e ".[dev]"      # 开发用：ruff + 数据集发布工具依赖
```

AMD ROCm 环境在训练/推理前 source 一次：

```bash
source offline/rocm_env.sh   # BLAS 后端、硬件队列上限、Triton/pip 缓存目录
```

ROCm 上 `causal-conv1d` 没有预编译轮，需先源码编译再装 `[kernels]`：

```bash
export PYTORCH_ROCM_ARCH=gfx942 MAX_JOBS=8 CAUSAL_CONV1D_FORCE_BUILD=TRUE
pip install causal-conv1d --no-build-isolation
pip install -e ".[kernels]"
```

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
# 以 Hugging Face CLI 为例
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
  --revision 5d854aab08710c16b980ec6d603d863b3821b915 \
  --local-dir models/Qwen3-VL-8B-Instruct
```

## 准备数据

两件事：**三模态图像**与**标注产物**。

1. 图像按 `docs/data-contract.md` 的布局放入 `data/`；`scripts/prepare_rgbdt.py` 完成
   原始数据到可用输入的整理——逐帧校验三路模态存在且尺寸对齐、解析 `groundtruth.txt`
   并归一化真值框、把 16 位毫米深度按固定标定渲染为 JET 伪彩。
2. 训练只消费 `outputs/annotations/<run_id>/{train,val}/approved.json`（协议 12）。
   该产物由上游数据工程仓 [query-foundry](https://github.com/FanGou-code/query-foundry)
   生产：跨集去重、镜头序列级划分、三模态事实普查与文本质检、4 重 SHA-256 指纹封包。

```bash
python scripts/prepare_rgbdt.py --dataset-root data              # 校验 + 深度伪彩
python scripts/prepare_rgbdt.py --dataset-root data --dry-run     # 只校验，不写盘
```

## 训练

三个可训练模型的超参默认值相同，均可用 CLI 覆盖；覆盖值计入运行身份。

```bash
# 冒烟：单步前向 + 反向，只跑 1 个 micro-batch
python offline/train.py --annotation-run-id <run_id> --model qwen3vl \
  --model-path models/Qwen3-VL-8B-Instruct --data-dir data \
  --batch-size 1 --eval-batch-size 1 --checkpoint-interval 20 --smoke-test

# 完整训练
python offline/train.py --annotation-run-id <run_id> --model qwen3vl \
  --model-path models/Qwen3-VL-8B-Instruct --data-dir data \
  --batch-size 1 --gradient-accumulation-steps 16 --learning-rate 1e-4 --epochs 3 \
  --eval-batch-size 1 --num-workers 4 --checkpoint-interval 20 --run-tag qwen3vl-r1
```

训练其它模型时替换 `--model`（`qwen3vl` / `qwen3_5` / `glm46v`）与
`--model-path`。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--batch-size` | 1 | micro-batch |
| `--gradient-accumulation-steps` | 16 | 等效批大小 = `--batch-size` × 本值 |
| `--learning-rate` | 1e-4 | |
| `--epochs` | 3 | |
| `--eval-batch-size` | 1 | 与 `--batch-size` 保持一致：训练与验证共用一种张量形状 |
| `--best-metric` | `acc_at_0_5` | `acc_at_0_5` / `mean_iou` 越大越好，`val_loss` 越小越好 |
| `--max-pixels` | 3072×28×28 | 单帧视觉 token 预算；显存不足时优先下调，会改变运行身份 |

## 推理与评测

```bash
# 验证集评测：带真值，直接打印 ACC@0.5 / 平均 IoU / 解析失败数
python offline/infer.py --model qwen3vl --model-path models/Qwen3-VL-8B-Instruct \
  --lora-path outputs/output_lora/<train_run_id>/best/epoch_03 \
  --test-json outputs/annotations/<run_id>/val/approved.json --annotation-run-id <run_id> \
  --data-dir data --num-shards 1 --num-workers 2 --batch-size 2 --batch-save 100 \
  --run-tag val-eval

# 测试集推理：无真值，只写 predictions.json，不打包
python offline/infer.py --model qwen3vl --model-path models/Qwen3-VL-8B-Instruct \
  --lora-path outputs/output_lora/<train_run_id>/best/epoch_03 \
  --test-json data/Test/queries/queries.json \
  --data-dir data --num-shards 1 --num-workers 2 --batch-size 2 --batch-save 100 \
  --run-tag test-full
```

`--limit 100` 用于小样本试跑。中断后用 `--resume`（默认开启）续跑。

## 融合与提交

融合与打包都是显式步骤：推理只产出 `predictions.json`，融合只产出融合后的
`predictions.json`，提交包由 `submission` 单独生成。训练与推理入口都不会自动打包。

```bash
python -m aicomp_grounding.fusion.wbf \
  --predictions outputs/inference/<run_a>/predictions.json \
                outputs/inference/<run_b>/predictions.json \
  --weights 1.0 1.0 --iou-threshold 0.55 \
  --output-dir outputs/fusion

python -m aicomp_grounding.submission \
  --test-json data/Test/queries/queries.json \
  --predictions outputs/fusion/<run_id>/predictions.json \
  --output-dir outputs/submission/<run_id>
```

`--scores` 可传各模型的分数文件做乘性加权，空字符串表示该模型不计分数；当前阵容没有
模型产出原生置信度，该参数暂无使用者。`submission` 默认要求预测 ID 与官方模板完全
一致、bbox 全部合法，否则拒绝出包；`--allow-fallback` 会把缺失或无效的框填成占位框，
仅用于显式不完整的诊断包。

## 序数后处理（可选，显式步骤）

推理之后、融合之前可插入一步序数后处理：解析 query 意图，枚举同类别的全部实例，再按
坐标轴取第 k 个覆盖该模型的框。它只覆盖"在多个同类实例里按序选一个"这一类题，其余题
逐字节不改；产出每个模型各一份修正后的预测，随后照常进 WBF。

```bash
# 每个在役模型各跑一次：解析（纯文本）→ 对排序题枚举两次（基座权重 + 仅可见光）
python -m aicomp_grounding.ordinal.enumerate \
  --model qwen3_5 --model-path models/Qwen3.5-9B \
  --test-json data/Test/queries/queries.json --data-dir data \
  --max-new-tokens 1024 --temperature 0.7 \
  --output-dir outputs/enum --run-tag ord-full

# 汇总三个模型：枚举两次清单必须一一对上
python -m aicomp_grounding.ordinal.resolve \
  --enum-run outputs/enum/<id_a> --predictions outputs/inference/<infer_a>/predictions.json \
  --enum-run outputs/enum/<id_b> --predictions outputs/inference/<infer_b>/predictions.json \
  --enum-run outputs/enum/<id_c> --predictions outputs/inference/<infer_c>/predictions.json \
  --test-json data/Test/queries/queries.json --data-dir data \
  --output-dir outputs/ordinal --run-tag ord-full
```

每个模型走完整套：自己的解析 → 自己的两次枚举 → 自己的取第 k。`resolve` 汇总三个
模型，**至少两个模型都判定要换**才采纳这一轮，否则三份输出全部回退为原框。
轴限 `x` / `y` / `depth` / `area` / `ir`；数据缺失的轴不触发。`--temperature` 必须
大于 0 —— 两次枚举逐字相同就失去自证意义。

判定用的门：解析合法、未截断（自报数等于清单长度）、两次清单一一对上、`k ≤ N`、
基准框能在清单里找到。任一不过就保留原框。第 k 个若就是基准框那个物体，保留基准框
坐标（WBF 的坐标比单次枚举更准）。


## 运行身份与产物

产物目录名即运行身份，由模型 revision、提示词哈希、数据与图像指纹、像素预算、超参、
种子与 `--run-tag` 共同哈希得出：

- 同一份输入与参数命中同一个 run id（默认 `--resume` 续跑）；
- 参数或数据变更分流到新目录，不覆盖既有产物；
- 训练产物在 `outputs/output_lora/<run_id>/`（`plan.json`、`checkpoints/`、`best/`、
  `last/`、`completed.json`），推理产物在 `outputs/inference/<run_id>/`。

## 扩展：新增一个模型

适配器协议在 `aicomp_grounding/models/base.py`，最小完整示例是
`models/mock.py`（约 60 行）。需要实现：

| 类别 | 方法 |
| --- | --- |
| 标识 | `name`、`model_name`、`model_revision`、`supports_lora`、`prompt_hash()`、`identity()` |
| 推理 | `load()`、`predict()`；可选 `prepare_inputs()` + `predict_from_inputs()`（批量预处理）；`generate_messages()`（调用方自备消息、返回原始文本，序数后处理用它） |
| 训练 | `training_hyperparameters()`、`lora_target_modules()`、`load_for_training()`、`build_training_batch()`、`collate_training_batch()`、`build_grounding_batch()`、`decode_grounding_outputs()`、`parse_grounding_text()` |

新增后需同步：注册表 `models/__init__.py`、可训练白名单
`training_core.TRAINABLE_MODELS`（若支持训练）、像素预算判定
`offline/infer.py:PIXEL_BUDGET_MODELS`（若构造器接收 `max_pixels`），以及
`tests/test_models.py` 的 revision 表、投影名单与 LoRA 锚定断言。LoRA 目标层必须经
`language_model_lora_targets()` 构造，不要用裸后缀名单。

## 测试

```bash
python -m unittest discover -s tests                      # 258 项，纯 CPU
python -m compileall aicomp_grounding scripts offline      # 语法检查
```

覆盖坐标与解析、产物合同与指纹、训练/推理身份与状态机、断点恢复、融合与提交包、
mock 端到端链路；不覆盖真实权重加载、生成质量、步时与显存（需 GPU）。

## 目录结构

```text
aicomp_grounding/          核心库：坐标与合同、运行身份、训练核心、推理状态、融合与提交
aicomp_grounding/models/   各底座适配器（qwen3vl / qwen3_5 / glm46v / mock）
aicomp_grounding/fusion/   加权框融合（WBF）
aicomp_grounding/ordinal/  序数后处理（解析 / 枚举 / 取第 k）+ prompts/*.md
offline/                   训练与推理 CLI、ROCm 环境脚本
scripts/                   数据预处理与数据集发布工具
tests/                     CPU 单元测试
docs/                      架构、数据合同、赛题说明
```

## 文档

| 文档 | 内容 |
| --- | --- |
| `docs/architecture.md` | 分层、模块地图、结构不变量、契约边界 |
| `docs/data-contract.md` | `data/` 布局、三模态格式、`approved.json` 字段表、校验点、提交格式 |
| `docs/research.md` | 赛题背景与官方评测口径 |
| [query-foundry](https://github.com/FanGou-code/query-foundry) | 上游数据工程仓：跨集去重、序列级划分、三模态事实普查、人审质检与产物封包 |

## 限制

- 不含数据、权重、标注产物与结果；无标注产物时训练无法启动。
- 训练与推理需自备 GPU（CUDA 或 ROCm）；CPU 只能跑契约测试与 `mock` 链路。
- `qwen3_5` 依赖 GDN 加速内核（`pip install -e ".[kernels]"`），未安装时静默回退到
  较慢的 torch 实现。
- 三模态输入须已时间同步与空间对齐；本仓库不做配准与同步。
- 本仓库不校验 `--model-path` 目录与声明 revision 的对应关系，也不校验图像字节版本。
- 恢复已有训练记录时，产物目录必须原地保留（记录里保存的是绝对路径，不提供跨目录
  迁移）；新训练可以用不同的 `--output-root`。

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
