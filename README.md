# RGBDT-Grounding

离线、平台无关的 RGB-D-T（可见光 / 热红外 / 深度）三模态目标视觉定位实验单元。

输入同一场景已对齐的三张图像与一句英文目标描述（query），输出目标在可见光图像中的
归一化边界框 `[x1, y1, x2, y2]`。唯一评测指标 `ACC@0.5`：与真值框 IoU ≥ 0.5 记为
命中，命中数 / 查询总数。支持 4 款开源视觉大模型（LoRA 微调）与 1 个零样本检测基线，
统一 adapter 协议，用于在一致口径下做底座对比实验。

**运行契约**：所有模型加载走 `local_files_only=True`，训练与推理入口在运行期不访问
网络、不自动下载。权重、数据与标注产物全部由调用者以本地目录提供，可复现性不依赖
任何平台或镜像。

**本仓库不包含**：数据集与图像、模型权重、标注产物、运行结果与榜单成绩。

## 模型支持

| `--model` | 底座 | 模态 | 坐标协议 | 训练 |
| --- | --- | --- | --- | --- |
| `qwen3vl` | `Qwen/Qwen3-VL-8B-Instruct` | RGB + 红外 + 深度 | `<|box_start|>(x1,y1),(x2,y2)<|box_end|>`，整数 0–1000 | LoRA |
| `qwen3_5` | `Qwen/Qwen3.5-9B` | RGB + 红外 + 深度 | 同上 | LoRA |
| `mimo_vl` | `XiaomiMiMo/MiMo-VL-7B-RL` | RGB + 红外 + 深度 | JSON `[{"bbox_2d": [x1,y1,x2,y2]}]`，像素坐标按帧尺寸归一 | LoRA |
| `glm46v` | `zai-org/GLM-4.6V-Flash` | RGB + 红外 + 深度 | `|begin_of_box|x1,y1,x2,y2|end_of_box|`，整数 0–1000 | LoRA |
| `groundingdino` | `IDEA-Research/grounding-dino-base` | 仅 RGB | 归一化 XYXY + 原生置信度 | 仅推理（零样本） |
| `mock` | — | 接口占位 | 归一化 XYXY | 仅测试 |

## 安装

Python 3.12；依赖的唯一声明源是根 `pyproject.toml`。

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

底座权重需自行下载到本地目录，再用 `--model-path` 指给入口。适配器按下表记录来源
快照（`tests/test_models.py` 逐值钉死）；执行时**不校验**目录与 revision 是否对应——
目录内容即事实，revision 只用于追溯与运行身份。

| `--model` | 权重仓库 | revision |
| --- | --- | --- |
| `qwen3vl` | `Qwen/Qwen3-VL-8B-Instruct` | `5d854aab08710c16b980ec6d603d863b3821b915` |
| `qwen3_5` | `Qwen/Qwen3.5-9B` | `460979c3d11864dd16408d860ac930a360a2fac2` |
| `mimo_vl` | `XiaomiMiMo/MiMo-VL-7B-RL` | `d307865d4a3b6ad9ae35e574bcabaa563038c8fb` |
| `glm46v` | `zai-org/GLM-4.6V-Flash`（镜像 `ZhipuAI/GLM-4.6V-Flash`） | `a4ec61fcdfab32bbccdf26c5ca8cb5a437b7ca41` |
| `groundingdino` | `IDEA-Research/grounding-dino-base` | `d06985a44c66b6133c131bd273293be8649cfe3a` |

```bash
# 以 Hugging Face CLI 为例；其它镜像取到同版本目录即可
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

四个可训练模型的超参默认值相同，均可用 CLI 覆盖；覆盖值计入运行身份。

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

换 `--model`（`qwen3vl` / `qwen3_5` / `mimo_vl` / `glm46v`）与对应的 `--model-path`
即可训练其它底座。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--batch-size` | 1 | micro-batch |
| `--gradient-accumulation-steps` | 16 | 等效批大小 = `--batch-size` × 本值 |
| `--learning-rate` | 1e-4 | |
| `--epochs` | 3 | |
| `--eval-batch-size` | 1 | 与 `--batch-size` 保持一致：训练与验证共用一种张量形状 |
| `--best-metric` | `acc_at_0_5` | `acc_at_0_5` / `mean_iou` 越大越好，`val_loss` 越小越好 |
| `--max-pixels` | 3072×28×28 | 单帧视觉 token 预算；显存不足时优先下调，会改变运行身份 |

**冒烟自检**：通过时打印 `Training smoke passed` 与 `trainable params: <N>`，
`<N>` 应为 43,646,976（`qwen3vl`）/ 29,097,984（`qwen3_5`）/ 41,435,136（`mimo_vl`）/
27,443,200（`glm46v`）；数值不符通常意味着 `--model-path` 指向了别的 revision，
或 LoRA 目标层被改动。

## 推理与评测

```bash
# 验证集评测：带真值，直接打印 ACC@0.5 / 平均 IoU / 解析失败数
python offline/infer.py --model qwen3vl --model-path models/Qwen3-VL-8B-Instruct \
  --lora-path outputs/output_lora/<train_run_id>/best/epoch_03 \
  --test-json outputs/annotations/<run_id>/val/approved.json --annotation-run-id <run_id> \
  --data-dir data --num-shards 1 --num-workers 2 --batch-size 2 --batch-save 100 \
  --run-tag val-eval

# 测试集推理：无真值，跑完后自动打包 submission.zip
python offline/infer.py --model qwen3vl --model-path models/Qwen3-VL-8B-Instruct \
  --lora-path outputs/output_lora/<train_run_id>/best/epoch_03 \
  --test-json data/Test/queries/queries.json \
  --data-dir data --num-shards 1 --num-workers 2 --batch-size 2 --batch-save 100 \
  --run-tag test-full
```

加 `--limit 100` 先小样本试跑。`--num-shards N` 在本机多卡上拆分推理，每个分片独立
落 checkpoint，中断后加 `--resume`（默认开启）继续；结果冲突会被拒绝而不是静默覆盖。
`groundingdino` 为单图零样本，不传 `--lora-path`，用 `--num-workers 4 --batch-size 8`。

## 融合与提交

```bash
python -m aicomp_grounding.fusion.wbf \
  --predictions outputs/inference/<run_a>/predictions.json \
                outputs/inference/<run_b>/predictions.json \
  --weights 1.0 1.0 --iou-threshold 0.55 \
  --test-json data/Test/queries/queries.json --output-dir outputs/fusion

python -m aicomp_grounding.submission \
  --test-json data/Test/queries/queries.json \
  --predictions outputs/fusion/<run_id>/predictions.json \
  --output-dir outputs/submission/<run_id>
```

`--scores` 可传各模型的分数文件（检测器用原生置信度做乘性加权），空字符串表示该模型
不计分数；`--allow-fallback` 会把无效框填成占位框，仅用于诊断包。

## 运行身份与产物

训练与推理的产物目录名即运行身份：由模型 revision、提示词哈希、数据与图像指纹、
像素预算、超参、种子与 `--run-tag` 共同哈希得出。因此：

- 同一份输入与参数会命中同一个 run id，可安全重跑（默认 `--resume` 续跑）；
- 任何参数或数据变更都会分流到新目录，**不覆盖**既有产物；
- 训练产物在 `outputs/output_lora/<run_id>/`（`plan.json`、`checkpoints/`、`best/`、
  `last/`、`completed.json`），推理产物在 `outputs/inference/<run_id>/`。

## 扩展：新增一个模型

适配器协议在 `aicomp_grounding/models/base.py`，最小完整示例是
`models/mock.py`（约 60 行）。需要实现：

| 类别 | 方法 |
| --- | --- |
| 标识 | `name`、`model_name`、`model_revision`、`supports_lora`、`prompt_hash()`、`identity()` |
| 推理 | `load()`、`predict()`；可选 `prepare_inputs()` + `predict_from_inputs()`（批量预处理） |
| 训练 | `training_hyperparameters()`、`lora_target_modules()`、`load_for_training()`、`build_training_batch()`、`collate_training_batch()`、`build_grounding_batch()`、`decode_grounding_outputs()`、`parse_grounding_text()` |

新增后需同步：注册表 `models/__init__.py`、可训练白名单
`training_core.TRAINABLE_MODELS`（若支持训练）、像素预算判定
`offline/infer.py:PIXEL_BUDGET_MODELS`（若构造器接收 `max_pixels`），以及
`tests/test_models.py` 的 revision 表、投影名单与 LoRA 锚定断言。LoRA 目标层必须经
`language_model_lora_targets()` 构造，不要用裸后缀名单。

## 测试

```bash
python -m unittest discover -s tests                      # 约 200 项，纯 CPU
python -m compileall aicomp_grounding scripts offline      # 语法检查
```

覆盖：坐标与解析、产物合同与指纹、训练/推理身份与状态机、断点恢复、融合与提交包、
mock 端到端链路。**不覆盖**：真实权重加载、生成质量、步时与显存——这些需在 GPU 上
用冒烟与真实评测验证。

## 目录结构

```text
aicomp_grounding/          核心库：坐标与合同、运行身份、训练核心、推理状态、融合与提交
aicomp_grounding/models/   各底座适配器（qwen3vl / qwen3_5 / mimo_vl / glm46v /
                           groundingdino / mock）
aicomp_grounding/fusion/   加权框融合（WBF）
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
- 三模态输入须已时间同步与空间对齐；本单元不做配准与同步。
- 本单元不校验 `--model-path` 目录与声明 revision 的对应关系，也不校验图像字节版本。
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
