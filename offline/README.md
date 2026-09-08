# Offline End — 离线端

离线端提供平台无关的训练与推理入口。`offline/infer.py` 调用推理核心，
`offline/train.py` 与云端共用 `aicomp_grounding/training_core`。
本目录**不特指任何平台**（DSW / Modal / 实验室 GPU 机通用）。

数据索引与查重审计已随标注生产线迁入伴生仓 `query-foundry/data/`，
本仓不存放；训练只读 `outputs/annotations/` 下的 `approved.json`。

## 环境安装（通用）

任何 GPU 平台，先让机器上有 torch（平台镜像自带，或自装 CUDA 版），再：

```bash
pip install -e .            # 通用依赖（torch 范围 + 四件套 + numpy/pillow/opencv）
pip install -e ".[kernels]" # 仅 GDN 架构模型（Qwen3.5-9B）接入时按需
```

torch 声明为范围，平台已有版本自动跳过；分层与按需内核见 `envs/README.md`。

## 通用用法（仓库根目录执行）

离线训练（与云端同一套核心与指纹）：

```bash
python offline/train.py \
  --annotation-run-id <ANNOTATION_RUN_ID> \
  --model qwen3vl \
  --data-dir data \
  --annotation-root outputs/annotations \
  --output-root outputs \
  --run-tag offline-exp
```

训练适配器和检查点默认写入 `outputs/output_lora/<training_run_id>/`。如平台将
输出挂载到其他目录，可通过 `--output-root` 指定新的仓库级输出根。

推理本体（`--model` 选适配器：qwen3vl / internvl35 / groundingdino / mock）。
VLM 单卡推理标准用 DataLoader 预取（`--num-workers 2` + `--batch-size 4`），
不要用 `--num-shards >1` 在单卡上拉起多个模型副本：

```bash
python offline/infer.py \
  --model qwen3vl \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --lora-path YOUR_LORA_PATH \
  --output-dir outputs/inference \
  --num-workers 2 \
  --batch-size 4 \
  --run-tag offline-test
```

验证集推理显式传入 approved 标注：

```bash
python offline/infer.py \
  --model qwen3vl \
  --test-json outputs/annotations/<ANNOTATION_RUN_ID>/val/approved.json \
  --annotation-run-id <ANNOTATION_RUN_ID> \
  --data-dir data \
  --lora-path YOUR_LORA_PATH
```

所有相对路径相对仓库根解析；平台从其他工作目录启动时可用 `--project-root` 指定。
模型适配器约定（坐标解析、prompt、identity/指纹）见 `aicomp_grounding/models/`。

单机多卡推理才使用 `--num-shards N`，且 N 应等于可见 GPU 数；单卡固定
`--num-shards 1`。多进程路径会写独立的 `shard_checkpoints/shard_<id>.checkpoint.json`，
但恢复以最终 `predictions.json` 为准，中断后请保持稳定会话完整跑完。

`--num-workers 2` 通过 DataLoader 预取图像并与 GPU 推理并行；`--num-workers 0`
使用串行加载。

## 仓库内容与外部数据清单

**仓库自带**：全部代码 / 测试 / 黄金标注集
（`outputs/annotations/<approved_annotation_id>/{train,val}/approved.json`）。

**需要另外获取**：

| 缺的东西 | 体量 | 获取方式 |
| --- | --- | --- |
| `data/Train` 原始三模态（400 序列） | 共 ~43G | 由数据提供方另行获取 |
| `data/Test` 测试集 | 含在 43G 内 | 同上 |
| `data/Processed`（depth JET 伪彩） | 含在内 | 跟着传，或自己跑 `scripts/prepare_rgbdt.py` 重生成 |
| 基础模型权重 | Qwen-8B 17G / InternVL 17G / DINO 0.7G | 下载到持久目录 |

**不需要**：LoRA 权重（融合只交换各自 predictions.json）。

## AMD ROCm 环境

AMD（MI300X）实例上，训练/推理前加载 ROCm 环境变量（已在 venv 激活钩子里自动加载）：

```bash
source offline/rocm_env.sh
```

TunableOp 默认关闭：当前 torch/ROCm 栈直接开启曾有 MI300X 显存泄漏/OOM 风险。
DSW 上宿主机 amdgpu 驱动 6.10.5 与用户态 ROCm 7.2.3 不匹配时，优先反馈平台
提供匹配镜像（L0/L1 层属平台，仓库无法修复，见 `envs/README.md`）。
