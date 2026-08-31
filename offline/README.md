# Offline End — 离线端

离线端提供平台无关的训练与推理入口。`offline/infer.py` 调用推理核心，
`offline/train.py` 与云端共用 `aicomp_grounding/training_core`。

数据索引位于 `data/`，已批准标注和训练/推理产物位于仓库级 `outputs/`。
Modal 由 `cloud/` 适配器使用 Volume 布局，离线端不依赖该布局。

## 平台 × 任务矩阵

| 平台 | 训练 | 推理 | 环境怎么装 |
| --- | --- | --- | --- |
| 云端 GPU 工作台（预装 PyTorch） | 依据实际显存与像素预算决定 | `infer.py` / `train.py` | 沿用平台 PyTorch，按下方「云端 GPU 工作台环境」补齐 VLM 依赖 |
| **实体 GPU / 任意 GPU 机** | `train.py`（显存 ≥48GB / 24GB 需下调像素预算） | `infer.py` | `pip install -r requirements-lock.txt`，并按下方说明补齐 VLM 依赖 |

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

训练适配器和检查点默认写入
`outputs/output_lora/<training_run_id>/`。如平台将输出挂载到其他目录，可通过
`--output-root` 指定新的仓库级输出根。

直接调推理本体（`--model` 选适配器：qwen3vl / internvl35 / groundingdino / mock）：
VLM 单卡推理标准用 DataLoader 预取（`--num-workers 2` + `--batch-size 4`），不要用
`--num-shards >1` 在单卡上拉起多个模型副本：

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

# 其他模型示例（zero-shot，无需 LoRA）：
python offline/infer.py --model groundingdino --test-json data/Test/queries/queries.json ...
python offline/infer.py --model internvl35 --test-json data/Test/queries/queries.json ...
```

推理可直接使用 `data/Test/queries/queries.json`；`inference_core` 会在内存中
把原始模态路径映射到 `Test/Images/...` 与
`Processed/Test/depth_jet/...`。验证集推理显式传入 approved 标注，例如：

```bash
python offline/infer.py \
  --model qwen3vl \
  --test-json outputs/annotations/<ANNOTATION_RUN_ID>/val/approved.json \
  --annotation-run-id <ANNOTATION_RUN_ID> \
  --data-dir data \
  --lora-path YOUR_LORA_PATH
```

All relative paths resolve from the repository root. `--project-root` can be
used when a platform starts the process from another working directory.

模型适配器的约定（坐标解析、prompt、identity/指纹）见
`aicomp_grounding/models/`；InternVL 与 GroundingDINO 的协议契约已单测对齐，
真实 GPU 路径仍应在首跑时用小切片验证。

单机多卡推理才使用 `--num-shards N`，且 N 应等于可见 GPU 数；单卡 MI300X
固定为 `--num-shards 1`。多进程路径会写入独立的
`shard_checkpoints/shard_<id>.checkpoint.json`，但当前恢复以最终
`predictions.json` 为准，中断后请保持稳定会话完整跑完。

`--num-workers 2` 通过 DataLoader 预取图像并与 GPU 推理并行；
`--num-workers 0` 使用串行加载。

## 仓库内容与外部数据清单

**仓库自带**：全部代码 / 测试 / 黄金标注集
（`outputs/annotations/<approved_annotation_id>/{train,val}/approved.json`）。

**需要另外获取**：

| 缺的东西 | 体量 | 获取方式 |
| --- | --- | --- |
| `data/Train` 原始三模态（400 序列） | 共 ~43G | 由数据提供方另行获取 |
| `data/Test` 测试集 | 含在 43G 内 | 同上 |
| `data/Processed`（depth JET 伪彩） | 含在内 | 跟着传，或自己跑 `scripts/prepare_rgbdt.py` 重生成（确定性输出） |
| `data/train.json`、`data/val.json`、`data/split_manifest.json`、`data/excluded_overlap.json` | KB 级 | 仅本地预处理/审计使用，云端不需要；官方 `Test/queries/queries.json` 随 `data/Test` 提供 |
| 基础模型权重 | Qwen-8B 17G / InternVL 17G / DINO 0.7G | 下载到持久目录 `/mnt/workspace/models` |

**不需要**：LoRA 权重（融合只交换各自 predictions.json）、Zhipu API Key
（标注已随仓库分发，仅重新生成标注时才需要）、Modal 凭据。

## 云端 GPU 工作台环境

云端 GPU 工作台通常已预装 torch / torchvision / pillow。应优先沿用平台 PyTorch，
避免覆盖镜像自带版本；VLM 适配层依赖的 pin 以 `envs/gpu.txt` 为单一来源：

```bash
pip install -r envs/gpu.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

基础模型建议从魔搭镜像下载到本地，`--model-path` 指定本地目录，
避免直连 HuggingFace。

### AMD ROCm 环境

在 AMD MI300X 实例上，每次训练/推理前先加载仓库内置环境：

```bash
source offline/rocm_env.sh
```

TunableOp 默认关闭：当前 torch/ROCm 栈直接开启曾有 MI300X 显存泄漏/OOM 风险，
如需回开必须先用持久结果文件做离线 tuning 并重新 benchmark。脚本没有配置
allocator / expandable-segments；DSW 上宿主机 amdgpu 驱动 6.10.5 与用户态
ROCm 7.2.3 不匹配时，优先反馈平台提供匹配镜像。

也可以追加到持久虚拟环境的激活脚本，让 `source <venv>/bin/activate` 自动生效：

```bash
echo 'source /mnt/workspace/aicomp-multimodal-grounding/offline/rocm_env.sh' >> /mnt/workspace/aicomp_env/bin/activate
```
