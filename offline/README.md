# 💻 Offline End — 离线端

离线端：平台无关的训练与推理（兜底 / 免费算力）。推理本体是 `infer.py`，训练本体是
`train.py`（与云端共用 `aicomp_grounding/training_core`，行为完全一致）；任何单卡
GPU 机器装好依赖即可运行，各平台差异只体现在"怎么装环境"。

可移植仓库的约定是：数据索引位于 `data/`，已批准标注和训练/推理产物位于
仓库级 `outputs/`。Modal 仍由 `cloud/` 适配器使用历史 Volume 布局，离线端不依赖该布局。

## 平台 × 任务矩阵

| 平台 | 训练 | 推理 | 环境怎么装 |
| --- | --- | --- | --- |
| 云端 GPU 工作台（预装 PyTorch） | 依据实际显存与像素预算决定 | `infer.py` / `train.py` | 沿用平台 PyTorch，按下方「云端 GPU 工作台环境」补齐 VLM 依赖 |
| **实体 GPU / 任意 GPU 机** | `train.py`（显存 ≥48GB / 24GB 需下调像素预算） | `infer.py` | `pip install -r requirements-lock.txt`，并按下方说明补齐 VLM 依赖 |

## 通用用法（仓库根目录执行）

离线训练（与云端同一套核心与指纹）：

```bash
python offline/train.py \
  --annotation-run-id annot_ac72f1d926bb2d23 \
  --model qwen3vl \
  --data-dir data \
  --annotation-root outputs/annotations \
  --output-root outputs \
  --run-tag offline-exp
```

训练适配器和检查点默认写入
`outputs/output_lora/<training_run_id>/`。如平台将输出挂载到其他目录，可通过
`--output-root` 指定新的仓库级输出根。

直接调推理本体（`--model` 选适配器：qwen3vl / qwen3vl32 / internvl35 / groundingdino / mock）：
单卡推理推荐用 DataLoader 预取（`--num-workers 4`），不要用
`--num-shards >1` 在单卡上拉起多个模型副本：

```bash
python offline/infer.py \
  --model qwen3vl \
  --test-json data/test.json \
  --data-dir data \
  --lora-path YOUR_LORA_PATH \
  --output-dir outputs/inference \
  --num-workers 4 \
  --batch-size 4 \
  --run-tag offline-test

# 其他模型示例（zero-shot，无需 LoRA）：
python offline/infer.py --model groundingdino --test-json data/test.json ...
python offline/infer.py --model internvl35 --test-json data/test.json ...
```

`data/test.json` is the processed worker index used to load images. The
official `data/Test/queries/queries.json` file is a submission template only;
it is consumed when a complete test run packages `submission.zip`. For
validation, pass the approved artifact explicitly, for example:

```bash
python offline/infer.py \
  --model qwen3vl \
  --test-json outputs/annotations/annot_ac72f1d926bb2d23/val/approved.json \
  --annotation-run-id annot_ac72f1d926bb2d23 \
  --data-dir data \
  --lora-path YOUR_LORA_PATH
```

All relative paths resolve from the repository root. `--project-root` can be
used when a platform starts the process from another working directory.

模型适配器的约定（坐标解析、prompt、identity/指纹）见
`aicomp_grounding/models/`；InternVL / GroundingDINO 的 GPU 路径尚未冒烟，
首次使用先跑小切片验证。

单机多卡推理才使用 `--num-shards N`，且 N 应等于可见 GPU 数；单卡 MI300X
固定为 `--num-shards 1`。多进程路径会写入独立的
`shard_checkpoints/shard_<id>.checkpoint.json`，但当前恢复以最终
`predictions.json` 为准，中断后请保持稳定会话完整跑完。

`--num-workers 4` 是单卡推荐默认值，通过 DataLoader 预取图像与 GPU 推理
并行；`--num-workers 0` 保留旧的串行加载行为。

## 仓库内容与外部数据清单

**仓库自带**：全部代码 / 测试 / 黄金标注集
（`outputs/annotations/annot_ac72f1d926bb2d23/{train,val}/approved.json`）。

**需要另外获取**：

| 缺的东西 | 体量 | 获取方式 |
| --- | --- | --- |
| `data/Train` 原始三模态（400 序列） | 共 ~43G | 由数据提供方另行获取 |
| `data/Test` 测试集 | 含在 43G 内 | 同上 |
| `data/Processed`（depth JET 伪彩） | 含在内 | 跟着传，或自己跑 `scripts/prepare_rgbdt.py` 重生成（确定性输出） |
| `train/val/test.json`、`split_manifest.json` | KB 级 | 直接拷贝（**不要**重生成，避免指纹漂移） |
| 基础模型权重 | Qwen-8B 17G / **Qwen-32B ~66G** / InternVL 17G / DINO 0.7G | 各自从 HF 或魔搭镜像下载；32B 不落持久盘，下载到实例临时盘（非持久）、每次开机重新拉取 |

**不需要**：LoRA 权重（融合只交换各自 predictions.json）、Zhipu API Key
（标注已随仓库分发，仅重新生成标注时才需要）、Modal 凭据。

## 云端 GPU 工作台环境

云端 GPU 工作台通常已预装 torch / torchvision / pillow。应优先沿用平台 PyTorch，
避免覆盖镜像自带版本；再按需补齐 VLM 适配层依赖。版本以
`aicomp_grounding/config.py` 中的 `MODAL_GPU_PACKAGES` 为基准：

```bash
pip install transformers==4.57.3 peft==0.19.1 accelerate==1.14.0 \
    qwen-vl-utils==0.0.14 \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

基础模型建议从魔搭镜像下载到本地，`--model-path` 指定本地目录，
避免直连 HuggingFace。
