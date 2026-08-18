# 💻 Offline End — 离线推理端

离线端：平台无关的纯推理（兜底 / 免费算力）。本体只有 `infer.py`，
任何单卡 CUDA 机器装好依赖即可运行；各平台差异只体现在"怎么装环境"。

## 平台 × 任务矩阵

| 平台 | 训练 | 推理 | 环境怎么装 |
| --- | --- | --- | --- |
| 魔搭 DSW (A10 24G) | ❌ 卡不够 | `infer.py` | `bash offline/dsw/setup.sh` |
| **实体 GPU / 任意 CUDA 机** | 视显存 | `infer.py` | `pip install -r requirements-lock.txt`，**零额外脚本** |
| （未来离线平台 X） | ? | `infer.py` | 往本目录加一个 `x/setup.sh` 预设即可 |

## 通用用法（仓库根目录执行）

直接调本体：

```bash
python offline/infer.py \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --lora-path <下载到本地的 LoRA 目录> \
  --output-dir outputs/inference \
  --run-tag offline-test
```

或用通用启动器（带默认参数 + 数据存在性检查）：

```bash
bash offline/run.sh [LORA_PATH] [DATA_DIR] [MODEL_PATH]
```

## 队友接入清单（git pull 之后你手里有什么、缺什么）

**仓库自带**：全部代码 / 测试 / 黄金标注集
（`outputs/annotations/annot_ac72f1d926bb2d23/{train,val}/approved.json`）。

**需要另外获取**：

| 缺的东西 | 体量 | 获取方式 |
| --- | --- | --- |
| `data/Train` 原始三模态（400 序列） | 共 ~43G | 仓库维护者线下传 |
| `data/Test` 官方测试集 | 含在 43G 内 | 同上 |
| `data/Processed`（depth JET 伪彩） | 含在内 | 跟着传，或自己跑 `scripts/prepare_rgbdt.py` 重生成（确定性输出） |
| `train/val/test.json`、`split_manifest.json` | KB 级 | 直接拷贝（**不要**重生成，避免指纹漂移） |
| 基础模型权重 | Qwen 17G / InternVL 17G / DINO 0.7G | 各自从 HF 或魔搭镜像下载 |

**不需要**：维护者的 LoRA 权重（融合只交换各自 predictions.json）、Zhipu API Key
（标注已随仓库分发，仅重新生成标注时才需要）、维护者的 Modal 凭据。

## DSW 预设

见 [`dsw/`](dsw/)：只有 `setup.sh`（阿里云镜像 + 与云端 `MODAL_GPU_PACKAGES` 同版本
锁定的环境安装）是 DSW 专属；启动器与推理本体全平台通用。
