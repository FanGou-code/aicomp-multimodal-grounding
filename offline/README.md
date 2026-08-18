# 💻 Offline End — 离线端

离线端：平台无关的训练与推理（兜底 / 免费算力）。推理本体是 `infer.py`，训练本体是
`train.py`（与云端共用 `aicomp_grounding/training_core`，行为完全一致）；任何单卡
CUDA 机器装好依赖即可运行，各平台差异只体现在"怎么装环境"。

## 平台 × 任务矩阵

| 平台 | 训练 | 推理 | 环境怎么装 |
| --- | --- | --- | --- |
| 魔搭 DSW (A10 24G) | ❌ 卡不够（QLoRA 低分辨率实验除外） | `infer.py` | 见下方「魔搭 DSW 环境」 |
| **实体 GPU / 任意 CUDA 机** | `train.py`（≥48GB 满血 / 24GB 需降像素预算） | `infer.py` | `pip install -r requirements-lock.txt` |

## 通用用法（仓库根目录执行）

离线训练（与云端同一套核心与指纹）：

```bash
python offline/train.py \
  --annotation-run-id annot_ac72f1d926bb2d23 \
  --data-dir data \
  --run-tag offline-exp
```

直接调推理本体（`--model` 选适配器：qwen3vl / internvl35 / groundingdino / mock）：

```bash
python offline/infer.py \
  --model qwen3vl \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --lora-path <下载到本地的 LoRA 目录> \
  --output-dir outputs/inference \
  --run-tag offline-test

# 队友方向示例（zero-shot，无需 LoRA）：
python offline/infer.py --model groundingdino --test-json data/test.json ...
python offline/infer.py --model internvl35 --test-json data/test.json ...
```

模型适配器的约定（坐标解析、prompt、identity/指纹）见
`aicomp_grounding/models/`；InternVL / GroundingDINO 的 GPU 路径尚未冒烟，
首次使用先跑小切片验证。

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

## 魔搭 DSW 环境

DSW 镜像已预装 torch / torchvision / pillow（不重装，避免覆盖镜像自带版本），
只需补齐四个推理库——与云端 `aicomp_grounding/config.py` 的 `MODAL_GPU_PACKAGES`
同版本锁定，走阿里云镜像：

```bash
pip install transformers==4.57.3 peft==0.19.1 accelerate==1.14.0 \
    qwen-vl-utils==0.0.14 \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

基础模型建议从魔搭镜像下载到本地，`--model-path` 指定本地目录，
避免直连 HuggingFace。
