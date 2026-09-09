# RGBDT 视觉定位：训练与推理 SOP

> 本文仅描述魔搭 DSW 上的数据落位、模型下载、训练和推理操作。
> 数据预处理、索引生成和标注流程见根 README。

## 路径约定

```text
/mnt/workspace/                            持久盘
/mnt/workspace/aicomp_env/                持久虚拟环境
/mnt/workspace/aicomp-multimodal-grounding/ 仓库
/mnt/workspace/aicomp-multimodal-grounding/data/ 数据集（仓库内，持久）
/mnt/workspace/models/                     持久模型目录
/root/                                     非持久临时工作区
```

## 1. 克隆项目仓库（首次执行）

```bash
cd /mnt/workspace
git clone https://YOUR_GITHUB_TOKEN@ghfast.top/https://github.com/FanGou-code/aicomp-multimodal-grounding.git
cd aicomp-multimodal-grounding
```

## 2. 准备数据集（首次执行）

```bash
cd /mnt/workspace/aicomp-multimodal-grounding
mkdir -p /root/rgbdt-download data

modelscope download --dataset Fang001/rgbdt-grounding-dataset data.tar \
  --local_dir /root/rgbdt-download

tar -xf /root/rgbdt-download/data.tar -C data --no-same-owner
rm -rf /root/rgbdt-download

test -f data/Test/queries/queries.json
test -d data/Train
test -d data/Processed/Train
```

## 3. 创建并激活环境

依赖唯一声明源是根 `pyproject.toml`（`dependencies` + extras）。torch 为范围
（`>=2.8,<3`），平台镜像自带版本落在范围内即被 pip 判定已满足、自动跳过；
其余四件套 `==` 紧 pin，跨平台一致。

**首次执行（建一次）**：

```bash
python3 -m venv --system-site-packages /mnt/workspace/aicomp_env
source /mnt/workspace/aicomp_env/bin/activate
cd /mnt/workspace/aicomp-multimodal-grounding

pip install -e . \
  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

source offline/rocm_env.sh
echo 'source /mnt/workspace/aicomp-multimodal-grounding/offline/rocm_env.sh' \
  >> /mnt/workspace/aicomp_env/bin/activate
```

之后每次执行：

```bash
source /mnt/workspace/aicomp_env/bin/activate
cd /mnt/workspace/aicomp-multimodal-grounding
```

验证（10 秒，激活后必做）：

```bash
python -c "import transformers, peft; print(transformers.__version__, transformers.__file__)"
# 应输出 5.14.1，路径落在 venv 内
```

目标模型为混合线性注意力架构（`qwen3_5` 类）时，额外安装加速内核
`pip install -e ".[kernels]"`：

- **N 卡 / CUDA 机**：一键即可（FLA 与 causal-conv1d 均有预编译轮）。
- **A 卡（DSW）**：FLA 有预编译；causal-conv1d 无 HIP 预编译轮，需源码编译：

```bash
export PYTORCH_ROCM_ARCH=gfx942 MAX_JOBS=8
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
pip install causal-conv1d --no-build-isolation \
  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
# 编不过可直接跳过；FLA 是主要收益来源
python -c "import causal_conv1d; print('conv OK')"
```

注意：causal-conv1d 源码编译产物不进 pip 缓存，重建 venv 后需重编。

**venv 在持久盘 `/mnt/workspace` 上，同镜像代际的实例间直接复用，无需重建。**
仅当持久盘被清空、或镜像大版本更换导致底座 python 路径变化时，按「首次执行」
重走一遍即可（pip 与 Triton 缓存持久，重装为秒级）。

## 4. 下载模型（首次执行）

```bash
export MODEL_ROOT=/mnt/workspace/models
mkdir -p "$MODEL_ROOT"

modelscope download --model Qwen/Qwen3-VL-8B-Instruct \
  --local_dir "$MODEL_ROOT/Qwen/Qwen3-VL-8B-Instruct"

modelscope download --model OpenGVLab/InternVL3_5-8B-HF \
  --local_dir "$MODEL_ROOT/OpenGVLab/InternVL3_5-8B-HF"

modelscope download --model AI-ModelScope/grounding-dino-base \
  --local_dir "$MODEL_ROOT/AI-ModelScope/grounding-dino-base"

modelscope download --model Qwen/Qwen3.5-9B \
  --local_dir "$MODEL_ROOT/Qwen/Qwen3.5-9B"

modelscope download --model ZhipuAI/GLM-4.6V-Flash \
  --local_dir "$MODEL_ROOT/ZhipuAI/GLM-4.6V-Flash"
```

> Youtu-VL-4B-Instruct 不在 DSW 下载：需 `transformers>=4.56.0,<=4.57.1` +
> `trust_remote_code`，与本环境 5.14.1 冲突。由管理员在 Modal 专属 Image 内拉取，
> 流程见 `cloud/README.md`；预测文件回流本仓后入 WBF 融合。

下载前检查持久盘空间：

```bash
df -h /mnt/workspace
```

## 5. 训练

训练命令统一使用 `offline/train.py`。先跑 smoke，再启动完整训练。

### Qwen3-VL-8B

```bash
python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model qwen3vl \
  --model-path /mnt/workspace/models/Qwen/Qwen3-VL-8B-Instruct \
  --data-dir data \
  --num-workers 4 \
  --checkpoint-interval 20 \
  --smoke-test

python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model qwen3vl \
  --model-path /mnt/workspace/models/Qwen/Qwen3-VL-8B-Instruct \
  --data-dir data \
  --run-tag exp-qwen8-retrain \
  --num-workers 4 \
  --checkpoint-interval 20
```

### Qwen3.5-9B

```bash
python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model qwen3_5 \
  --model-path /mnt/workspace/models/Qwen/Qwen3.5-9B \
  --data-dir data \
  --num-workers 4 \
  --checkpoint-interval 20 \
  --smoke-test

python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model qwen3_5 \
  --model-path /mnt/workspace/models/Qwen/Qwen3.5-9B \
  --data-dir data \
  --run-tag exp-qwen35-9b \
  --num-workers 4 \
  --checkpoint-interval 20
```

### GLM-4.6V-Flash

```bash
python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model glm46v \
  --model-path /mnt/workspace/models/ZhipuAI/GLM-4.6V-Flash \
  --data-dir data \
  --num-workers 4 \
  --checkpoint-interval 20 \
  --smoke-test

python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model glm46v \
  --model-path /mnt/workspace/models/ZhipuAI/GLM-4.6V-Flash \
  --data-dir data \
  --run-tag exp-glm46v-flash \
  --num-workers 4 \
  --checkpoint-interval 20
```

### InternVL3.5-8B

```bash
python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model internvl35 \
  --model-path /mnt/workspace/models/OpenGVLab/InternVL3_5-8B-HF \
  --data-dir data \
  --smoke-test

python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model internvl35 \
  --model-path /mnt/workspace/models/OpenGVLab/InternVL3_5-8B-HF \
  --data-dir data \
  --run-tag exp-internvl-01
```

### GroundingDINO-B

GroundingDINO 走零样本推理，不执行 `offline/train.py`，也不需要 LoRA。

## 6. 推理

VLM 三图模型统一使用：

```text
--test-json data/Test/queries/queries.json
--data-dir data
--num-shards 1
--num-workers 2
--batch-size 4
--batch-save 100
```

每个 VLM 按微调后（带 LoRA）执行。GroundingDINO 使用
`--num-workers 4 --batch-size 8` 且不传 `--lora-path`。每类先 smoke，再全量。

### Qwen3-VL-8B

微调后：

```bash
python offline/infer.py \
  --model qwen3vl \
  --model-path /mnt/workspace/models/Qwen/Qwen3-VL-8B-Instruct \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --limit 100 --num-shards 1 --num-workers 2 \
  --batch-size 4 --batch-save 100 --run-tag qwen8-lora-smoke

python offline/infer.py \
  --model qwen3vl \
  --model-path /mnt/workspace/models/Qwen/Qwen3-VL-8B-Instruct \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 2 \
  --batch-size 4 --batch-save 100 --run-tag qwen8-lora-full
```

### Qwen3.5-9B

微调后（命令形状与 Qwen3-VL-8B 一致，超参 α32/3ep 已落 adapter 默认）：

```bash
python offline/infer.py \
  --model qwen3_5 \
  --model-path /mnt/workspace/models/Qwen/Qwen3.5-9B \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --limit 100 --num-shards 1 --num-workers 2 \
  --batch-size 4 --batch-save 100 --run-tag qwen35-9b-lora-smoke

python offline/infer.py \
  --model qwen3_5 \
  --model-path /mnt/workspace/models/Qwen/Qwen3.5-9B \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 2 \
  --batch-size 4 --batch-save 100 --run-tag qwen35-9b-lora-full
```

### GLM-4.6V-Flash

微调后（亦可直接 zero-shot 探针，省略 `--lora-path`）：

```bash
python offline/infer.py \
  --model glm46v \
  --model-path /mnt/workspace/models/ZhipuAI/GLM-4.6V-Flash \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --limit 100 --num-shards 1 --num-workers 2 \
  --batch-size 4 --batch-save 100 --run-tag glm46v-lora-smoke

python offline/infer.py \
  --model glm46v \
  --model-path /mnt/workspace/models/ZhipuAI/GLM-4.6V-Flash \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 2 \
  --batch-size 4 --batch-save 100 --run-tag glm46v-lora-full
```

### InternVL3.5-8B

微调后：

```bash
python offline/infer.py \
  --model internvl35 \
  --model-path /mnt/workspace/models/OpenGVLab/InternVL3_5-8B-HF \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --limit 100 --num-shards 1 --num-workers 2 \
  --batch-size 4 --batch-save 100 --run-tag internvl-lora-smoke

python offline/infer.py \
  --model internvl35 \
  --model-path /mnt/workspace/models/OpenGVLab/InternVL3_5-8B-HF \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 2 \
  --batch-size 4 --batch-save 100 --run-tag internvl-lora-full
```

### GroundingDINO-B（Zero-shot）

```bash
python offline/infer.py \
  --model groundingdino \
  --model-path /mnt/workspace/models/AI-ModelScope/grounding-dino-base \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --limit 100 --num-shards 1 --num-workers 4 \
  --batch-size 8 --batch-save 100 --run-tag dino-smoke

python offline/infer.py \
  --model groundingdino \
  --model-path /mnt/workspace/models/AI-ModelScope/grounding-dino-base \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 4 \
  --batch-size 8 --batch-save 100 --run-tag dino-full
```

### Youtu-VL-4B（Zero-shot，Modal 专属）

Youtu-VL 不走 `offline/infer.py`：需 `transformers<=4.57.1` + `trust_remote_code` +
`pydensecrf`，与 DSW 5.14.1 冲突。由管理员在 Modal 专属 Image 内执行推理，预测
文件 `predictions.json` 回流本仓 `outputs/inference/` 后入 WBF 融合。适配层
`youtu_vl` 已就绪，本地解析单测覆盖；GPU 路径（多图 `img_input` kwarg）在 Modal
首跑时冒烟验证。

## 7. 提交包

推理完成后，使用预测结果生成 `submission.zip`：

```bash
python -m aicomp_grounding.submission \
  --test-json data/Test/queries/queries.json \
  --predictions outputs/inference/YOUR_RUN_ID/predictions.json \
  --output-dir outputs/submission/YOUR_RUN_ID \
  --allow-fallback
```
