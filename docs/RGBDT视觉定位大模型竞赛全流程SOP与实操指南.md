# RGBDT 视觉定位：训练与推理 SOP

> 本文仅描述魔搭 DSW 上的数据落位、模型下载、训练和推理操作。
> 数据预处理、索引生成和标注流程见根 README。

## 路径约定

```text
/mnt/workspace/                            持久盘
/mnt/workspace/aicomp_env/                持久虚拟环境
/mnt/workspace/aicomp_env_q38/           Qwen3.8 独立虚拟环境
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

首次执行：

```bash
python3 -m venv --system-site-packages /mnt/workspace/aicomp_env
source /mnt/workspace/aicomp_env/bin/activate
cd /mnt/workspace/aicomp-multimodal-grounding

pip install transformers==4.57.3 peft==0.19.1 \
  accelerate==1.14.0 qwen-vl-utils==0.0.14 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

source offline/rocm_env.sh
echo 'source /mnt/workspace/aicomp-multimodal-grounding/offline/rocm_env.sh' \
  >> /mnt/workspace/aicomp_env/bin/activate
```

之后每次执行：

```bash
source /mnt/workspace/aicomp_env/bin/activate
cd /mnt/workspace/aicomp-multimodal-grounding
```

### Qwen3.8-27B 独立环境

Qwen3.8-27B 的 `model_type` 为 `qwen3_5`，需 `transformers>=5.8.0`，与主环境
`transformers==4.57.3` 不兼容，单独建虚拟环境（venv 落持久盘，小）：

```bash
python3 -m venv --system-site-packages /mnt/workspace/aicomp_env_q38
source /mnt/workspace/aicomp_env_q38/bin/activate
cd /mnt/workspace/aicomp-multimodal-grounding

pip install transformers==5.8.0 peft==0.19.1 \
  accelerate==1.14.0 qwen-vl-utils==0.0.14 \
  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

source offline/rocm_env.sh
echo 'source /mnt/workspace/aicomp-multimodal-grounding/offline/rocm_env.sh' \
  >> /mnt/workspace/aicomp_env_q38/bin/activate
```

之后每次执行 Qwen3.8 训练/推理前，激活此环境：

```bash
source /mnt/workspace/aicomp_env_q38/bin/activate
cd /mnt/workspace/aicomp-multimodal-grounding
```

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
```

下载前检查持久盘空间：

```bash
df -h /mnt/workspace
```

### Qwen3.8-27B（临时盘，每次新实例重下）

Qwen3.8-27B 权重约 55.6 GB，持久盘配额放不下，下到非持久临时盘 `/root`，
每次新实例重下（区别于其他模型落持久盘）：

```bash
export MODEL_ROOT_Q38=/root/models
mkdir -p "$MODEL_ROOT_Q38"

modelscope download --model Qwen/Qwen3.8-27B \
  --local_dir "$MODEL_ROOT_Q38/Qwen/Qwen3.8-27B"
```

下载前检查临时盘空间：

```bash
df -h /root
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

### Qwen3.8-27B

```bash
python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model qwen3_8 \
  --model-path /root/models/Qwen/Qwen3.8-27B \
  --data-dir data \
  --num-workers 4 \
  --checkpoint-interval 20 \
  --smoke-test

python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model qwen3_8 \
  --model-path /root/models/Qwen/Qwen3.8-27B \
  --data-dir data \
  --run-tag exp-qwen38-27b-01 \
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

### Qwen3.8-27B

微调后（需先激活 Qwen3.8 独立环境，权重在临时盘，每次新实例需重下）：

```bash
python offline/infer.py \
  --model qwen3_8 \
  --model-path /root/models/Qwen/Qwen3.8-27B \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --limit 100 --num-shards 1 --num-workers 2 \
  --batch-size 4 --batch-save 100 --run-tag qwen38-lora-smoke

python offline/infer.py \
  --model qwen3_8 \
  --model-path /root/models/Qwen/Qwen3.8-27B \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 2 \
  --batch-size 4 --batch-save 100 --run-tag qwen38-lora-full
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

## 7. 提交包

推理完成后，使用预测结果生成 `submission.zip`：

```bash
python -m aicomp_grounding.submission \
  --test-json data/Test/queries/queries.json \
  --predictions outputs/inference/YOUR_RUN_ID/predictions.json \
  --output-dir outputs/submission/YOUR_RUN_ID \
  --allow-fallback
```
