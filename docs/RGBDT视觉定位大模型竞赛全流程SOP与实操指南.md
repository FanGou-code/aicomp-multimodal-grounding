# RGBDT 视觉定位：训练与推理 SOP

> 本文仅描述魔搭 DSW 上的数据落位、模型下载、训练和推理操作。
> 数据预处理、索引生成和标注流程见根 README。

## 路径约定

```text
/mnt/workspace/                            持久盘
/mnt/workspace/aicomp_env/                持久虚拟环境
/mnt/workspace/aicomp-multimodal-grounding/ 仓库
/mnt/workspace/aicomp-multimodal-grounding/data/ 数据集（仓库内）
/root/models/                              非持久模型目录
```

模型全部下载到 `/root/models`，关机即失，每次 GPU 实例开机后重新下载。

## 1. 准备数据集

```bash
cd /mnt/workspace/aicomp-multimodal-grounding
mkdir -p /tmp/rgbdt-download data

modelscope download --dataset Fang001/rgbdt-grounding-dataset data.tar \
  --token YOUR_MODELSCOPE_TOKEN --local_dir /tmp/rgbdt-download

tar -xf /tmp/rgbdt-download/data.tar -C data --no-same-owner

test -f data/Test/queries/queries.json
test -d data/Train
test -d data/Processed/Train
```

## 2. 创建并激活环境

首次执行：

```bash
python3 -m venv --system-site-packages /mnt/workspace/aicomp_env
source /mnt/workspace/aicomp_env/bin/activate
cd /mnt/workspace/aicomp-multimodal-grounding

pip install -r requirements.txt transformers==4.57.3 peft==0.19.1 \
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

## 3. 下载模型

GPU 实例开机后执行：

```bash
export MODEL_ROOT=/root/models
mkdir -p "$MODEL_ROOT"

modelscope download --model Qwen/Qwen3-VL-8B-Instruct \
  --local_dir "$MODEL_ROOT/Qwen/Qwen3-VL-8B-Instruct"

modelscope download --model Qwen/Qwen3-VL-32B-Instruct \
  --local_dir "$MODEL_ROOT/Qwen/Qwen3-VL-32B-Instruct"

modelscope download --model OpenGVLab/InternVL3_5-8B-HF \
  --local_dir "$MODEL_ROOT/OpenGVLab/InternVL3_5-8B-HF"

modelscope download --model AI-ModelScope/GroundingDINO \
  --local_dir "$MODEL_ROOT/AI-ModelScope/grounding-dino-base"
```

下载前检查临时盘空间：

```bash
df -h /root
```

## 4. 训练

训练命令统一使用 `offline/train.py`。先跑 smoke，再启动完整训练。

### Qwen3-VL-8B

```bash
python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model qwen3vl \
  --model-path /root/models/Qwen/Qwen3-VL-8B-Instruct \
  --data-dir data \
  --smoke-test

python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model qwen3vl \
  --model-path /root/models/Qwen/Qwen3-VL-8B-Instruct \
  --data-dir data \
  --run-tag exp-qwen-01
```

### Qwen3-VL-32B

```bash
python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model qwen3vl32 \
  --model-path /root/models/Qwen/Qwen3-VL-32B-Instruct \
  --data-dir data \
  --num-workers 0 \
  --checkpoint-interval 20 \
  --smoke-test

python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model qwen3vl32 \
  --model-path /root/models/Qwen/Qwen3-VL-32B-Instruct \
  --data-dir data \
  --run-tag exp-qwen32-01 \
  --num-workers 0 \
  --checkpoint-interval 20
```

### InternVL3.5-8B

```bash
python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model internvl35 \
  --model-path /root/models/OpenGVLab/InternVL3_5-8B-HF \
  --data-dir data \
  --smoke-test

python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model internvl35 \
  --model-path /root/models/OpenGVLab/InternVL3_5-8B-HF \
  --data-dir data \
  --run-tag exp-internvl-01
```

## 5. 推理

推理统一使用：

```text
--test-json data/Test/queries/queries.json
--data-dir data
--num-shards 1
--num-workers 4
```

每个模型按微调后（带 LoRA）执行。每类先 smoke，再全量。

### Qwen3-VL-8B

微调后：

```bash
python offline/infer.py \
  --model qwen3vl \
  --model-path /root/models/Qwen/Qwen3-VL-8B-Instruct \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_03 \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --limit 100 --num-shards 1 --num-workers 4 \
  --batch-size 16 --batch-save 100 --run-tag qwen8-lora-smoke

python offline/infer.py \
  --model qwen3vl \
  --model-path /root/models/Qwen/Qwen3-VL-8B-Instruct \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_03 \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 4 \
  --batch-size 16 --batch-save 100 --run-tag qwen8-lora-full
```

### Qwen3-VL-32B

微调后：

```bash
python offline/infer.py \
  --model qwen3vl32 \
  --model-path /root/models/Qwen/Qwen3-VL-32B-Instruct \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_03 \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --limit 100 --num-shards 1 --num-workers 2 \
  --batch-size 4 --batch-save 100 --run-tag qwen32-lora-smoke

python offline/infer.py \
  --model qwen3vl32 \
  --model-path /root/models/Qwen/Qwen3-VL-32B-Instruct \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_03 \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 2 \
  --batch-size 4 --batch-save 100 --run-tag qwen32-lora-full
```

### InternVL3.5-8B

微调后：

```bash
python offline/infer.py \
  --model internvl35 \
  --model-path /root/models/OpenGVLab/InternVL3_5-8B-HF \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_03 \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --limit 100 --num-shards 1 --num-workers 4 \
  --batch-size 16 --batch-save 100 --run-tag internvl-lora-smoke

python offline/infer.py \
  --model internvl35 \
  --model-path /root/models/OpenGVLab/InternVL3_5-8B-HF \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_03 \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 4 \
  --batch-size 16 --batch-save 100 --run-tag internvl-lora-full
```

## 6. 提交包

推理完成后，使用预测结果生成 `submission.zip`：

```bash
python -m aicomp_grounding.submission \
  --test-json data/Test/queries/queries.json \
  --predictions outputs/inference/YOUR_RUN_ID/predictions.json \
  --output-dir outputs/submission/YOUR_RUN_ID \
  --allow-fallback
```
