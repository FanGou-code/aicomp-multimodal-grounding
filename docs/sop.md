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

## 3. 激活环境（先唤起，验证失败才重建）

依赖 pin 的单一来源是仓库 `envs/gpu.txt`（transformers==5.14.1 与 DSW
镜像自带版本一致；torch/ROCm 归镜像管，不进 pip）。

**首选：直接唤起持久盘上的 venv**（同一镜像代际的实例间可直接复用）：

```bash
source /mnt/workspace/aicomp_env/bin/activate
cd /mnt/workspace/aicomp-multimodal-grounding
```

**唤起后必做验证（10 秒，这一步是判定器）**：

```bash
python -c "import transformers, peft; print(transformers.__version__)"
# 应输出 5.14.1
```

若目标模型为**混合线性注意力架构**（`qwen3_5` 类），需额外安装 A 卡加速内核
FLA 与 causal-conv1d——完整命令见 `envs/README.md` 的「A 卡加速内核（按需）」
一节（标准注意力模型不需要）。

**仅当验证失败时重建**。两种触发：报 `bad interpreter`（镜像更新导致底座
python 路径变化，venv 的解释器符号链接悬空）；或版本号不是 5.14.1
（`envs/gpu.txt` 升级后）。重建有持久缓存（`PIP_CACHE_DIR` / `TRITON_CACHE_DIR`），
包不重新下载、Triton 内核不重新编译：

```bash
deactivate 2>/dev/null
rm -rf /mnt/workspace/aicomp_env
python3 -m venv --system-site-packages /mnt/workspace/aicomp_env
source /mnt/workspace/aicomp_env/bin/activate

pip install -r envs/gpu.txt \
  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

echo 'source /mnt/workspace/aicomp-multimodal-grounding/offline/rocm_env.sh' \
  >> /mnt/workspace/aicomp_env/bin/activate

python -c "import transformers, peft; print(transformers.__version__)"
# 重建后必须再次通过验证，然后才能进入后续步骤
```

venv 原理备注：`bin/python3` 是指向镜像底座解释器的符号链接，镜像更新可能
使其悬空；已安装的库文件在持久盘上不会丢失，重建只是重新链接并从本地
缓存解包。

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
