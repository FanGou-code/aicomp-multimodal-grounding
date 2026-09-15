# RGBDT 视觉定位：GPU 训练与推理 SOP

> 本文是 GPU 实操的唯一真相源：环境、模型下载、逐模型训练与推理命令。
> 数据布局与标注产物合同见 `docs/data-contract.md`，架构约定见 `docs/architecture.md`。

## 路径约定

```text
/mnt/workspace/                                持久盘
/mnt/workspace/aicomp_env/                     持久虚拟环境
/mnt/workspace/aicomp-multimodal-grounding/    仓库
/mnt/workspace/aicomp-multimodal-grounding/data/  数据集（仓库内，持久）
/mnt/workspace/models/                         持久模型目录
/root/                                         非持久临时工作区
```

## 1. 获取仓库（首次执行）

```bash
cd /mnt/workspace
git clone https://github.com/FanGou-code/aicomp-multimodal-grounding.git
cd aicomp-multimodal-grounding
```

## 2. 准备数据集（首次执行）

数据包已包含生成好的 `Processed/`，解压后直接使用，不重复生成：

```bash
cd /mnt/workspace/aicomp-multimodal-grounding
mkdir -p /root/rgbdt-download data

modelscope download --dataset Fang001/rgbdt-grounding-dataset data.tar \
  --local_dir /root/rgbdt-download

tar -xf /root/rgbdt-download/data.tar -C data --no-same-owner
rm -rf /root/rgbdt-download
```

另将赛事渠道取得的 Test 目录放入 `data/Test/`，并放入训练用标注产物
`outputs/annotations/<run_id>/{train,val}/approved.json`。检查布局：

```bash
test -f data/Test/queries/queries.json
test -d data/Train
test -d data/Processed/Train
test -f outputs/annotations/<run_id>/train/approved.json
```

## 3. 创建并激活环境

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
# 应输出 5.15.1
```

`qwen3_5` 为混合线性注意力架构（GDN），需要额外安装加速内核；缺内核时不会报错，
但会静默回退到较慢的 torch 实现：

- **N 卡 / CUDA 机**：一键安装（FLA 与 causal-conv1d 均有预编译轮）：

```bash
pip install -e ".[kernels]" \
  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

- **A 卡（ROCm）**：先源码编译 `causal-conv1d`（无 HIP 预编译轮），再安装 `[kernels]`：

```bash
export PYTORCH_ROCM_ARCH=gfx942 MAX_JOBS=8
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
pip install causal-conv1d --no-build-isolation \
  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

pip install -e ".[kernels]" \
  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

python -c "import fla, causal_conv1d; print(fla.__version__, causal_conv1d.__version__)"
```

## 4. 下载模型（首次执行）

```bash
export MODEL_ROOT=/mnt/workspace/models
mkdir -p "$MODEL_ROOT"

modelscope download --model Qwen/Qwen3-VL-8B-Instruct \
  --revision 5d854aab08710c16b980ec6d603d863b3821b915 \
  --local_dir "$MODEL_ROOT/Qwen/Qwen3-VL-8B-Instruct"

modelscope download --model Qwen/Qwen3.5-9B \
  --revision 460979c3d11864dd16408d860ac930a360a2fac2 \
  --local_dir "$MODEL_ROOT/Qwen/Qwen3.5-9B"

modelscope download --model XiaomiMiMo/MiMo-VL-7B-RL \
  --revision d307865d4a3b6ad9ae35e574bcabaa563038c8fb \
  --local_dir "$MODEL_ROOT/XiaomiMiMo/MiMo-VL-7B-RL"

modelscope download --model ZhipuAI/GLM-4.6V-Flash \
  --revision a4ec61fcdfab32bbccdf26c5ca8cb5a437b7ca41 \
  --local_dir "$MODEL_ROOT/ZhipuAI/GLM-4.6V-Flash"

modelscope download --model IDEA-Research/grounding-dino-base \
  --revision d06985a44c66b6133c131bd273293be8649cfe3a \
  --local_dir "$MODEL_ROOT/IDEA-Research/grounding-dino-base"
```

下载前检查持久盘空间：

```bash
df -h /mnt/workspace
```

## 5. 训练

### 冒烟

冒烟只跑 1 个 micro-batch；权重加载期间没有输出属正常，用 `rocm-smi` 看利用率。
通过时打印 `Training smoke passed` 与 `trainable params: <N>`，`<N>` 期望值：

| 模型 | `trainable params` |
| --- | --- |
| `qwen3vl` | 43,646,976 |
| `qwen3_5` | 29,097,984 |
| `mimo_vl` | 41,435,136 |
| `glm46v` | 27,443,200 |

### Qwen3-VL-8B

```bash
python offline/train.py \
  --annotation-run-id <run_id> \
  --model qwen3vl \
  --model-path /mnt/workspace/models/Qwen/Qwen3-VL-8B-Instruct \
  --data-dir data \
  --batch-size 1 \
  --eval-batch-size 1 \
  --checkpoint-interval 20 \
  --smoke-test

python offline/train.py \
  --annotation-run-id <run_id> \
  --model qwen3vl \
  --model-path /mnt/workspace/models/Qwen/Qwen3-VL-8B-Instruct \
  --data-dir data \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 \
  --epochs 3 \
  --eval-batch-size 1 \
  --run-tag exp-qwen3vl \
  --num-workers 4 \
  --checkpoint-interval 20
```

### Qwen3.5-9B

```bash
python offline/train.py \
  --annotation-run-id <run_id> \
  --model qwen3_5 \
  --model-path /mnt/workspace/models/Qwen/Qwen3.5-9B \
  --data-dir data \
  --batch-size 1 \
  --eval-batch-size 1 \
  --checkpoint-interval 20 \
  --smoke-test

python offline/train.py \
  --annotation-run-id <run_id> \
  --model qwen3_5 \
  --model-path /mnt/workspace/models/Qwen/Qwen3.5-9B \
  --data-dir data \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 \
  --epochs 3 \
  --eval-batch-size 1 \
  --run-tag exp-qwen35 \
  --num-workers 4 \
  --checkpoint-interval 20
```

### MiMo-VL-7B-RL

```bash
python offline/train.py \
  --annotation-run-id <run_id> \
  --model mimo_vl \
  --model-path /mnt/workspace/models/XiaomiMiMo/MiMo-VL-7B-RL \
  --data-dir data \
  --batch-size 1 \
  --eval-batch-size 1 \
  --checkpoint-interval 20 \
  --smoke-test

python offline/train.py \
  --annotation-run-id <run_id> \
  --model mimo_vl \
  --model-path /mnt/workspace/models/XiaomiMiMo/MiMo-VL-7B-RL \
  --data-dir data \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 \
  --epochs 3 \
  --eval-batch-size 1 \
  --run-tag exp-mimo-vl \
  --num-workers 4 \
  --checkpoint-interval 20
```

### GLM-4.6V-Flash

```bash
python offline/train.py \
  --annotation-run-id <run_id> \
  --model glm46v \
  --model-path /mnt/workspace/models/ZhipuAI/GLM-4.6V-Flash \
  --data-dir data \
  --batch-size 1 \
  --eval-batch-size 1 \
  --checkpoint-interval 20 \
  --smoke-test

python offline/train.py \
  --annotation-run-id <run_id> \
  --model glm46v \
  --model-path /mnt/workspace/models/ZhipuAI/GLM-4.6V-Flash \
  --data-dir data \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 \
  --epochs 3 \
  --eval-batch-size 1 \
  --run-tag exp-glm46v \
  --num-workers 4 \
  --checkpoint-interval 20
```

### 训练参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--batch-size` | 1 | micro-batch |
| `--gradient-accumulation-steps` | 16 | 等效批大小 = `--batch-size` × 本值 |
| `--learning-rate` | 1e-4 | |
| `--epochs` | 3 | |
| `--eval-batch-size` | 1 | 与 `--batch-size` 保持一致：训练与验证共用一种张量形状 |
| `--best-metric` | `acc_at_0_5` | `acc_at_0_5` / `mean_iou` 越大越好，`val_loss` 越小越好 |
| `--max-pixels` | 3072×28×28 | 单帧视觉 token 预算；显存不足时优先下调，会改变 run id |

### GroundingDINO-B

GroundingDINO 走零样本推理，不执行 `offline/train.py`，也不需要 LoRA。

## 6. 推理

```text
--data-dir data
--num-shards 1
--num-workers 2
--batch-size 2
--batch-save 100
```

加 `--limit 100` 先冒烟，再去掉全量。验证集评测把 `--test-json` 指向
`outputs/annotations/<run_id>/val/approved.json` 并加 `--annotation-run-id <run_id>`，
会打印 ACC@0.5 / 平均 IoU / 解析失败数；测试集用官方模板
`data/Test/queries/queries.json`。GroundingDINO 用 `--num-workers 4 --batch-size 8`
且不传 `--lora-path`。

### Qwen3-VL-8B

```bash
python offline/infer.py \
  --model qwen3vl \
  --model-path /mnt/workspace/models/Qwen/Qwen3-VL-8B-Instruct \
  --lora-path outputs/output_lora/<train_run_id>/best/epoch_03 \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --limit 100 --num-shards 1 --num-workers 2 \
  --batch-size 2 --batch-save 100 --run-tag qwen3vl-smoke

python offline/infer.py \
  --model qwen3vl \
  --model-path /mnt/workspace/models/Qwen/Qwen3-VL-8B-Instruct \
  --lora-path outputs/output_lora/<train_run_id>/best/epoch_03 \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 2 \
  --batch-size 2 --batch-save 100 --run-tag qwen3vl-full
```

### Qwen3.5-9B

```bash
python offline/infer.py \
  --model qwen3_5 \
  --model-path /mnt/workspace/models/Qwen/Qwen3.5-9B \
  --lora-path outputs/output_lora/<train_run_id>/best/epoch_03 \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --limit 100 --num-shards 1 --num-workers 2 \
  --batch-size 2 --batch-save 100 --run-tag qwen35-smoke

python offline/infer.py \
  --model qwen3_5 \
  --model-path /mnt/workspace/models/Qwen/Qwen3.5-9B \
  --lora-path outputs/output_lora/<train_run_id>/best/epoch_03 \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 2 \
  --batch-size 2 --batch-save 100 --run-tag qwen35-full
```

### MiMo-VL-7B-RL

```bash
python offline/infer.py \
  --model mimo_vl \
  --model-path /mnt/workspace/models/XiaomiMiMo/MiMo-VL-7B-RL \
  --lora-path outputs/output_lora/<train_run_id>/best/epoch_03 \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --limit 100 --num-shards 1 --num-workers 2 \
  --batch-size 2 --batch-save 100 --run-tag mimo-smoke

python offline/infer.py \
  --model mimo_vl \
  --model-path /mnt/workspace/models/XiaomiMiMo/MiMo-VL-7B-RL \
  --lora-path outputs/output_lora/<train_run_id>/best/epoch_03 \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 2 \
  --batch-size 2 --batch-save 100 --run-tag mimo-full
```

### GLM-4.6V-Flash

```bash
python offline/infer.py \
  --model glm46v \
  --model-path /mnt/workspace/models/ZhipuAI/GLM-4.6V-Flash \
  --lora-path outputs/output_lora/<train_run_id>/best/epoch_03 \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --limit 100 --num-shards 1 --num-workers 2 \
  --batch-size 2 --batch-save 100 --run-tag glm46v-smoke

python offline/infer.py \
  --model glm46v \
  --model-path /mnt/workspace/models/ZhipuAI/GLM-4.6V-Flash \
  --lora-path outputs/output_lora/<train_run_id>/best/epoch_03 \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 2 \
  --batch-size 2 --batch-save 100 --run-tag glm46v-full
```

### GroundingDINO-B（Zero-shot）

```bash
python offline/infer.py \
  --model groundingdino \
  --model-path /mnt/workspace/models/IDEA-Research/grounding-dino-base \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --limit 100 --num-shards 1 --num-workers 4 \
  --batch-size 8 --batch-save 100 --run-tag dino-smoke

python offline/infer.py \
  --model groundingdino \
  --model-path /mnt/workspace/models/IDEA-Research/grounding-dino-base \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 4 \
  --batch-size 8 --batch-save 100 --run-tag dino-full
```

## 7. 融合与提交包

```bash
python -m aicomp_grounding.fusion.wbf \
  --predictions outputs/inference/<run_a>/predictions.json \
                outputs/inference/<run_b>/predictions.json \
  --weights 1.0 1.0 --iou-threshold 0.55 \
  --test-json data/Test/queries/queries.json \
  --output-dir outputs/fusion

python -m aicomp_grounding.submission \
  --test-json data/Test/queries/queries.json \
  --predictions outputs/inference/<run_id>/predictions.json \
  --output-dir outputs/submission/<run_id>
```

`--scores` 可传各模型的分数文件（检测器用原生置信度做乘性加权），空字符串表示
该模型不计分数。`--allow-fallback` 仅用于显式不完整的诊断包：它会把无效框填成
占位框，正式提交不要使用。
