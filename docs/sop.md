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
```

数据包已包含生成好的 `Processed/`，解压后直接使用。另将赛事渠道取得的
Test 目录放入 `data/Test/`，检查布局：

```bash
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
# 应输出 5.15.1
```

目标模型为混合线性注意力架构（`qwen3_5` 类）时，额外安装加速内核：

- **N 卡 / CUDA 机**：一键安装（FLA 与 causal-conv1d 均有预编译轮）：

```bash
pip install -e ".[kernels]" \
  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

- **A 卡（DSW）**：先源码编译 `causal-conv1d`，再安装 `[kernels]`：

```bash
export PYTORCH_ROCM_ARCH=gfx942 MAX_JOBS=8
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
pip install causal-conv1d --no-build-isolation \
  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

pip install -e ".[kernels]" \
  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

python -c "import fla, causal_conv1d; print(fla.__version__, causal_conv1d.__version__)"
```

**venv 在持久盘 `/mnt/workspace` 上，同镜像代际的实例间直接复用，无需重建。**
仅当持久盘被清空、或镜像大版本更换导致底座 python 路径变化时，按「首次执行」
重走一遍即可（pip 与 Triton 缓存持久，重装为秒级）。

## 4. 下载模型（首次执行）

训练和推理不自动下载底座。先完成下载，再传 `--model-path`；缺文件时程序报错。
每条命令都带 `--revision`，取值与适配器里的 `MODEL_REVISION` 相同：身份字符串
和磁盘上的字节指向同一个快照。`qwen36_27b` 从 HF 下载，见 `cloud/README.md`。

```bash
export MODEL_ROOT=/mnt/workspace/models
mkdir -p "$MODEL_ROOT"

modelscope download --model Qwen/Qwen3-VL-8B-Instruct \
  --revision 5d854aab08710c16b980ec6d603d863b3821b915 \
  --local_dir "$MODEL_ROOT/Qwen/Qwen3-VL-8B-Instruct"

modelscope download --model OpenGVLab/InternVL3_5-8B-HF \
  --revision 1c352b29d4066a61b465b5c6d044a1ebec1349ef \
  --local_dir "$MODEL_ROOT/OpenGVLab/InternVL3_5-8B-HF"

modelscope download --model IDEA-Research/grounding-dino-base \
  --revision d06985a44c66b6133c131bd273293be8649cfe3a \
  --local_dir "$MODEL_ROOT/IDEA-Research/grounding-dino-base"

modelscope download --model Qwen/Qwen3.5-9B \
  --revision 460979c3d11864dd16408d860ac930a360a2fac2 \
  --local_dir "$MODEL_ROOT/Qwen/Qwen3.5-9B"

modelscope download --model ZhipuAI/GLM-4.6V-Flash \
  --revision a4ec61fcdfab32bbccdf26c5ca8cb5a437b7ca41 \
  --local_dir "$MODEL_ROOT/ZhipuAI/GLM-4.6V-Flash"

modelscope download --model XiaomiMiMo/MiMo-VL-7B-RL \
  --revision d307865d4a3b6ad9ae35e574bcabaa563038c8fb \
  --local_dir "$MODEL_ROOT/XiaomiMiMo/MiMo-VL-7B-RL"
```

下载前检查持久盘空间：

```bash
df -h /mnt/workspace
```

## 5. 训练

训练命令统一使用 `offline/train.py`。先跑 smoke，再启动完整训练。
仅检查数据合同可使用 `--preflight-only`，不会加载权重或写训练计划。
更改代码中的目标格式、解析或参数后，下面的示例标签应改为新标签，不混入旧结果。
本节覆盖 DSW 上可训练的五个适配器；`qwen36_27b`（Qwen3.6-27B）只在 Modal 上运行，
镜像、权重与 Volume 布局见 `cloud/README.md`。

### 显式超参数（命令行覆盖）

训练超参数已解耦为显式 CLI 参数，默认值均为 `None`（不传时沿用适配器基准默认值）。
传参会自动覆盖并计入 `training_run_id` 哈希指纹，天然分流到新 run 目录：

| CLI 参数 | 默认值 | 对应超参键 | 作用说明 |
| --- | --- | --- | --- |
| `--batch-size` | `None` (默认 1) | `batch_size` | 训练 micro-batch 样本数（亦作为 val loss 批大小） |
| `--gradient-accumulation-steps` | `None` (默认 16) | `gradient_accumulation_steps` | 梯度累积步数（等效 batch = batch_size × accum） |
| `--learning-rate` | `None` (默认 1e-4) | `learning_rate` | 初始学习率 |
| `--epochs` | `None` (默认 3) | `epochs` | 训练轮数 |
| `--eval-batch-size` | `None` (默认 1) | `eval_batch_size` | 验证集生成评测（ACC@0.5）批量 |
| `--best-metric` | `None` (默认 `acc_at_0_5`) | `best_epoch_primary_metric` | 最佳检查点判定指标（可选 `acc_at_0_5` / `mean_iou` / `val_loss`） |

**冒烟口径**：
- 冒烟测试仅执行 1 个训练样本（前向 + 反向 + 优化器步）与 1 个验证样本（前向 val loss），
  全程保持单一张量形状（`batch_size=1`），避免多形状重复 JIT 编译；
- 执行期间有显式阶段日志（`[smoke] Running training forward + backward...`、
  `[smoke] Running validation loss forward...`、`[smoke] One-batch verification finished.`）；
- 最终打印 `Training smoke passed: train_loss=..., val_loss=...` 即为成功闭环；
- 冒烟命令不带 `--num-workers`。

### Qwen3-VL-8B

```bash
python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model qwen3vl \
  --model-path /mnt/workspace/models/Qwen/Qwen3-VL-8B-Instruct \
  --data-dir data \
  --batch-size 1 \
  --eval-batch-size 1 \
  --checkpoint-interval 20 \
  --smoke-test

python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model qwen3vl \
  --model-path /mnt/workspace/models/Qwen/Qwen3-VL-8B-Instruct \
  --data-dir data \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 \
  --epochs 3 \
  --eval-batch-size 1 \
  --run-tag exp-qwen8-retrain-auditfix \
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
  --batch-size 1 \
  --eval-batch-size 1 \
  --checkpoint-interval 20 \
  --smoke-test

python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model qwen3_5 \
  --model-path /mnt/workspace/models/Qwen/Qwen3.5-9B \
  --data-dir data \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 \
  --epochs 3 \
  --eval-batch-size 1 \
  --run-tag exp-qwen35-9b-auditfix \
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
  --batch-size 1 \
  --eval-batch-size 1 \
  --checkpoint-interval 20 \
  --smoke-test

python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model glm46v \
  --model-path /mnt/workspace/models/ZhipuAI/GLM-4.6V-Flash \
  --data-dir data \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 \
  --epochs 3 \
  --eval-batch-size 1 \
  --run-tag exp-glm46v-flash-auditfix \
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
  --batch-size 1 \
  --eval-batch-size 1 \
  --checkpoint-interval 20 \
  --smoke-test

python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model internvl35 \
  --model-path /mnt/workspace/models/OpenGVLab/InternVL3_5-8B-HF \
  --data-dir data \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 \
  --epochs 3 \
  --eval-batch-size 1 \
  --run-tag exp-internvl-01-auditfix \
  --num-workers 4 \
  --checkpoint-interval 20
```

### MiMo-VL-7B-RL

```bash
python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model mimo_vl \
  --model-path /mnt/workspace/models/XiaomiMiMo/MiMo-VL-7B-RL \
  --data-dir data \
  --batch-size 1 \
  --eval-batch-size 1 \
  --checkpoint-interval 20 \
  --smoke-test

python offline/train.py \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model mimo_vl \
  --model-path /mnt/workspace/models/XiaomiMiMo/MiMo-VL-7B-RL \
  --data-dir data \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 \
  --epochs 3 \
  --eval-batch-size 1 \
  --run-tag exp-mimo-vl-auditfix \
  --num-workers 4 \
  --checkpoint-interval 20
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
--batch-size 2
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
  --batch-size 2 --batch-save 100 --run-tag qwen8-lora-smoke-auditfix

python offline/infer.py \
  --model qwen3vl \
  --model-path /mnt/workspace/models/Qwen/Qwen3-VL-8B-Instruct \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 2 \
  --batch-size 2 --batch-save 100 --run-tag qwen8-lora-full-auditfix
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
  --batch-size 2 --batch-save 100 --run-tag qwen35-9b-lora-smoke-auditfix

python offline/infer.py \
  --model qwen3_5 \
  --model-path /mnt/workspace/models/Qwen/Qwen3.5-9B \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 2 \
  --batch-size 2 --batch-save 100 --run-tag qwen35-9b-lora-full-auditfix
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
  --batch-size 2 --batch-save 100 --run-tag glm46v-lora-smoke-auditfix

python offline/infer.py \
  --model glm46v \
  --model-path /mnt/workspace/models/ZhipuAI/GLM-4.6V-Flash \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 2 \
  --batch-size 2 --batch-save 100 --run-tag glm46v-lora-full-auditfix
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
  --batch-size 2 --batch-save 100 --run-tag internvl-lora-smoke-auditfix

python offline/infer.py \
  --model internvl35 \
  --model-path /mnt/workspace/models/OpenGVLab/InternVL3_5-8B-HF \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 2 \
  --batch-size 2 --batch-save 100 --run-tag internvl-lora-full-auditfix
```

### MiMo-VL-7B-RL

微调后：

```bash
python offline/infer.py \
  --model mimo_vl \
  --model-path /mnt/workspace/models/XiaomiMiMo/MiMo-VL-7B-RL \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --limit 100 --num-shards 1 --num-workers 2 \
  --batch-size 2 --batch-save 100 --run-tag mimo-lora-smoke-auditfix

python offline/infer.py \
  --model mimo_vl \
  --model-path /mnt/workspace/models/XiaomiMiMo/MiMo-VL-7B-RL \
  --lora-path outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 2 \
  --batch-size 2 --batch-save 100 --run-tag mimo-lora-full-auditfix
```

### GroundingDINO-B（Zero-shot）

```bash
python offline/infer.py \
  --model groundingdino \
  --model-path /mnt/workspace/models/IDEA-Research/grounding-dino-base \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --limit 100 --num-shards 1 --num-workers 4 \
  --batch-size 8 --batch-save 100 --run-tag dino-smoke-auditfix

python offline/infer.py \
  --model groundingdino \
  --model-path /mnt/workspace/models/IDEA-Research/grounding-dino-base \
  --test-json data/Test/queries/queries.json \
  --data-dir data \
  --num-shards 1 --num-workers 4 \
  --batch-size 8 --batch-save 100 --run-tag dino-full-auditfix
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
