# cloud/ — Modal 执行入口

运行配置为 H100、8 CPU、32 GiB 内存，调用与离线端相同的训练/推理编排。
`modal` 命令由管理员执行。

## 部署说明

Modal 专为 Qwen3.6-27B 部署，该模型从 HuggingFace 直接下载（`Qwen/Qwen3.6-27B`），
需要 `[kernels]` 依赖（flash-linear-attention、causal-conv1d）以加速混合注意力计算。
训练需要 48GB 内存，推理需要 32GB。

## 模型与数据准备

```text
Volume aicomp 的根目录，运行时挂载到 /mnt/workspace：
  data/
  models/Qwen/Qwen3.6-27B/
  outputs/annotations/<run_id>/
  outputs/output_lora/<run_id>/
  outputs/inference/<run_id>/
```

模型下载（需要 HF Token）：

```bash
huggingface-cli download Qwen/Qwen3.6-27B --local-dir ./Qwen3.6-27B
```

上传到 Modal Volume：

```bash
modal volume create aicomp
modal volume put aicomp ./Qwen3.6-27B /models/Qwen/Qwen3.6-27B
```

训练/推理通过 `/mnt/workspace/models/Qwen/Qwen3.6-27B` 读取。
数据和标注按上表放入 Volume。训练/推理不自动下载或补齐模型文件。

## 训练

```bash
modal run cloud/train.py --model qwen36_27b \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model-path /mnt/workspace/models/Qwen/Qwen3.6-27B --smoke-test

modal run cloud/train.py --model qwen36_27b \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model-path /mnt/workspace/models/Qwen/Qwen3.6-27B \
  --run-tag modal-qwen36-train-auditfix
```

## 推理

```bash
modal run cloud/infer.py --model qwen36_27b \
  --model-path /mnt/workspace/models/Qwen/Qwen3.6-27B \
  --lora-path /mnt/workspace/outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --limit 100 --batch-size 2 --run-tag modal-qwen36-smoke-auditfix

modal run cloud/infer.py --model qwen36_27b \
  --model-path /mnt/workspace/models/Qwen/Qwen3.6-27B \
  --lora-path /mnt/workspace/outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --batch-size 2 --run-tag modal-qwen36-full-auditfix
```

## 执行约束

数据预检可在本地用 `offline/train.py --preflight-only`。通过 cloud 入口运行预检
仍会分配 H100。中断后维持原路径和标签继续；改参数或解析行为时使用新标签。
checkpoint 写入后调用 Volume commit；推理超时 8 小时，训练超时 24 小时。

公共镜像直接读取根 `pyproject.toml` 的 dependencies，不复制另一份依赖列表。
各模型 GPU 验证进度只记录在 `docs/handoff.md`，注册适配器不代表平台已完成验证。
