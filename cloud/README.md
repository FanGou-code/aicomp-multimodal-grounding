# cloud/ — Modal 执行入口

运行配置为 H100、8 CPU、32 GiB 内存，调用与离线端相同的训练/推理编排。
`modal` 命令由管理员执行。

## 模型与数据准备

```text
Volume aicomp 的根目录，运行时挂载到 /mnt/workspace：
  data/
  models/<org>/<name>/
  outputs/annotations/<run_id>/
  outputs/output_lora/<run_id>/
  outputs/inference/<run_id>/
```

模型先按 `docs/sop.md` 中的来源和版本下载，再上传完整目录。Volume CLI 的目标
路径相对 Volume 根；它不包含运行时的挂载前缀 `/mnt/workspace`。

```bash
modal volume create aicomp
modal volume put aicomp LOCAL_MODEL_DIR /models/Qwen/Qwen3-VL-8B-Instruct
```

上传后，程序通过 `/mnt/workspace/models/Qwen/Qwen3-VL-8B-Instruct` 读取。
数据和标注按上表放入 Volume。训练/推理不自动下载或补齐模型文件。

## 推理

```bash
modal run cloud/infer.py --model qwen3vl \
  --model-path /mnt/workspace/models/Qwen/Qwen3-VL-8B-Instruct \
  --lora-path /mnt/workspace/outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --limit 100 --batch-size 2 --run-tag modal-smoke-auditfix

modal run cloud/infer.py --model qwen3vl \
  --model-path /mnt/workspace/models/Qwen/Qwen3-VL-8B-Instruct \
  --lora-path /mnt/workspace/outputs/output_lora/YOUR_RUN_ID/best/epoch_XX \
  --batch-size 2 --run-tag modal-full-auditfix
```

## 训练

```bash
modal run cloud/train.py --model qwen3vl \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model-path /mnt/workspace/models/Qwen/Qwen3-VL-8B-Instruct --smoke-test

modal run cloud/train.py --model qwen3vl \
  --annotation-run-id YOUR_ANNOTATION_RUN_ID \
  --model-path /mnt/workspace/models/Qwen/Qwen3-VL-8B-Instruct \
  --run-tag modal-train-auditfix
```

数据预检可在本地用 `offline/train.py --preflight-only`。通过 cloud 入口运行预检
仍会分配 H100。中断后维持原路径和标签继续；改参数或解析行为时使用新标签。
checkpoint 写入后调用 Volume commit；推理超时 8 小时，训练超时 24 小时。

## 依赖边界

公共镜像直接读取根 `pyproject.toml` 的 dependencies，不复制另一份依赖列表。
Youtu 的原生代码需要独立兼容环境，公共镜像没有提供该环境。各模型 GPU 验证
进度只记录在 `docs/handoff.md`，注册适配器不代表平台已完成验证。
