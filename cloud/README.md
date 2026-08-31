# cloud/ — Modal 壳（H100 / 单卡）

硬编码即全部：**H100、8 核、32GiB 内存**（Iteration 02 实测够用的配置）；
超参在适配器里、推理参数用 offline 默认值——本目录没有任何业务逻辑。

## Volume 布局（与魔搭 /mnt/workspace 契约对齐）

```text
Volume "aicomp" 挂载于 /mnt/workspace
  data/                 数据集（推理只需 Test/ + Processed/Test/，训练需 Train/）
  models/<org>/<name>/  权重（可选：不传则 qwen3vl/internvl 从 HF 快速下载）
  outputs/
    annotations/<run_id>/      approved.json（训练输入）
    output_lora/<run_id>/      训练产物
    inference/<run_id>/        预测/提交包
```

首次准备：

```bash
modal volume create aicomp            # 或让首次运行自动创建
modal volume put aicomp data.tar /mnt/workspace/data.tar   # 上传后可在任务内解压
modal volume put aicomp <本地权重目录> /mnt/workspace/models/<org>/<name>
```

## 推理

```bash
# 冒烟（100 条）
modal run cloud/infer.py --model qwen3vl \
  --lora-path /mnt/workspace/outputs/output_lora/<RUN_ID>/best/epoch_XX \
  --limit 100 --run-tag modal-smoke

# 全量 Test（自动打包 submission.zip）
modal run cloud/infer.py --model qwen3vl \
  --lora-path /mnt/workspace/outputs/output_lora/<RUN_ID>/best/epoch_XX \
  --run-tag modal-full
```

- 模型/LoRA 路径全部以 `/mnt/workspace` 开头（Volume 内）
- `--model-path` 省略时从 HuggingFace 下载（Modal 机房快）；16B/27B 级权重请先传 Volume

## 训练

```bash
# CPU 预检（不占 GPU 时长也挂 H100，秒级结束）
modal run cloud/train.py --annotation-run-id annot_dc189f029d962b27 --preflight-only

# 单 batch 冒烟
modal run cloud/train.py --annotation-run-id annot_dc189f029d962b27 --smoke-test

# 全量训练（3 epochs；断点/抢占自动恢复，每次 checkpoint 提交 Volume）
modal run cloud/train.py --annotation-run-id annot_dc189f029d962b27 --run-tag exp-<name>
```

## 费用与护栏

- H100 ≈ $4.5/h + CPU/内存费；全量 Test 推理一趟约 $25-40，**训练 16B 级超出 $30 月额度**——付费训练仅限 8B 级短跑
- 护栏 = 函数 timeout（推理 8h / 训练 24h）+ 断点续跑；费用看 Modal 面板 Usage 页

## 新模型接入（不动核心文件）

`aicomp_grounding/models/` 加适配器 → `models/__init__.py` 注册一行 →
`envs/gpu.txt` 补依赖（如需）→ `--model <name>` 在 cloud/infer、cloud/train、
offline 三端同时可用。
