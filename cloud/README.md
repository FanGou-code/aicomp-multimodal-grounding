# ☁️ Cloud End — Modal

云端端：Modal 专用适配器。通用实验室电脑、魔搭工作台或其他真实 GPU
环境请使用仓库根目录下的 `offline/` 入口；本目录只负责把同一套核心逻辑接入
Modal 的远程容器、Volume 和分片调度。Modal 账号、Volume 名称和登录状态不属于
仓库身份。

## 入口

| 脚本 | 用途 |
| --- | --- |
| `cloud/train.py` | H100 LoRA 训练（支持 preflight / smoke / 断点续跑） |
| `cloud/infer.py` | H100 批量推理（Val 评估 / 官方 Test + 8 卡分片 + 自动打包） |

所有命令**在仓库根目录**执行：

```bash
# 训练（先 preflight 再正式）
modal run cloud/train.py --annotation-run-id annot_ac72f1d926bb2d23 --model qwen3vl --run-tag exp-h100 --preflight-only
modal run cloud/train.py --annotation-run-id annot_ac72f1d926bb2d23 --model qwen3vl --run-tag exp-h100

# 验证集评估（Modal 的本地入口；使用 approved 标注）
modal run cloud/infer.py --split val --adapter-path /data/data/output_lora/<RUN_ID>/best/epoch_XX

# 官方测试集 8 卡分片推理 + 自动生成 submission.zip；worker 使用 data/test.json
modal run cloud/infer.py --split test --num-shards 8 --adapter-path /data/data/output_lora/<RUN_ID>/best/epoch_XX

# 其他模型（--model 换适配器；LoRA 可选）
modal run cloud/infer.py --model internvl35 --split val --adapter-path ""
modal run cloud/infer.py --model groundingdino --split test --num-shards 8 --adapter-path ""
```

## 计费要点（2026-08 官网单价实测）

- H100 SXM5 ≈ $3.95/h（参考价，请以 Modal 当前计费页为准）
- CPU ≈ $0.0472/核·h、内存 ≈ $0.008/GiB·h，占比很小但计入账单
- **不要选 region**（+50~75%）、**不要开 non-preemptible**（3×）
- Volume 存储 1 TiB/月免费，数据集（~43G）远在额度内

Modal Volume 内部的 `/data/data` 是本适配器的历史兼容布局。它不应被复制到
`offline/` 或其他云工作台；整仓迁移时以仓库根目录为项目根，并使用
`offline/train.py` / `offline/infer.py`。
