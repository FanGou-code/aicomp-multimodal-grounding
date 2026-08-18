# ☁️ Cloud End — Modal

云端端：付费弹性算力（训练 + 大批量推理）。Modal 免费账号每月 $30 额度、10 GPU 并发上限。

## 入口

| 脚本 | 用途 |
| --- | --- |
| `cloud/train.py` | H100 LoRA 训练（支持 preflight / smoke / 断点续跑） |
| `cloud/infer.py` | H100 批量推理（Val 评估 / 官方 Test + 8 卡分片 + 自动打包） |

所有命令**在仓库根目录**执行：

```bash
# 训练（先 preflight 再正式）
modal run cloud/train.py --annotation-run-id annot_ac72f1d926bb2d23 --run-tag exp-h100 --preflight-only
modal run cloud/train.py --annotation-run-id annot_ac72f1d926bb2d23 --run-tag exp-h100

# 验证集评估
modal run cloud/infer.py --split val --adapter-path /data/data/output_lora/<RUN_ID>/best/epoch_XX

# 官方测试集 8 卡分片推理 + 自动生成 submission.zip
modal run cloud/infer.py --split test --num-shards 8 --adapter-path /data/data/output_lora/<RUN_ID>/best/epoch_XX
```

## 计费要点（2026-08 官网单价实测）

- H100 SXM5 ≈ $3.95/h（训练最优卡：A100-80 实测总价更贵且易被抢占）
- CPU ≈ $0.0472/核·h、内存 ≈ $0.008/GiB·h，占比很小但计入账单
- **不要选 region**（+50~75%）、**不要开 non-preemptible**（3×）
- Volume 存储 1 TiB/月免费，数据集（~43G）远在额度内
