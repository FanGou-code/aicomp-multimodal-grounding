# 交接文档

> 全仓库唯一的状态与交接记录：做到哪了、成绩、下一步。每次交接或阶段性
> 完成时更新「当前状态」并在「交接日志」追加一条（新的写最上面）。
> 结构与代码约定见 `architecture.md`，调研背景见 `research.md`。

## 当前状态（最后更新 2026-08-18）

- **基线（唯一完整成绩）**：`train_9e468a454061153b`（Qwen3-VL-8B + LoRA），
  官方测试集 ACC@0.5 = **0.7439**
- **Iteration 02 代码全部就绪、尚未训练**：训练侧升级 + 推理引擎 + 三模型
  adapter + WBF 融合已落地 main，测试 165 全绿
- **下一步（维护者）**：新 Modal 账号就绪后
  `modal run cloud/train.py --preflight-only` → `--smoke-test` → 正式 H100 训练
  （预算约 $14.5 / 3.3h；超参未动，run id 与历史一致）
- **等待队友**：InternVL / GroundingDINO zero-shot 首跑——GPU 路径未冒烟，
  先 `--limit 100` 小切片验证；正式 run 前必须 pin `model_revision`
  （当前为 main，adapter 内有 TODO 标记）
- **分工**：维护者负责 Qwen3-VL 单模型与统一仓库；队友各自负责
  InternVL-3.5 与 GroundingDINO（自筹 Modal/魔搭算力），最终 WBF 加权框融合

## Iteration 02 改动明细（已落地 main，尚未训练）

### 训练策略

* LoRA alpha 32 → **48**
* 训练轮数 2 → **3**
* cosine 调度加入 **min_lr = 1e-5 下限**（基线已用 cosine，但会衰减到 0）
* `total_steps` 强制取整，避免浮点值进入 `range()`

### 硬件

* 训练卡 A100-80GB（64GB RAM）→ **H100**（32GB RAM）
* 梯度检查点开启 `use_reentrant=False`（非重入）

### 推理与仓库基础设施

* `cloud/infer.py`（原 `infer_modal.py`）生产级推理引擎
  （Batch-4 + `--num-shards 8` 分片 + 自动构建提交包）
* 双端布局：`cloud/`（Modal 壳）+ `offline/`（离线壳）；训练/推理核心下沉
  `aicomp_grounding/{training_core,inference_core}.py`，两端共用
* `models/` 适配层：qwen3vl（参考实现）/ internvl35 / groundingdino / mock，
  两个推理入口 `--model` 切换
* `fusion/wbf.py`：多模型加权框融合，CLI 可直出提交包
* 超参与 Qwen identity 一字未动 → training run id 与历史一致
  （`tests/test_models.py` 钉死防漂移）

### 已评估并剔除的方向

* **图像滤镜增强**（CLAHE / 双边滤波）：破坏预训练特征分布，放弃
* **文本标准化 `standardize_query`**：实测仅覆盖 2.4% query，收益接近零，剔除
* **极小目标外扩 `calibrate_bbox`**（3% padding）：未经验证的启发式，且会
  扭曲框几何、干扰 WBF，剔除
* **Selective retry**：与 `calibrate_bbox` 绑定，一并剔除

### 配置快照

* `MAX_PIXELS = 3072 * 28 * 28`（1080p 无损输入）、
  `MODEL_REVISION = 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` 等模型 identity
  在 `aicomp_grounding/models/qwen3vl.py`（值不变）
* 训练数据：原 split（`annot_ac72f1d926bb2d23`，2875 Train / 719 Val）

## 交接日志（追加式，新的写最上面）

### 2026-08-18（维护者）

* 剔除未验证启发式（`calibrate_bbox` / `standardize_query` / selective retry）
* 仓库重构完成：双端布局 + 核心下沉 + `models/` 适配层 + `fusion/wbf.py`
  + `docs/architecture.md`；测试 165 全绿、0 跳过
* 文档体系定型：README（用法）/ architecture（结构约定）/ handoff（本文，
  状态与交接）/ research（调研）；删除 `offline/dsw/` 预设目录，
  DSW 环境安装命令内联进 `offline/README.md`
* 待办：Iteration 02 训练（等新 Modal 账号）；队友模型首跑冒烟

### 2026-08 前期（基线，追记）

* `train_9e468a454061153b` 完整训练 + 推理，官方测试集 ACC@0.5 = 0.7439
