# 🔄 Iteration 02: Conservative Optimization (保守优化)

---

## 一、基线（实际成绩）

* **唯一完整跑通的一套**：`train_9e468a454061153b`（Qwen3-VL-8B-Instruct + LoRA）
* **竞赛成绩 ACC@0.5**：**0.7439**（官方测试集）

---

## 二、本轮已落地改动（尚未训练）

### 1. 训练策略

* LoRA alpha 32 → **48**
* 训练轮数 2 → **3**
* cosine 调度加入 **min_lr = 1e-5 下限**（基线已用 cosine，但会衰减到 0）
* `total_steps` 强制取整，避免浮点值进入 `range()`

### 2. 硬件

* 训练卡 A100-80GB（64GB RAM）→ **H100**（32GB RAM）
* 梯度检查点开启 `use_reentrant=False`（非重入）

### 3. 推理基础设施

* 新增 `cloud/infer.py`（原 `infer_modal.py`）生产级推理引擎（Batch-4 + `--num-shards 8` 分片 + 自动构建提交包）

### 4. 已评估并剔除的方向

* **图像滤镜增强**（CLAHE / 双边滤波）：破坏预训练特征分布，放弃。
* **文本标准化 `standardize_query`**：实测仅覆盖 2.4% query，收益接近零，已从仓库剔除。
* **极小目标外扩 `calibrate_bbox`（3% padding）**：未经验证的启发式，且会扭曲框几何、干扰后续 WBF 多模型融合，已从仓库剔除。
* **Selective retry（强约束 prompt 重跑解析失败样本）**：与 `calibrate_bbox` 绑定，一并剔除。

---

## 三、当前配置快照

* `MAX_PIXELS = 3072 * 28 * 28`（1080p 无损输入）
* `MODEL_REVISION = 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
* 训练数据：原 split（`annot_ac72f1d926bb2d23`，2875 Train / 719 Val）

---

## 四、状态

**尚未开始训练**。上述改动已落地 `main`，下一步启动 H100 训练并评估 Val 与官方测试集。

团队分工：本仓库负责 Qwen3-VL 单模型（融合锚点）；队友分别负责 InternVL-3.5 与 GroundingDINO 异构模型，最终做 WBF 加权框融合。
