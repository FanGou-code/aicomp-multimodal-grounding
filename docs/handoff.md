# 交接文档

> 全仓库唯一的状态与交接记录：做到哪了、成绩、下一步。每次交接或阶段性
> 完成时更新「当前状态」并在「交接日志」追加一条（新的写最上面）。
> 结构与代码约定见 `architecture.md`，调研背景见 `research.md`。
> 交接日志保留历史记录，不作为当前状态结论；当前状态以上方最新条目为准。

## 当前状态（最后更新 2026-08-22）

- **最新进展**：魔搭 AMD MI300X 192G 已完成一轮训练，当前进入测试集推理阶段；
  离线推理与训练 I/O 已做单卡优化并推送（175 测试全绿）。
- **离线推理推荐**：单卡固定 `--num-shards 1 --num-workers 4`，由 DataLoader
  预取图像与 GPU 推理并行；192GB 首轮 Qwen/InternVL 用 `--batch-size 8`，
  GroundingDINO 用 `--batch-size 32`。冒烟与全量同参数，`--limit 100` 通过后
  去掉 limit 直接全量，OOM 时回退到 batch 4 / 16。
- **训练 I/O 调优**：DataLoader `num_workers=4`、`persistent_workers=True`，
  step checkpoint 从每 20 步改为每 50 步；不影响训练指标、随机性或 run id。
- **GPU 检测**：推理期间用 `watch -n 1 rocm-smi --showuse --showmemuse`
  观察利用率和显存，判断是否还有提升空间。
- **Obsidian SOP 已同步**：推理命令、WBF 输出文件名、`scores.json` 说明、
  单卡 batch 推荐均已与当前仓库代码对齐。

- **基线（完整成绩）**：`train_89aa55f31aee5478`（Qwen3-VL-8B + LoRA），
  测试集 ACC@0.5 = **0.7439**
- **Iteration 02 训练已推进**：训练侧升级 + 推理引擎 + 三模型 adapter +
  WBF 融合已落地 main；当前在魔搭单卡 MI300X 上执行训练/推理。
- **训练核心已适配多模型**：`training_core.py` 由 adapter 驱动，支持
  `qwen3vl` 与 `internvl35`；每轮训练在全量验证集上计算 ACC/mIoU 并用于 best epoch。
- **离线推理已支持 DataLoader 预取**：`offline/infer.py --num-workers` 默认 4；
  多卡场景仍支持 `--num-shards`，单卡不建议使用多进程副本。
- **本地开发环境**：`qwen_vg` conda 环境使用 Python 3.12，本地 CPU
  校验依赖按 `requirements-lock.txt` 安装；真实 GPU 训练/推理使用 `offline/`。
- **运行边界已明确**：本仓库是唯一可移植实验单元；本地电脑只做 CPU 测试和静态检查，
  实验室电脑/新 GPU/魔搭工作台使用 `offline/`，`cloud/` 仅保留 Modal 适配。
- **训练产物路径已统一**：offline 默认写入 `outputs/output_lora/<id>/`，approved
  标注位于 `outputs/annotations/<id>/`；Modal 未传输出根时继续使用历史
  `/data/data/output_lora/<id>/` 布局，不改变 run id。
- **下一步**：在 MI300X 上按最新 SOP 执行 Qwen/InternVL/DINO `--limit 100`
  冒烟，通过后跑全量推理，再执行 WBF 与提交；Modal 账号恢复后才执行 `cloud/`。
- **待验证模型**：InternVL / GroundingDINO zero-shot 首跑——GPU 路径未冒烟，
  先 `--limit 100` 小切片验证；两者的 `model_revision` 已 pin 具体 commit。
- **模型覆盖**：Qwen3-VL 与 InternVL3.5 已接入训练循环，GroundingDINO
  保持 zero-shot 推理；三者预测结果最终进入 WBF 融合。

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
* Qwen 模型 identity 与训练超参未变；运行身份增加 Python runtime 版本，
  旧基线产物已按新环境身份重算 run id
  （`tests/test_models.py` 继续钉死 Qwen identity 防漂移）

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

### 2026-08-23（仓库，离线推理断点续跑支持 checkpoint.json 自动恢复）

* 修复 `offline/infer.py`：开启 `--resume` 时，若未生成全量 `predictions.json` 但存在阶段性 `checkpoint.json`，自动从中恢复已完成预测，实现单命令无缝断点续跑。
* 单元测试 175 项全绿。

### 2026-08-22（仓库，单卡 MI300X 推理与训练 I/O 优化）

* `offline/infer.py` 增加 `--num-workers`（默认 4）DataLoader 预取路径，
  单卡推理可与 GPU 并行加载图像；单卡不再建议 `--num-shards >1`。
* 删除推理循环中的 `torch.cuda.empty_cache()`，单卡路径补上中间 checkpoint。
* 训练 DataLoader 调整为 `num_workers=4`、`persistent_workers=True`，
  step checkpoint 从每 20 步调整为每 50 步；不影响训练指标与随机性。
* README / offline README / Obsidian SOP 已同步单卡 MI300X 推荐参数；
  验证 175 项测试通过。

### 2026-08-19（仓库，训练核心适配器化）

* 新增 `TrainableGroundingAdapter` 训练协议，Qwen3-VL 与 InternVL3.5 接入统一训练循环。
* 训练入口增加 `--model`，当前支持 `qwen3vl`、`internvl35`。
* best epoch 改为全量验证集 `ACC@0.5` 优先，`val_loss` 平局辅助。
* `offline/infer.py` 增加 `--num-shards` 本地多进程推理。
* 验证：`unittest discover -s tests` 175 项通过，`compileall` 与 `git diff --check` 通过。

### 2026-08-19（仓库，审查修复与旧产物重算）

* 固定 InternVL3.5 与 GroundingDINO 的模型 revision 为具体 commit。
* 修正离线环境安装说明，并移除测试中的 Pillow `getdata()` 弃用调用。
* 运行身份纳入 Python 3.12.13，旧 `train/infer/submission` 产物按新身份重算并迁移：
  `train_89aa55f31aee5478`、`infer_val_base_cf21ef82acde823b`、
  `infer_test_base_d1b8b06e5e1985e8`。
* 重建测试集提交包 `outputs/submission/infer_test_base_d1b8b06e5e1985e8/submission.zip`。

### 2026-08-19（仓库，本地环境迁移）

* 本地 `qwen_vg` conda 环境迁移到 Python 3.12，开发/CPU 校验依赖按
  `requirements-lock.txt` 安装；未执行 Modal 命令。
* 同步 README、offline/README、cloud/README 与配置注释中的 Python 版本和环境边界说明。
* 验证：`unittest discover -s tests` 共 **173 tests，全部通过**；
  `compileall` 与 `git diff --check` 通过。

### 2026-08-18（仓库，仓库可移植性与路径修复）

* 明确 portable repo / offline 主入口 / Modal adapter 边界：整个仓库作为唯一实验单元复制到
  实验室电脑、GPU 工作台或云端环境，安装依赖后使用 `offline/` 训练和推理。
* 新增 `aicomp_grounding.paths.ProjectPaths`，统一项目根、数据、approved 标注、推理/融合和提交模板路径。
* 修复 `cloud/infer.py` 的 Val/Test 路径：`data/test.json` 仅作为 worker 索引，官方
  `data/Test/queries/queries.json` 仅用于 submission；Val 不再静默 fallback 到空 query 索引。
* 修复 GroundingDINO 后处理参数名、WBF 得分文件重复读盘和得分指纹缺失，并修正文案中的历史硬件名称。
* `offline/train.py` 增加 `--project-root`、`--annotation-root`、`--output-root`；训练核心显式支持
  仓库级输出根，同时保留 Modal 历史默认布局。
* 验证：`unittest discover -s tests` 共 **173 tests，全部通过**；`compileall` 与 `git diff --check`
  通过；未执行任何 Modal 命令。
* 残余风险：未在真实 GPU 上验证 Qwen/InternVL/GroundingDINO 前向；AMD ROCm 尚未冒烟；InternVL
  与 GroundingDINO 的 `model_revision="main"` 仍待正式运行前根据最终模型源 pin 具体 commit。

### 2026-08-18（仓库）

* 剔除未验证启发式（`calibrate_bbox` / `standardize_query` / selective retry）
* 仓库重构完成：双端布局 + 核心下沉 + `models/` 适配层 + `fusion/wbf.py`
  + `docs/architecture.md`；测试 165 全绿、0 跳过
* 文档体系定型：README（用法）/ architecture（结构约定）/ handoff（本文，
  状态与交接）/ research（调研）；删除 `offline/dsw/` 预设目录，
  DSW 环境安装命令内联进 `offline/README.md`
* 待办：Iteration 02 训练（等新 Modal 账号）；其他模型首跑冒烟

### 2026-08 前期（基线，追记）

* `train_89aa55f31aee5478` 完整训练 + 推理，测试集 ACC@0.5 = 0.7439
