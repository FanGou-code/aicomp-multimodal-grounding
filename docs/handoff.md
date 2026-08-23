# 交接文档

> 全仓库唯一的状态与交接记录：做到哪了、成绩、下一步。每次交接或阶段性
> 完成时更新「当前状态」并在「交接日志」追加一条（新的写最上面）。
> 结构与代码约定见 `architecture.md`，调研背景见 `research-v2.md`。
> 交接日志保留历史记录，不作为当前状态结论；当前状态以上方最新条目为准。

## 当前状态（最后更新 2026-08-23，交接）

### 成绩一览

| 项目 | 成绩 |
| --- | --- |
| **基线**（Qwen3-VL-8B + LoRA，α32/2ep） | Test **0.7439** |
| **Iteration 02**（α48/3ep/min_lr，同标注） | Test **0.7322** |
| **双模型 WBF**（基线+Iter02 融合） | Test **0.7453** ← 当前最佳 |

### 核心诊断（两层，缺一不可）

1. **静态层（数据天花板）**：标注 Query 风格与测试集漂移——官方 Test 平均 10.3 词 /
   66.3% 空间词 / 33.5% 序数，旧标注仅 6 词 / 26% / 4.3%。这是 Val（同分布）0.92 与
   Test 0.73 之间 19 分鸿沟的根因。
2. **动态层（超参加深）**：两次训练用**同一套标注**，drop（0.7439→0.7322）纯粹是
   α48+3ep+min_lr 加深造成的过拟合（Val 0.903→0.9235、Test 反降）。→ 结论：**在标注
   未对齐前，"向深调参"是负收益**；但注意这是同分布下的交互效应，不能外推"新标注下
   深训也无用"（详见下一步的 A/B 实验设计）。

### Qwen3-VL-8B 触顶判断

- **当前标注下：基本触顶**。浅训 / 深训 / 融合已覆盖 0.7322 ~ 0.7453，超参再折腾仅
  ±0.01 量级；Val 0.92 说明已对数据饱和拟合。
- **是数据天花板，不是模型天花板**。8B 预训练自含空间推理能力，只是当前标注 74% 的
  样本不练它。换对齐标注 → 8B 可望 0.76-0.78；换更大模型（32B/72B）→ 叠加更强预训练
  空间知识，属"降维打击"，**合规**（官方明列 Qwen-VL 为允许模型）。

### 本会话已落地（全部推送 main）

- 标注侧：prompt 重写（官方风格 5 大缺口补齐）+ `audit_query_style.py` + 风格门控
  （`validate_query_style`，实测拦截旧标注 27.4% 短标签）+ `--verify-queries` 自定位
  验证（无红框复定位 IoU<0.5 重写）。
- 训练侧：断点保留策略（step 留 2 / epoch 留 1 / 完成后清空，run 从 16G 降到 ~0.6G）。
- 推理侧：`prepare_inputs`/`predict_from_inputs` 拆分进 DataLoader worker（MI300X
  吞吐预期 0.3 → 1.5+ samples/s），VLM 冒烟 batch 建议 **16**。
- 测试侧：209 全绿（4 跳过，torch 环境执行）。mock 模型名统一 `glm-4.6v`。

### 下一步（执行顺序已定案，替代旧"三模型 WBF 优先"排序）

**总路线**：标注对齐+数据扩展 → 32B/38B 训练 → 后期融合（WBF 与 DINO
替换均属后期，不阻塞主线）。

**阶段一：标注对齐 + 数据扩展（共享地基，零 GPU，只花 GLM-4.6V API）**
- 风格对齐：`--verify-queries` pilot（10 序列）→ `audit_query_style.py` 对齐官方
  （均值 ≈10 词 / 空间 ≥60% / 序数 ~33%）→ 人工抽检 → 全量重生成 Train/Val →
  更新 `.gitignore` 白名单分发并提交。
- 帧扩展：GT 插值把每序列 10 帧扩到数百帧（gap ≤15 帧，`--verify-queries` 门控）。
- 序列全量：500 序列（现 400）+ SHA-256 同源审计。

**阶段二：二代目训练与选型（定案）**
| 槽位 | 模型 | 参数/权重 | 理由 |
| --- | --- | --- | --- |
| 1 主力 VLM | `Qwen3-VL-32B-Instruct` | 33B dense / ~66G | 网格范式 + 全图余量，`qwen3vl` 适配器直接放大 |
| 2 切片 VLM | `InternVL3-38B-Instruct` | 38B dense / ~76G | 切片范式与槽位 1 错误去相关；78B 放弃（静态 156G + 全图 KV ~20G ≈ 180G+，batch=1 训练顶爆且单卡过慢） |
| 3 辅助定位 | `GroundingDINO`（锚定） | 0.2B | 唯一 confidence 来源，WBF 打分；**后期替换更强定位器**（候选 GroundingDINO 1.5-Open-Set） |

**阶段三：WBF（非常后期）**
- 强模型（32B + 38B + 替换后的定位器）出框后，Val 网格标定权重融合；DINO 上车需先过
  "出框率体检 + Val 消融 ΔACC"两关（新标注 Val 719 带真值，可逐样本判定放行/剔除）。

**存储策略（定案）**
- 模型权重**不落持久盘**（100G 配额留给 venv/代码/标注/断点/输出）；每次 GPU 启动从
  魔搭内网拉取模型到临时工作区（同机房内网快，66G 约 6-15 分钟）。
- 前提：实例临时盘空位 ≥ ~80G（32B 轮次；38B 轮次需求更大）。
- 边界：`checkpoint.json` + `outputs/output_lora` + `outputs/annotations` 与提交包
  必须留 `/mnt/workspace`（持久）——模型可失，断点不可失。
- 权衡：每次启动烧数分钟 GPU 墙钟用于下载，对 100h 免费额度占比可忽略。

### 环境与运行边界（沿用）

- 本地 `qwen_vg` conda（Python 3.12）只做 CPU 测试/静态检查；GPU 训练推理用 `offline/`。
- 推理推荐单卡 `--num-shards 1 --num-workers 4`，VLM batch 16 / DINO 32，OOM 退 8/16。
- 本仓库是唯一可移植实验单元；`cloud/`（Modal）账号恢复后才启用。
- 断点续跑语义、run id 指纹连续性均未破坏；`tests/test_models.py` 钉死 Qwen identity。
- 模型权重不落持久盘（100G 配额给 venv/代码/标注/断点/输出）；每次 GPU 启动从魔搭
  内网拉模型到临时工作区，`checkpoint`/标注/提交仍必须留 `/mnt/workspace`。

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

### 2026-08-23（交接，二代目选型与执行策略定案）

* 三人小队定为"一队一模型"，只叠加不删除：旧脚本/配置/产物全保留，新增 `qwen3vl32`
  /`internvl3_38` 一类二代目适配器后不改旧适配器。
* **模型选型定案**（详见「当前状态·下一步」）：
  * 槽位 1 主力 = `Qwen3-VL-32B-Instruct`（33B dense，~66G 权重，网格范式 + 全图余量）；
  * 槽位 2 切片 = `InternVL3-38B-Instruct`（38B dense，~76G，切片范式去相关）；
    78B 放弃——静态 156G + 三图全图 KV ~20G ≈ 180G+，batch=1 训练顶爆，且单卡过慢。
  * 槽位 3 辅助 = `GroundingDINO` 锚定（唯一 confidence 来源），后期替换更强定位器
    （候选 `GroundingDINO 1.5-Open-Set`）。
* **选型依据**：比赛不限"系列"（仅限开源权重 + 推理禁商业闭源 API），但"四输入 +
  原生出框 + 192G 可训"筛掉了 GLM-4.6V（借 API 非自持权重，且 106B 塞不进 192G；
  仅保留为打标教师）、Molmo（单图像/点指协议）、Qwen3.8（无框输出）、更大 MoE
  （单卡不可训）。参数量与 zero-shot 能力同家族内正向相关，但受"单卡可训"与
  "全图不降采样"两条硬度约束。
* **执行顺序替换旧排序**：标注三件套（风格对齐 + GT 插值扩帧 + 序列全量）→ 32B/38B
  训练 → WBF（后期）。旧"三模型 WBF 优先"降级为后期动作；新标注上的 A/B 超参实验
  降级为可选诊断，不再优先。
* **存储策略定案**：模型不落持久盘（100G 配额给 venv/代码/标注/断点/输出），每次 GPU
  启动从魔搭内网拉模型到临时工作区（同机房内网快）；`checkpoint`/标注/提交仍在
  `/mnt/workspace`。前提：实例临时盘 ≥ ~80G 空位。
* **DINO 判定流程**：出框率体检（官方风格 query 上 `--limit` 冒烟）→ Val 消融
  ΔACC（3 模型 vs 2 模型）→ 放行则加权进 WBF，否则直接剔除。
* 本轮仅文档更新，未改代码；测试基线 209 全绿不受影响。

### 2026-08-23（交接，最终成绩与策略定案）

* **成绩定格**：基线 0.7439 / Iter02 0.7322 / **双模型 WBF 0.7453（当前最佳）**。
  融合结果已提交打榜验证，确认双 checkpoint 融合管线可用。
* **触顶判断**：Qwen3-VL-8B 在当前标注下基本触顶（±0.01 量级）；但属数据天花板而非
  模型天花板——8B 预训练含空间推理知识，当前标注 74% 不练它。换对齐标注 8B 可望
  0.76-0.78；32B/72B 属"降维打击"，合规（官方允许 Qwen-VL）。
* **策略定案（下次执行时以此为准，替代早先"基线超参重训"的旧结论）**：
  ① 先做三模型 WBF（InternVL+DINO 推理 + fusion/wbf.py，权重新标注 Val 网格标定，
  预期 0.755-0.765）；② 新标注轮次照跑（pilot→审计→全量）；③ 新标注上跑 **A/B
  两配置**（A: α32/3ep/去min_lr vs B: α48/3ep/保min_lr）分离"分布错配"与"深训过拟合"
  的交互效应，不再盲猜单一超参方向；④ 并行探测 Qwen3-VL 更大变体。
* 早先"在新标注上直接回落基线超参"的建议已被 A/B 实验设计取代（理由：迭代 02 的
  drop 是同分布下的交互效应，不能外推新标注情境，A/B 才是对其主效应的诚实测定）。

### 2026-08-23（仓库，Iteration 02 复盘与标注风格修复）

* 复盘 Iteration 02 测试集退步（0.7439 → 0.7322）：确认根因为标注 Query 风格
  与官方测试集分布漂移（详见「当前状态」），超参改动本身执行无误但优化了失真的
  Val 信号。
* 重写 `FRAME_QUERY_PROMPT` 为官方风格导向（序数/空间关系优先、6-15 词、
  禁止短标签）；更新 `tests/test_modal_workflow_wiring.py` 的 prompt 合同断言。
* 新增 `scripts/audit_query_style.py` 与 `tests/test_audit_query_style.py`：
  量化审计 Query 风格分布，实测当前标注（6.0 词 / 26% 空间 / 4.3% 序数）与
  官方测试模板（10.3 词 / 66.3% 空间 / 33.5% 序数）的差距。
* 补算基线 adapter Val ACC ≈ 0.903（300 条子集），与 Iteration 02 的 0.9235
  对照，确认「Val 涨、Test 跌」的分布过拟合结论。
* 新增训练断点保留策略：`training_core.py` 增加 `_prune_checkpoints`，
  step 断点保留最近 2 个（`_STEP_CHECKPOINT_RETENTION`）、epoch 断点保留最新
  1 个、epoch 结束清空 step 断点、completed.json 写入后清空 `checkpoints/`；
  单 run 断点占用从约 12G 降至训练中峰值约 1.8G、完成后 0。续跑语义与
  run id 均不变；新增 `tests/test_training_checkpoint_retention.py`。
* 推理提速：VLM adapter 拆分 `prepare_inputs`/`predict_from_inputs`
  （`base.py` 协议新增 `supports_prepared_inputs`），DataLoader 的 collate 在
  worker 进程内完成 processor 预处理，消除主进程串行 CPU 瓶颈（此前
  MI300X 实测 0.3 samples/s，GPU 利用率低）；加载时打印 image processor
  类型用于 fast/slow 诊断。新增 `tests/test_dataloader_inference.py`
  （本地无 torch 跳过，GPU 环境执行）。
* 标注质量门控：`query.py` 新增 `validate_query_style`（<5 词且无空间/序数/
  多目标词拒绝）；`sequence.py` 新增 `parse_verification_bbox`；`generate_queries.py`
  新增 `--verify-queries`（生成后用无红框原图复定位，IoU<0.5 重写，验证参数进
  run id）。mock 模型名从占位符统一为 `glm-4.6v`。新增用例覆盖风格门控/验证/bbox
  解析。
* 下一步：新 prompt pilot（10 序列）→ 全量重生成标注 → 基线超参重训。

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
