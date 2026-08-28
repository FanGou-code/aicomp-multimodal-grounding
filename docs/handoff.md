# 交接文档

> 全仓库唯一的状态与交接记录：做到哪了、成绩、下一步。每次交接或阶段性
> 完成时更新「当前状态」并在「交接日志」追加一条（新的写最上面）。
> 结构与代码约定见 `architecture.md`，调研背景见 `research-v2.md`。
> 交接日志保留历史记录，不作为当前状态结论；当前状态以上方最新条目为准。

## 仓库总览

**项目**：第八届全球校园人工智能算法精英大赛·赛题一「基于大模型的多模态视觉理解与推理」
——输入时间同步、空间对齐的 RGB / Infrared / Depth 与英文 Query，输出目标归一化边界框
`[x1, y1, x2, y2]`；唯一评分指标为 `ACC@0.5`（预测框与 GT 框 IoU≥0.5）。

**成绩**：Qwen3-VL-8B LoRA 基线 Test **0.7439**、Iteration 02 **0.7322**、
双模型 WBF **0.7453**（当前最佳）。瓶颈已定位为**旧标注 Query 风格与官方测试集漂移**
（数据天花板而非模型天花板），执行路线为：标注对齐+数据扩展 → 8B 新标注重训 → 后期 WBF。

**结构**：

| 位置 | 职责 |
| --- | --- |
| `aicomp_grounding/` | 核心库：数据/标注、训练与推理核心、模型适配层、WBF、提交包 |
| `cloud/` + `offline/` | 平台壳：Modal 云端 / 离线单机，共用同一核心 |
| `scripts/` | 数据预处理、切分查重、Query 生成与风格审计 |
| `tests/` | 离线单测与工作流契约 |
| `docs/` | 状态交接（本文件）/ 架构约定 / 调研报告 / 实操 SOP |

**文档顺序**：冷启动按 `AGENT.md`：先 `docs/handoff.md`（本文件，状态与下一步）→
`docs/architecture.md`（结构不变量）→ 根 `README.md`（用法）→ 就近 README；
GPU 实操以 `docs/RGBDT视觉定位大模型竞赛全流程SOP与实操指南.md` 为准。

**当前阶段与下一步**：阶段一「自适应视觉消歧新标注生产」已 100% 满额发布
（统一资产目录 `annot_dc189f029d962b27`，共 3,594 帧）。
当前正式进入 **Qwen3-VL-8B 新标注重训 + 打榜**阶段。
执行路线为：Qwen3-VL-8B LoRA 训练（3 epochs / bfloat16）→ 测试集推理与打榜；32B 路线已全面移除。

**运行边界**：本地 `qwen_vg` conda（Python 3.12）只做 CPU 测试/静态检查；GPU
训练/推理用 `offline/`；`modal` 命令由用户本人执行。`checkpoint`/标注/提交包必须留
`/mnt/workspace`。

## 当前状态（最后更新 2026-08-28，8B 新标注重训与打榜）

### 当前专注：Qwen3-VL-8B 大模型微调（2026-08-28）
- **决策**：移除 `qwen3vl32` 适配器与 32B 相关文档，集中算力跑 Qwen3-VL-8B 新标注训练。
- **标注资产基准**：统一使用全新发布的自适应消歧标注 `annot_dc189f029d962b27`（Train 2,875 帧 + Val 719 帧）。
- **8B 训练参数**：`batch_size=1`，`grad_accum=16`，LoRA Rank 16 / Alpha 48，`num_workers=4`，checkpoint 每 20 optimizer step 保存。
- **下一步**：8B full-run smoke → 3 epoch 正式训练 → Val/Test 推理与提交。

### 成绩一览

| 项目 | 成绩 |
| --- | --- |
| **基线**（Qwen3-VL-8B + LoRA，α32/2ep） | Test **0.7439** |
| **Iteration 02**（α48/3ep/min_lr，同标注） | Test **0.7322** |
| **双模型 WBF**（基线+Iter02 融合） | Test **0.7453** ← 当前最佳 |

**总路线**：标注重构+质量对齐 → 8B 重训 → 后期融合（WBF 与 DINO 替换均属后期）

**阶段一：高质量自适应消歧标注生成（共享地基，零 GPU，只花 GLM-4.6V API）**
- 风格对齐：32 序列等距抽样 pilot → `audit_query_style.py` 审计 → 人工抽检 → 全量重生成 Train/Val →
  更新 `outputs/annotations/<run_id>/{train,val}/approved.json` 并提交。
- 帧扩展：GT 插值把每序列 10 帧扩到数百帧（gap ≤15 帧）。
- 序列全量：500 序列（现 400）+ SHA-256 同源审计。

**阶段三：WBF（非常后期）**
- WBF 与 DINO 替换保持为后续项；DINO 上车需先过
  "出框率体检 + Val 消融 ΔACC" 两关（新标注 Val 719 带真值，可逐样本判定放行/剔除），
  全过才加权进 WBF。

**存储策略（定案）**
- 模型权重**不落持久盘**（100G 配额留给 venv/代码/标注/断点/输出）；每次 GPU 启动从
  魔搭内网拉取模型到临时工作区（同机房内网快，66G 约 6-15 分钟）。
- 边界：`checkpoint.json` + `outputs/output_lora` + `outputs/annotations` 与提交包
  必须留 `/mnt/workspace`（持久）——模型可失，断点不可失。
- 权衡：每次启动烧数分钟 GPU 墙钟用于下载，对 100h 免费额度占比可忽略。

### 环境与运行边界（沿用）

- 本地 `qwen_vg` conda（Python 3.12）只做 CPU 测试/静态检查；GPU 训练推理用 `offline/`。
- 推理推荐单卡：Qwen 8B 等小模型 `--num-shards 1 --num-workers 4`。
- 本仓库是唯一可移植实验单元；`cloud/`（Modal）账号恢复后才启用。
- 断点续跑语义、run id 指纹连续性均未破坏；`tests/test_models.py` 钉死 Qwen identity。
- 模型权重不落持久盘（100G 配额给 venv/代码/标注/断点/输出）；每次 GPU 启动从魔搭
  内网拉模型到临时工作区，`checkpoint`/标注/提交仍必须留 `/mnt/workspace`。
- DSW 平台现状：`/opt/rocm 7.2.3` + torch `2.11.0+git` + HIP `7.2.53211`，但 amdgpu
  内核驱动为 `6.10.5`，`rocm-smi` 读不出 GPU 名称（`get_name` libdrm 报错）；该问题
  属于平台镜像/宿主机组合，等待平台提供匹配镜像。

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

### 2026-08-28（仓库，离线推理 --resume 默认值对齐为 True）

* `offline/infer.py` 的 `--resume` 参数改为 `argparse.BooleanOptionalAction`，默认值调整为 `True`，与 `offline/train.py` 对齐。
* 离线推理与训练均支持使用原命令断点续跑；显式重新运行传入 `--no-resume`。
* 新增 `tests/test_inference_checkpoint.py` 对应单测；全量单测 205 项通过（4 skip）。

### 2026-08-28（仓库，全面移除 Qwen3-VL-32B 路线）

* 删除 `aicomp_grounding/models/qwen3vl32.py`、模型注册、CLI 路径、训练核心模型分支
  与 32B 相关测试。
* README、SOP、offline README 与 handoff 当前状态清理 32B 内容；32B 路线不再维护。
* SOP 8B 训练命令改为 `num_workers=4 / checkpoint_interval=20`，与新标注
  `annot_dc189f029d962b27` 对齐。

### 2026-08-28（仓库，训练稳定参数与运行时身份收敛）

* `offline/train.py` 新增 `--num-workers`（默认 0）与 `--checkpoint-interval`（默认 20）；
  `run_training` 支持非负 worker 与正 checkpoint interval。
* 训练 DataLoader 默认不再在大模型加载后 fork HIP worker；需要并发验证时显式开
  worker 并先 smoke。
* `config.py` 运行时版本改为动态：Python、torch、torchvision、HIP version 进入训练
  metadata，避免 DSW 实际 torch 2.11 被记成旧 pin 的 2.13。
* `rocm_env.sh` 注释与 README/SOP/offline README 同步：TunableOp 默认关闭，原因是
  MI300X 上直接开启曾有内存泄漏/OOM 风险；不再描述已不存在的 allocator 配置。
* 全量单测 207 项通过（本地无 torch 跳过 4 项），`compileall` 与 `git diff --check` 通过。

### 2026-08-27（仓库，移除反向闭环验证模块）

* **动因**：场景卡已在规划层解决单目标（属性）与多目标（序数/地标）的语法路由，
  消除二义性。反向验证在微小目标场景存在误杀（False Rejection），且调用量翻倍。
* **清理**：从 `scripts/generate_queries.py` 移除 `_verify_query`、`--verify-queries`、
  `VERIFICATION_PROMPT`；从 `aicomp_grounding/sequence.py` 移除 `parse_verification_bbox`；
  移除对应单测（`VerificationBBoxParsingTests` 与 API 验证测试）。
* `README.md` 与交接文档同步更新生成命令。
* 全量单测 215 项全部通过（4 skip），`compileall` 与 `git diff --check` 通过。

### 2026-08-26（仓库，data 目录回退旧布局并保留 test.json 删除）

* `data/` 恢复为顶层 `Train / Test / Processed` 与
  `train.json / val.json / split_manifest.json / excluded_overlap.json`。
* 移除 `raw / derived / indexes / audits` 目录与兼容 symlink。
* `data/test.json` 不保留；推理继续直接读 `data/Test/queries/queries.json`，
  `build_indexes.py` 不生成 processed test index。
* 保留 `offline/rocm_env.sh` 与 AMD ROCm 文档；数据路径相关代码和文档回退到旧布局。
* 全量单测 223 项通过（4 skip），`compileall` 与 `git diff --check` 通过。

### 2026-08-26（仓库，文档措辞审计与过期描述清理）

* 除 `research` 与 `handoff` 外，审计并修订 README、architecture、offline/cloud
  README、SOP 与相关代码注释。
* `cloud/README` 更新为官方模板直接作为 worker 索引，不再引用 `data/test.json`。
* `offline/README` 修正数据索引路径、统一使用中文描述，并移除宣传性措辞。
* `build_indexes.py` 输出描述不再出现 `test.json`；README 删除教师模型宣传词、
  “推荐的新链”、`快速验证` 等冗余表达。
* 全量单测 223 项通过（4 skip），`compileall` 与 `git diff --check` 通过。

### 2026-08-26（仓库，SOP 精简与文档宣传词清理）

* 重写训练/推理 SOP：删除 Modal 指令、本地预处理/标注流程、重复推理描述；
  数据预处理只在 README 说明。
* 所有模型统一下载到非持久路径 `/root/models`，CPU 阶段不再下载模型。
* README 数据目录改为“预处理前只有 raw，处理后才有 derived/Processed、
  indexes、audits”；教师模型段落删除宣传性描述。
* 清理 `High-Speed`、`Fast`、`二代主力` 等注释冗余词；research 文档未改动。

### 2026-08-26（仓库，数据目录重构与 AMD ROCm 环境配置落地）

* 本地数据目录改为：`data/raw/{Train,Test}`、`data/derived/Processed`、
  `data/indexes/*.json`、`data/audits/excluded_overlap.json`；旧路径保留
  symlink 兼容过渡。
* 已从 `data/` 移除 `test.json`；`build_indexes.py` 不再生成该文件，推理直接
  使用 `data/raw/Test/queries/queries.json` 并由 `inference_core` 内存映射路径。
* 新增 `offline/rocm_env.sh`：启用 hipBLASLt、ROCm Tunable Ops 与
  expandable-segments；README/SOP/offline README 已同步持久化激活方式。
* `ProjectPaths` 与新结构对齐，场景卡/style plan 增加 `--index-root`，
  离线推理默认读取 `data/raw/Test/queries/queries.json`。
* 全量单测 223 项通过（4 skip），`compileall` 与 `git diff --check` 通过。

### 2026-08-26（仓库，推理不再依赖 data/test.json）

* `inference_core.load_inference_items` 现可直接读取官方
  `data/Test/queries/queries.json`，在内存把 `Images/...` 映射为
  `Test/Images/...` 与 `Processed/Test/depth_jet/...`。
* `offline/infer.py` 与 `cloud/infer.py` 的 Test 推理默认走官方模板；
  云端不再需要 `data/test.json`，也不要上传该文件。
* README / SOP / offline README / architecture 同步更新，`test.json`
  降级为本地可选重建文件。
* 全量单测 223 项通过（4 skip）。

### 2026-08-26（仓库，数据分发架构优化：云端不再重复生成数据）

* README/SOP 更新为云端只上传 `Train / Test / Processed` 图片树与 `test.json`；
  `train.json`、`val.json`、`split_manifest.json`、`excluded_overlap.json`
  仅保留在本地流水线与审计中。
* 魔搭准备流程改为：临时盘下载 `data.tar` -> 全量解压到持久盘 -> 删除压缩包，
  不再在云端执行 Depth-JET 生成和全量 SHA-256 查重。
* 全量 SHA-256 查重明确只在本地首次准备数据时执行。

### 2026-08-27（仓库，自适应视觉消歧标注全量 Train+Val 满额发布）

* **全量数据满额发布**：
  - 统一 Run ID：`annot_dc189f029d962b27`
  - Train 集：`outputs/annotations/annot_dc189f029d962b27/train/approved.json`（2,875 样本，320 序列，100% 成功发布）；
  - Val 集：`outputs/annotations/annot_dc189f029d962b27/val/approved.json`（719 样本，80 序列，100% 成功发布）；
  - 全量总计 3,594 样本（0 失败，0 不确定，0 丢失）。
* **全量量化审计结果**：
  - 均值词长 10.98 词，词长中位数 10 词（官方 Test 集为 10.33/9 词，高度贴合）；
  - 序数消歧占比 22.1%（成功恢复至 20%~30% 黄金消歧区间）；
  - 空间地标占比 35.9%（空间锚定充沛）；
  - 纯属性动作占比 30.1%（彻底消灭旧基线的 65.9% 偷懒短标签）；
  - 格式与语法纯净度：0 句号残留，0 定语从句冗余，0 标注框伪影泄露。
* **模型与训练核心对接**：
  - 全量通过 `validate_approved_artifact` 校验，格式 100% 兼容 `aicomp_grounding/models/qwen_dataset.py`。
  - 阶段一「新标注生成与数据重构」正式圆满达成。

### 2026-08-27（仓库，重构自适应视觉消歧标注系统与架构极净化）

* **架构与流程极净化**：
  - 彻底废除离线两阶段场景卡（`build_scene_cards.py`）与槽位规划（`build_style_plan.py`），移除 `_q1, _q2, _q3` 伪样本膨胀。
  - 回归 **1 帧 1 Query**，全集总规模严格对应真实抽帧（`train.json` 2,875 帧 + `val.json` 719 帧 = 3,594 帧）。
* **自适应思维链提示词（`DISAMBIGUATION_QUERY_PROMPT`）**：
  - 输出结构化 JSON：`target_category` → `visible_attributes` → `action_or_state` → `spatial_landmark` → `disambiguation_cue` → `final_query`。
  - 确立「场景条件双轨制」：单目标场景专注描述属性与地标，`disambiguation_cue` 填 `null`；多同类共存场景强制输出序数/极值定位锚点（如 `leftmost`, `second from the left`）。
  - 严守语法与视觉安全护栏：紧凑名词短语（分词/介词后置定语），严禁定语从句（避免 `who/which`），首词冠词，无句末句号，严格观察者视角，严防红框标记颜色污染。
* **本地 Python 端确定性验收门控（Deterministic QC Gatekeeper）**：
  - 词数门控：严格限制 $6 \le \text{words} \le 20$；
  - 反偷懒门控：拦截孤立裸词标签（如单独的 `"The person"` 自动重试）；
  - 序列级防复读：在同一视频序列内，如果当前帧生成的 Query 与已生成帧完全一致，强制触发重试并注入差异化提示；
  - 标点自动清理：`final_query.rstrip('.?!;')`。
* **测试与文档**：
  - 更新 `tests/test_query_style.py`、`tests/test_annotation_api.py`、`tests/test_query.py` 等单测，全量 212 项单测 100% 通过（4 skip）。
  - 统一更新 `README.md`、`docs/architecture.md`、`docs/handoff.md`。

### 2026-08-26（仓库，训练/推理实时进度日志统一）

* 训练与推理进度日志统一使用绝对时间戳 `YYYY-MM-DD HH:MM:SS`。
* 推理日志移除用途有限的 `Elapsed`，改为 `Progress 3000/9555 (31.4%) | speed | ETA`。
* 训练日志继续保留 `loss`、`s/step`、`ETA`，Epoch 结果也补上绝对时间戳。
* 新增进度日志格式化单测；全量单测 222 项通过（4 skip）。

### 2026-08-26（仓库，文档同步与新标注策略提交说明）

* 同步 README、architecture、SOP 与 handoff：新 `query_style` 核心模块、
  场景卡、style plan、expanded data root 和分组提示词均已纳入文档。
* 旧自由生成模式已移除，README/SOP 只保留新风格计划链路命令。
* 技术文档与公开 CLI 不再保留测试集分析相关工具和参数。
* `offline/infer.py` 的推理进度输出 bug 修复一并保留，未合并其他无关改动。

### 2026-08-25（仓库，新标注策略基础接入，旧流程零兼容破坏）

* 新增 `query_style.py`，把“语义内容”和“句式结构”解耦：官方 Query 全量分析、
  场景卡结构化字段、style plan、合成样本 ID、分组提示词全部为纯逻辑可单测。
* 新增三个脚本：官方模板全量分析、400 序列场景卡、可生成 expanded data root
  的 style plan。expanded root 使用相对 symlink 指向原 `data/Train` 与
  `data/Processed`，原始索引不修改。
* `generate_queries.py` 保留旧模式；仅当 item 存在 `annotation_style` 字段时
  注入对应官方模板族提示词，返回 JSON 允许可选 `style` 字段。
* 审计升级：`scripts/audit_query_style.py --full` 增加语义组覆盖输出，
  官方 Query 实测分组为 ordinal 28.5% / spatial 34.0% / distance 10.5% /
  scene_location 5.7% / attribute_action 21.3%。
* 验证：220 项单测全绿（4 skip），`compileall` 与 `git diff --check` 通过。
  API key、场景卡实际运行、pilot 与全量生成均由用户亲手执行。

### 2026-08-23（仓库，标注验证空 content 根因修复与 prompt 混比约束）

* **根因**：`glm-4.6v` 的 Zhipu endpoint 默认启用思考模式。验证调用虽返回
  `finish_reason=stop` 且产生 completion tokens，但内容被 reasoning 占满，
  `message.content` 为空；客户端因此反复报 `API response has no final content`，
  少数请求伴随 `1210` HTTP 400。
* **修复**：`OpenAIProtocolClient` 增加 `thinking_mode` 参数，请求体发送
  `{"thinking":{"type":"disabled"}}`；`generate_queries.py` 的生成与验证共用
  `thinking_mode=disabled`，避免继续猜测旧 `enable_thinking` 参数。
* **验证消息流**：调整为先给无红框原图，再给 `Referring query`、定位任务和
  JSON bbox 合同，降低语境分裂和 JSON 合同被前文抢占。
* **生成 prompt**：加入“约 2/3 空间关系、约 1/3 序数”的混比约束，强调优先最短
  清晰表述、同类别对象才计数序数；避免旧 prompt 把所有帧都推向同一种空间/序数模板。
* **验证**：212 项单测全绿（4 skip），`compileall` 与 `git diff --check` 通过。
  重跑本地 pilot 时应使用新 run-tag；`generation_config` 已变，旧 run id 不复用。

### 2026-08-23（标注生成韧性加固）

* **标注脚本严密加固**：
  * [`aicomp_grounding/api_client.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/aicomp_grounding/api_client.py)：将空响应纳入 3 次自动重试；
  * [`scripts/generate_queries.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/scripts/generate_queries.py)：补齐 `group_keys_by_scene` 与 `APIError` 导入，在 `_verify_query` 中增加异常捕获，优化提示词示例中的空间表述。
* **下一步即时可执行动作**：
  1. **本地轨**：执行 10 序列 Pilot 生成（`--limit-sequences 10`）$\rightarrow$ 运行
     `audit_query_style.py` 风格审计 $\rightarrow$ 启动全量 Train/Val 新标注重生成并发布。

### 2026-08-23（仓库，交接总览补全）

* 在 `docs/handoff.md` 顶部新增「仓库总览」小节：汇总项目定位、当前成绩、仓库结构、
  文档顺序、当前阶段与运行边界，方便新成员冷启动后 5 分钟内接上上下文。
* 仅文档变更，未改代码；测试 211 项全绿（4 跳过），`compileall` 与 `git diff --check`
  通过。

### 2026-08-23（交接，模型路线与团队策略定案）

* 团队按"一队一模型"推进，当前聚焦 Qwen3-VL-8B；DINO/InternVL 方向由队友并行，
  不阻塞主路线。
* **存储策略定案**：模型不落持久盘（100G 配额给 venv/代码/标注/断点/输出），每次 GPU
  启动从魔搭内网拉模型到临时工作区；`checkpoint`/标注/提交留在 `/mnt/workspace`。
* **DINO 判定流程**：出框率体检（官方风格 query 上 `--limit` 冒烟）→ Val 消融
  ΔACC（3 模型 vs 2 模型）→ 放行则加权进 WBF，否则直接剔除。
* 本轮仅文档更新，未改代码；测试基线 209 全绿不受影响。

### 2026-08-23（交接，最终成绩与策略定案）

* **成绩定格**：基线 0.7439 / Iter02 0.7322 / **双模型 WBF 0.7453（当前最佳）**。
  融合结果已提交打榜验证，确认双 checkpoint 融合管线可用。
* **触顶判断**：Qwen3-VL-8B 在当前标注下基本触顶（±0.01 量级）；但属数据天花板而非
  模型天花板——8B 预训练含空间推理知识，当前标注 74% 不练它。换对齐标注 8B 可望
  0.76-0.78；更大模型路线后续再评估。
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
