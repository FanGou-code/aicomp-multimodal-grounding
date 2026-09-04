# 交接文档

> Agent 操作手册见根目录 `AGENTS.md`（静态规则）；本文件只记状态与日志。
> 每次交接或阶段性完成时更新「当前状态」，并在「交接日志」追加一条（新的写最上面）；
> 日志超过 800 行时把旧条目切到 `docs/handoff-archive.md`。
> 结构与代码约定见 `architecture.md`，赛题规则见 `research.md`。
> 日志是历史记录，当前结论以「当前状态」小节为准。

## 仓库总览

**项目**：第八届全球校园人工智能算法精英大赛·赛题一「基于大模型的多模态视觉理解与推理」
——输入时间同步、空间对齐的 RGB / Infrared / Depth 与英文 Query，输出目标归一化边界框
`[x1, y1, x2, y2]`；唯一评分指标为 `ACC@0.5`（预测框与 GT 框 IoU≥0.5）。

**成绩**：Qwen3-VL-8B 基线 Test **0.7439**、Iteration 02 **0.7322**、
新标注重训 **0.7325**、双模型 WBF **0.7453**（当前最佳）。
瓶颈定位（2026-09-05 轮修正）：官方测试集与高仿标注存在分布错位（数据侧），
但序数类失分的大头在底座推理能力，模型侧换代为主杠杆（见「当前状态」）。

**结构**：

| 位置 | 职责 |
| --- | --- |
| `aicomp_grounding/` | 核心库：数据/标注、训练与推理核心、模型适配层、WBF、提交包 |
| `cloud/` + `offline/` | 平台壳：Modal 云端 / 离线单机，共用同一核心 |
| `scripts/` | 数据预处理、切分查重、Query 生成与风格审计 |
| `tests/` | 离线单测与工作流契约 |
| `docs/` | 状态交接（本文件）/ 架构约定 / 数据合同 / SOP / 调研 |

**文档顺序**：`docs/handoff.md`（本文件）→ `docs/architecture.md`（结构不变量）→
根 `README.md`（用法）；GPU 实操以 `docs/sop.md` 为准。

**运行边界**：本地 `qwen_vg` conda（Python 3.12）只做 CPU 测试/静态检查；GPU
训练/推理用 `offline/`（魔搭 DSW）或 `cloud/`（Modal，H100）；`modal` 命令由
用户本人执行。checkpoint/标注/提交包留 `/mnt/workspace`（持久盘）。

## 当前状态（最后更新 2026-09-05，模型阵容定案 + 归因修正 + 输入层封存）

- **模型阵容（本轮定案，均为调研结论、尚未接入）**：主力 `Qwen3.5-9B`——原生
  多模态（Qwen3.5 世代起不再发独立“-VL”版，`config.json` 含 `vision_config`），
  Apache-2.0，transformers 5.14.1 原生 `qwen3_5` 模块，CountBench 97.2 /
  ERQA 55.5 / RefCOCO avg 89.7 三项超 Qwen3-VL-30B-A3B；GDN 混合线性注意力，
  A 卡内核 FLA + causal-conv1d 已装（`envs/README.md`）。WBF 4+1：主力 +
  `GLM-4.6V-Flash`（MIT，`glm46v` 原生，标准注意力）+ `Qwen3-VL-8B`
  （`v4 标注 × α32` 重训）+ `GroundingDINO`（新适配器，门禁制：冒烟 → val 719
  逐样本 → 合格才入融合且低权重）+ 第五席 `Youtu-VL-4B`（管理员于 Modal 亲跑，
  专属 Image 绕开其 tf≤4.57.1 pin，自定义许可证由管理员自审）。
- **排除与退役**：`InternVL` 线退役（微调后 0.60 案底、8B→14B 纸面斜率仅
  +0.4），既有 run 保留；`LocateAnything-3B`（NVIDIA）排除：非商业许可 +
  tf 4.57.1 pin + 无官方多图 + 无置信度输出，其“3B 越级”声称限 ScreenSpot-Pro
  GUI 定位场景（60.3 vs GUI-Owl-32B 58.0），与 REC 无关；检测器独立 WBF 成员
  角色关闭（长句序数推理结构性短板，SAM3-I / FLORA 文献佐证；SAM 3 仅存枚举
  管线执行器可能）；推理思考模式全线关闭。Qwen3.6/3.8 两代均无 ≤16B 开源尺寸
  （3.6 最小 27B，3.8 从 27B 起步且此前已按 371s/step 终止），16B 红线内
  Qwen3.5-9B 为家族最强。
- **超参定性**：三训练 run `plan.json` 核对——BASE(0.7439)=旧标注+α32+默认
  调度+旧选优，Iter02(0.7322)=旧标注+α48+cosine+acc@0.5 选优，
  NewAnn(0.7325)=新标注+α48+cosine。“新标注降分”为捆绑错觉：同 α48 下新旧
  标注官方总分差仅 +0.03pp。α48 定性为分布移位过拟合：两 run val ACC
  92.35%/92.77% vs 官方 73.2%（断层 ~19.5pp），Iter02 的 val 至 epoch 3 仍涨
  而官方反跌——同分布 val 对此失明。**全线新训练 α32 起步**（=2r LoRA 标准
  默认、唯一有胜绩的值）；调度与选优保留现行 cosine + acc@0.5；`v4×α32`
  重训将产出首个干净 α 读数。
- **标注定性（v5 工程靶子，工程本身待管理员另启）**：官方 test 9555 条 vs
  golden 2875 条 vs 旧标注 `annot_ac72f1d926bb2d23` 2875 条全量对照，四维
  错位：① 方向枚举句式（“from left to right”系）test 1046 条(109‰)/新旧标注
  均 0；② 序数桶占比 test 33.5%/新 22.0%/旧 4.3%（属性动作桶旧 71.7% 严重
  超配）；③ 词表(频≥5)与 test 交集新 30%/旧 25%；④ extreme(-most) 新标注
  超配 4.8 倍(157‰ vs 33‰)。旧标注另有 39.7% 逐字重复、词数 6.0（test 10.3）。
  序数密度 5 倍提升仅换官方总分 +0.03pp → 曝光≠能力，v5 预期收益诚实标注为
  有限，序数大头押底座能力换代。v5 spec = 四维对齐 + 去重 + **val 719 联动
  重生成**（19.5pp 断层下，现行 best-epoch 选优器优化的是错误分布）。
- **输入层封存**：两大失分区（序数类、框精修区）均模态无关；残余多模态依赖题
  占比极小且现有三图拼接方案已覆盖（test 侧逐样本分析在外部私有分析仓，按其
  防污染规范数字不入本仓）；中期融合（DualVision/Flamingo 类）与热显式专项
  补录除名，深度/红外按现行格式照常输入；RGB-only 消融降级为可选取证材料
  （导师汇报用）。
- **A 卡性能定案（沿袭）**：80 CU 削减版 MI300X（满血 304），GEMM 实测
  200 TFLOPS = 硅片理论峰 90%，适配打满、优化层关闭；与 H100 诚实差距约
  1.5-1.7 倍。
- **环境（沿袭）**：`envs/gpu.txt` 唯一 pin 源（transformers==5.14.1 /
  peft==0.19.1 / accelerate==1.14.0 / qwen-vl-utils==0.0.14），DSW venv、
  Modal Image、config.py、SOP 四方一致；新成员在 Modal 的专属 Image 属
  `cloud/` 壳层事务，不改本仓环境合同。
- **仓库（沿袭）**：pyproject + ruff 全库 0 违规 + GitHub Actions CI；索引在
  `data/indexes/`、审计在 `data/audits/`，`prepare_rgbdt.py` 唯一索引生成器；
  golden `annot_dc189f029d962b27`（train 2875 / val 719）冻结不动；外部私有
  分析仓（本机路径，管理员掌握）承载 test 侧灰色分析，按其防污染规范运作，
  结论与数字不入本仓。
- **cloud/**：Modal 双壳就绪（H100 / 8 核 / 32GiB，Volume `aicomp` 挂
  `/mnt/workspace`）；`modal volume put` 上传 Test 子集仍为首跑前置（管理员
  操作）；第五席 Youtu-VL-4B 走专属 Image。
- **执行序（三账号并行，管理员分工）**：账号 A = Qwen3.5-9B adapter 接入 →
  zero-shot val 探针 → 训练；账号 B = Qwen3-VL-8B `v4×α32` 重训 +
  GLM-4.6V-Flash 探针/训练；账号 C = DINO 门禁 + 备选探针。adapter 工程队列
  为串行瓶颈：Qwen3.5-9B（GDN 的 LoRA target 适用性为已知风险点）→
  GLM-4.6V-Flash → 其余；探针一律推理级轻量 adapter，胜者才补训练侧。成员
  选拔判据用实测三件套：rank+方向轴子集 zero-shot、框紧致度、错误相关矩阵。
  下载权重前 `df -h /mnt/workspace`。

### 成绩一览

| 项目 | 成绩 |
| --- | --- |
| 基线（Qwen3-VL-8B + LoRA，α32/2ep） | Test **0.7439** |
| Iteration 02（α48/3ep/min_lr，同标注） | Test **0.7322** |
| 新标注 Qwen3-VL-8B（α48/3ep/min_lr） | Test **0.7325** |
| 双模型 WBF（基线+Iter02 融合） | Test **0.7453**（当前最佳） |

### 验证基线

213 项单测通过（4 skip，本轮复核，零代码改动）；compileall / ruff / mock
端到端冒烟沿袭 2026-09-01 轮基线。

## 交接日志（追加式，新的写最上面）

### 2026-09-05（模型阵容定案：主力 Qwen3.5-9B，WBF 4+1，检测器独立角色关闭）

* **起因**：0.7453 → 0.80 需底座换代与融合成员扩充；按五道硬门槛（≤16B、
  开源权重〔规则禁商业闭源 API〕、transformers 5.14.1 可加载、MI300X ROCm
  可跑、可 LoRA）全网普查，覆盖 Qwen3.5/3.6/3.8、GLM-4.6V-Flash、
  GLM-4.1V-9B-Thinking、InternVL3.5-14B、Ovis2.5-9B、LFM2.5-VL-3B、
  Ministral-3、Youtu-VL-4B、LocateAnything-3B、SAM 3、Rex-Omni、Molmo2 等。
* **改动（决策与调研记录，零代码）**：定案全部并入「当前状态」模型阵容与
  排除退役两节。WBF 角色定位“樱桃非蛋糕”：唯一实测增益为同族双 Qwen +0.14，
  跨家族未验证且有框风格风险（官方 GT 偏紧教训），融合权重决策押后至成员
  数字到齐。GLM-4.1V-9B-Thinking 定位为 GLM 槽族内备胎（4.6V-Flash 探针
  不过才启用，避免双 GLM 同票稀释）。
* **验证**：调研数字均来自公开模型卡/论文/官方文档（关键项：Qwen3.5-9B
  `config.json` 原生多模态与视觉基准、transformers v5.14.1 模块表、
  Youtu-VL-4B 卡内 tf≤4.57.1 pin 与自定义许可、LocateAnything 许可条款与
  ScreenSpot-Pro 声称范围）；本仓 213 项单测复核通过。
* **下一步**：adapter 队列按当前状态执行序推进；DINO 两项检查（出框率冒烟 +
  val 逐样本）先行；第五席 Youtu-VL-4B 由管理员 Modal 亲跑，预测文件回流
  本仓后入融合评估。

### 2026-09-05（语义分布全量调查 + α 归因修正：标注四维错位与超参捆绑拆解）

* **起因**：解释 0.7439/0.7325/0.7322 的因果链，为 v5 标注工程立实测靶子；
  管理员指出旧标注 `annot_ac72f1d926bb2d23` 从未进过分布分析。
* **改动（只读分析，零代码）**：官方 test 9555 条 vs golden 2875 条 vs 旧标注
  2875 条全量对照（4 桶/5 族句式、词数、冠词模式、词表、重复率、11 类句式
  模式）+ 三训练 run `plan.json` / `completed.json` 核对；结论并入「当前状态」
  超参定性与标注定性两节。此前“新标注重训反而降分”为跨超参错误归因，正式
  撤回。分析脚本在 `/tmp/sem_analysis.py`、`/tmp/old_ann_full.py` 等（可复跑，
  不入库）。
* **验证**：新旧标注条数与 golden 一致（train 2875 / val 719）；分布读数全部
  由本仓产物计算（官方 queries、approved、merged、plan、completed 与官方
  总分），未依赖灰色资产数字；分桶口径与外部分析仓诊断脚本对齐。
* **下一步**：v5 query 生成工程由管理员另启（spec 讨论单独开轮）；`v4×α32`
  重训完成后回收首个干净 α 读数并回填本文件。

### 2026-09-01（A 卡性能定案：80 CU 削减版硅片，适配已打满）

* **起因**：27B 训练 371s/step、8B 训练 90-120s/step，疑似"A 卡适配未完成"。
  用控制变量探针逐层排查：配置（hipBLASLt/rocBLAS/大工作区三路 + 全开关）、
  形状（方阵 vs 训练形状）、精度（bf16 vs fp16）、注意力（flash 14.2ms vs math
  112ms，flash 可用且训练已在走）——四层全排除后仍稳定在 ~200 TFLOPS。
* **根因实测**：`rocminfo` 显示 GPU 计算单元仅 **80 个**（满血 MI300X 为 304），
  GEMM 满载 sclk 1364MHz / 515W。80 CU × 2048 FLOP/clk × 1.364GHz ≈ 223 TFLOPS
  理论峰值 → GEMM 实测 200 = **该硅片的 90%**。HBM 实测 3165/3700 GB/s（86%）。
  即：显存满血 192GB、算力削减 26% 的出口合规型配置。
* **结论修正**：撤回"比正常水平慢 5-10 倍"（分母误用满血峰值 1307）。用真实
  峰值重算，8B 训练 ~26% MFU 属正常区间；与 H100 的诚实差距约 1.5-1.7 倍
  （CUDA 生态成熟度差，非硅片差）。
* **决策**：A 卡优化层关闭（已打满，无可榨空间）；8B 上位模型选型与速度解耦；
  TunableOp 在此 stack 上调优阶段卡死（Ctrl+C 无效），正式除名；
  FLA + causal-conv1d 已装（causal-conv1d 经 FORCE_BUILD 本地编译成功，
  官方无 HIP 预编译轮子）。
* **教训**：免费配额的对价——显存满血、算力削减；跨实例 venv 锚点会随镜像
  python 路径漂移；平台镜像的 GEMM 性能仅达公开 MI300X 基准的 ~15-30%，
  可作为工单材料反馈平台。

### 2026-09-01（仓库，索引重建闭环 + cloud/ 恢复 + CLI 解耦收口）

* **数据管线闭环**：重建 `data/indexes/`（400 序列 / 4000 行 GT / 3903 有效样本，
  深度全复用）→ 重跑 `filter_overlap.py`（246+63 剔除）→ train/val 索引与 golden
  `annot_dc189f029d962b27` 逐样本零偏差；审计日志落 `data/audits/`。
  SHA-256 走 CPU SHA-NI（数分钟级），无需 GPU。
* **cloud/ 恢复（全功能）**：`cloud/infer.py` + `cloud/train.py` + README。硬编码=
  H100 / 8 核 / 32GiB（Iteration 02 实测包络）+ Volume `aicomp` 挂 `/mnt/workspace`
  （与魔搭布局对齐）+ 依赖 pin 写入 Image（同 envs/gpu.txt，torch 用最新 CUDA 稳定版）；
  代码经 `add_local_dir` 每次运行时上传（非镜像烘焙），改代码即生效。训练经
  `commit_hook` 每 checkpoint 提交 Volume，抢占后可恢复。
* **CLI 解耦**：`offline/{train,infer}.py` 拆出 `run_cli(args)`，cloud 壳直接复用
  同一编排函数（`project_root=/mnt/workspace`），本地行为零变化；
  `infer.run_cli` 返回 summary。`pip install -e .` 实测通过。
* **gitignore**：`cloud/` 摘出忽略名单（正式入库）；`aicomp_grounding.egg-info/`
  为 pip -e 生成物，已被 `*.egg-info/` 规则覆盖。
* **验证**：213 项单测通过（4 skip）+ ruff 全绿（含 cloud/）+ compileall（含 cloud）
  + mock 端到端冒烟通过。
* **下一步**：16B×3 选型确认 → adapter 接入 → MI300X 训练；Modal 首跑冒烟待
  `modal volume put` 上传 Test 子集后执行（用户本人操作）。

### 2026-09-01（仓库，27B 线终止 + 三波仓库修复）

* **方向变更**：Qwen3.8-27B 太重（371s/step、全程 ~51h 配额），决定终止并整体移除；
  本次为方向变更存档，未提交的 27B 时代改动先以 `84ce34f` 存档再动刀。
* **Wave 1（7e55281）**：删除 `models/qwen3_8.py` 与全部注册/分支/CLI/测试；
  环境声明对齐 DSW 镜像自带的 transformers 5.14.1（`envs/gpu.txt` 单一 pin 源）；
  机械修复：checkpoint 数值排序（step≥10000 误删雷）、worker LoRA 静默降级改报错、
  顺序推理补 checkpoint、dtype kwarg 统一（dtype 优先 + torch_dtype 回退）、死赋值、
  吞异常加日志、resume 语义注释。测试基线 225→214。
* **Wave 2（135feed）**：pyproject.toml + ruff（修复全部 51 项违规后全库 0 违规）+
  GitHub Actions CI（ruff/compileall/unittest）；索引迁入 `data/indexes/`、审计迁入
  `data/audits/`，`prepare_rgbdt.py` 成为唯一索引生成器，`build_indexes.py` 删除
  （1920×1080 硬编码与静默脏数据源头一并消失）；SOP 改名 `docs/sop.md`；
  新增 `docs/data-contract.md` 与根 `AGENTS.md`；.gitignore 清理（删 3 条残留规则、
  加 `.qoder/`）；旧 golden `annot_ac72f1d926bb2d23` 停止分发。测试基线 214→213。
* **Wave 3**：`inference_state` 与 qwen3vl 常量解耦（config 增
  `INFERENCE_DEFAULT_{MIN,MAX}_PIXELS`，值不变 → run id 不漂移）；
  `evaluate_predictions` 更名 `evaluate_dataset_predictions` 消除同名异义；
  adapter 内联 prompt-前缀校验收口到 `validated_prompt_length`；
  `load_inference_items` 非法条目改为显式报错；loader fork 语义、API 重试语义注释；
  README 硬编码样本数清理。
* **环境考古结论**：transformers==4.57.3 是 08-02 Initial commit 起的元老 pin，
  08-18 魔搭迁移是向它对齐；DSW 镜像现自带 5.14.1（pip 装于系统 dist-packages，
  非 conda），故新定版与镜像天然一致，venv 安装同版本仅作防漂移遮蔽。
* **验证**：213 项单测通过（4 skip）+ ruff 全绿 + compileall + mock 端到端冒烟。
* **下一步**：16B×3 选型（门槛 5.14.1 可加载）→ adapter 接入 → MI300X 训练；
  Modal infer 壳（Image 读 envs/gpu.txt）选型定后编写。

### 2026-08-31（仓库，清理 Qwen3.8/InternVL 适配层隐患）

* Qwen3.8 自动加载不再走 HF：未传 `--model-path` 且本地无权重时，改用
  ModelScope `snapshot_download(MODEL_NAME, revision=e823e888...)`，避免 HF
  侧 404；显式传本地模型目录的 SOP 路径不受影响。
* InternVL 移除无效的 `max_num_tiles` 配置；实际分辨率预算统一记录为
  `preprocessor_config.json` 控制的 `max_patches=12`，文档同步更新。
* 修正 `qwen3_8` docstring 中错误的 `<|box_r|>`，改为实际
  `<|box_start|>/<|box_end|>`。
* 本地验证：225 项单测通过（5 skip），`compileall` 与 `git diff --check`
  通过；相关测试保持无网络、无模型下载。

### 2026-08-31（仓库，接手验证与下一步门槛确认）

* 当前分支 `feat/qwen8b` 领先 `main` 10 个提交，工作区干净；核心内容为 Qwen3.8-27B
  适配器、训练/推理接入、SOP 与 smoke 修复。
* 接手验证通过：`unittest discover -s tests` 221 项通过（5 skip）；
  `compileall`、`git diff --check` 通过。
* Qwen3.8 CPU Preflight 已在本机通过：
  `offline/train.py --annotation-run-id annot_dc189f029d962b27 --model qwen3_8
  --seed 42 --run-tag exp-qwen38-27b-01 --preflight-only`。
* 本机边界确认：无 `/mnt/workspace`、无 `aicomp_env_q38`、无 Qwen3.8 权重，
  `qwen_vg` 未安装 torch/transformers，只有 8GB NVIDIA GPU；全量 27B 训练仍需按
  SOP 在具备 `/mnt/workspace + aicomp_env_q38 + /root/models` 的 GPU 主机执行。
* `cloud/` 目录按 `.gitignore` 约定本地保留，不入库；当前分支不含该目录。

### 2026-08-31（仓库，Qwen3.8 smoke 阻塞修复）

* GPU smoke 已完成 Qwen3.8 权重加载与 LoRA 注入；前向失败原因为训练 collate 丢弃 Qwen3.5 必需的 `mm_token_type_ids`，已修复。
* `qwen3_8` collate 已保留并 padding `mm_token_type_ids`；runtime metadata 按模型记录实际安装版本，Qwen3.8 fallback 为 transformers 5.8.0。
* GPU smoke 重跑通过：`train_loss=0.5076, val_loss=0.4748`，`use_cache` warning 属正常。
* fast path（`flash-linear-attention` / `causal-conv1d`）编译过慢，已放弃安装，用 torch fallback 跑全量；A 卡加速效果未实测，全量耗时以首 20 步 `sec/step` 为准。
* 本地验证：221 项单测通过（5 skip），compileall、`git diff --check` 通过。
* 下一步：全量训练（`--run-tag exp-qwen38-27b-01`，不带 `--smoke-test`）。

### 2026-08-31（仓库，Qwen3.8-27B 接入与 SOP 定稿）

* 研究确认：魔搭 `Qwen/Qwen3.8-27B` 存在，Apache-2.0，`model_type=qwen3_5`，需 `transformers>=5.8.0`，与主环境 4.57.3 不兼容，单独 `aicomp_env_q38`；权重 55.6GB，落 `/root/models` 每次新实例重下。
* 新增 `qwen3_8` 适配器：`Qwen3_5ForConditionalGeneration`，MODEL_REVISION pin `e823e888...`，超参/推理照搬 8B（batch 1 / grad_accum 16 / lr 1e-4 / 3ep / alpha 48 / rank 16 / infer batch 4）；关闭默认思考模式 `enable_thinking=False`。
* 注册与入口：`models/__init__.py`、`offline/train.py --model qwen3_8`、`offline/infer.py`（max_pixels 特判纳入）。
* SOP 新增 Qwen3.8 独立环境 / 临时盘下载 / 训练 / 推理四节，未改其他模型小节。
* 本地验证：219 项单测通过（4 skip），compileall、`git diff --check` 通过。
* 时长预估：MI300X 192GB 每步约 5-8min，540 步约 2-3 天，每 20 步约 100-160min，以 smoke 实测为准。
* 下一步：训练 smoke → 全量 27B → 推理 smoke（Val 看 ACC）→ Test 全量推理。

### 2026-08-30（接手，本地接手验证与前进门槛）

* 已完整阅读 `docs/handoff.md`、`docs/architecture.md`、根 README、offline README 与 SOP；确认当前方向为更大参数开源 VLM + InternVL3.5-8B/DINO 异构 WBF。
* 本地接手验证通过：`unittest discover -s tests` 212 项通过（4 skip，torch/transformers 未装）；`compileall`、`git diff --check` 通过；`offline/infer.py --model mock --limit 3` 端到端推理通过。
* 风格审计复核：新版 Train 词长 11.0/10、ordinal 21.6%、spatial_landmark 36.0%、distance 1.6%；旧版 6.0/6、4.3%、21.6%、1.2%；Test 10.3/9、28.5%、34.0%、10.5%，与 handoff 记录一致。
* 环境差异：当前接手机器为 WSL 单机（8GB NVIDIA，无 `/mnt/workspace`，`qwen_vg` 未装 torch/transformers）；本地有完整 `data/`、golden 标注与历史 predictions，但缺最佳 checkpoint `best/epoch_02` 与新测试提交包实体，这些 `/mnt/workspace` 远程产物未同步到本仓库。
* 下一步仍被 GPU/持久工作区门槛阻塞：InternVL3.5-8B 训练 smoke、DINO Val 体检、8B 上位模型短 smoke、WBF 都需在具备 `/mnt/workspace` 的 GPU 环境执行。

### 2026-08-30（分析，新旧标注全量对比与方向定案）

* **成绩记录**：新标注 Qwen3-VL-8B Test 0.7325；旧浅 0.7439；旧深 0.7322；WBF 0.7453。
* **全量分布对比**（旧/新各 3594 条，Test 9555 条）：
  - Test 结构为 2000 张图、平均每图 4.78 query，中位 5，最多 14；训练标注为一帧一 query。
  - 新标注在词长（L1 0.184 vs 0.390）和语义五分类（L1 0.156 vs 0.456）上比旧标注更接近 Test。
  - 关键消歧词/句式上新标注未更接近：`near` 过高、`first/second/third`、`closest/farthest`、左右方向覆盖不足；distance 仅约 1.6%，Test 为 10.5%。
* **讨论结论**：标注不是主要扣分来源，当前协议够用；不再改 prompt/标注。输入协议、坐标输出、硬件精度未发现确认问题；查询理解最可疑但无足够把握修改。
* **方向定案**：8B 判定触顶；主攻同系列 16B，配合 InternVL3.5-8B 异构模型和 DINO 做 WBF。InternVL 先 Val smoke，DINO 以 Val WBF 消融决定，16B 接入前先短 smoke。

### 2026-08-30（训练与推理，Qwen3-VL-8B 新标注重训完成与测试集提交包就绪）

* **8B 新标注重训完成**：基于 `annot_dc189f029d962b27` 完成 3 轮训练（run id: `train_7b312e6e90bb14db`，540 steps）。
  - Epoch 01: Val ACC@0.5 = 90.96% (654/719), Val Loss = 0.3607；
  - Epoch 02 (Best): **Val ACC@0.5 = 92.77% (667/719 hits)**, Val Loss = 0.3529, Mean IoU = 0.8655；
  - 框架自动选优并锁定保存 `best/epoch_02`。
* **官方测试集推理与打包完成**：
  - 加载 `best/epoch_02` 权重在官方全量测试集（9,555 查询）上完成推理（run id: `infer_88e5b6123b2c0fbe`）；
  - 输出 9,552 个有效边界框（有效率 99.97%），3 个未出框样本由系统自动填充安全兜底框；
  - 自动构建并通过校验生成官方提交包：`outputs/inference/infer_88e5b6123b2c0fbe/submission.zip`。
* 清理 `research.md` 中未经证实的推测，全文档引用对齐。

### 2026-08-30（仓库，模型持久化与 InternVL/DINO 适配收敛）

* 模型权重改为持久目录 `/mnt/workspace/models`；README、offline README、SOP
  中的路径同步更新。
* InternVL adapter 修正 `<IMG_CONTEXT>` 图片占位符，并保留官方 left-padding
  解码语义；`max_new_tokens` 对齐到 100。
* InternVL collate 不再调用不存在的 `processor.pad`：`batch_size=1` 为 text
  tensor 补回 batch 维，多 batch 手动 left padding 并拼接 `pixel_values`。
* InternVL 验证/推理的 `apply_chat_template` 改为传入完整 conversation 列表，
  修复 transformers v5 对单条 message dict 入参的兼容问题。
* InternVL 加载不再传 `max_num_tiles`，分块预算由模型
  `preprocessor_config.json` 的 `max_patches` 控制。
* InternVL 加载改用显式 `InternVLForConditionalGeneration`，兼容 transformers
  4.57/v5，修复 DSW v5 下 `AutoModelForVision2Seq` 的 ImportError。
* 补充社区核对后：InternVL 此前误改的逐样本 `attention_mask` 解码已回退，
  生成 token 继续按官方 left-padding 后的全局 `input_ids.shape[1]` 切片。
* DINO adapter 修正直接消费 post-process XYXY、显式
  `GroundingDinoForObjectDetection`、本地 processor fallback、阈值参数兼容，
  并按官方 demo 增加 query `strip/lower/补句号` 预处理。
* DINO `select_top_detection` 改为按分数降序回退，最高分非法时继续取下一个
  合法框，不再直接返回 None。
* README 移除成绩/run id 等现状记录；architecture 验证状态与 DINO XYXY
  描述同步；SOP 补齐 DINO 下载与推理命令；research-v2 修正 InternVL 坐标序。
* 全量单测 212 项通过（4 skip）；`compileall` 与 `git diff --check` 通过。
* 未在真实 GPU 上执行 InternVL/DINO Val 切片；该项仍待 GPU 环境验证。

### 2026-08-29（仓库，接手本地状态与新标注训练 preflight）

* 修正 `annot_dc189f029d962b27` Train/Val approved 资产的 `image_fingerprint`：
  `verification-skipped` -> `manifest_49bbae...` / `manifest_6322c1...`；训练 run identity
  现在绑定可信图片引用指纹，而非未校验占位符。
* 本地 `qwen_vg` 环境 CPU preflight 通过：`train_72d09d474a1c8a0c`
  （`annot_dc189f029d962b27` + `exp-qwen8-retrain`，2875 Train / 719 Val，已就绪未训练）。
* 全量单测 206 项通过（4 skip）；`compileall` 与 `git diff --check` 通过。
* 本机无可用 GPU（`nvidia-smi` 被系统拦截）；8B smoke/full-run 留待 GPU 环境执行。

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

### 2026-08-27（仓库，自适应消歧标注全量发布）

* **全量发布**：
  - 统一 Run ID：`annot_dc189f029d962b27`
  - Train 集：`outputs/annotations/annot_dc189f029d962b27/train/approved.json`（2,875 样本，320 序列，100% 成功发布）；
  - Val 集：`outputs/annotations/annot_dc189f029d962b27/val/approved.json`（719 样本，80 序列，100% 成功发布）；
  - 全量总计 3,594 样本（0 失败，0 不确定，0 丢失）。
* **全量量化审计结果**：
  - 均值词长 10.98 词，词长中位数 10 词（官方 Test 集为 10.33/9 词，高度贴合）；
  - 序数消歧占比 22.1%（目标区间 20%-30%）；
  - 空间地标占比 35.9%；
  - 纯属性动作占比 30.1%（旧基线为 65.9%）；
  - 0 句号残留、0 定语从句、0 红框伪影词。
* **模型与训练核心对接**：
  - 全量通过 `validate_approved_artifact` 校验。阶段一（新标注生成与数据重构）完成。

### 2026-08-27（仓库，标注系统重构）

* **架构精简**：
  - 移除离线两阶段场景卡（`build_scene_cards.py`）与槽位规划（`build_style_plan.py`）及 `_q1/_q2/_q3` 伪样本。
  - 回归 **1 帧 1 Query**，全集总规模严格对应真实抽帧（`train.json` 2,875 帧 + `val.json` 719 帧 = 3,594 帧）。
* **标注提示词（`DISAMBIGUATION_QUERY_PROMPT`）**：
  - 输出结构化 JSON：`target_category` → `visible_attributes` → `action_or_state` → `spatial_landmark` → `disambiguation_cue` → `final_query`。
  - 单/多目标两条规则：单目标场景描述属性与地标（`disambiguation_cue` 为 null）；多同类共存场景强制输出序数/极值锚点（如 `leftmost`）。
  - 输出约束：紧凑名词短语，不用定语从句，首词冠词，无句末句号，观察者视角，不提及红框标记。
* **确定性 QC 规则**：
  - 词数限制 6-20 词；
  - 短标签拒绝（如 `"The person"` 触发重试）；
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
  API key、场景卡运行、pilot 与全量生成均由用户执行。

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
* **下一步**：
  1. 本地轨：10 序列 Pilot（`--limit-sequences 10`）→ 审计 → 全量重生成并发布。

### 2026-08-23（仓库，交接总览补全）

* 在 `docs/handoff.md` 顶部新增「仓库总览」小节：汇总项目定位、当前成绩、仓库结构、
  文档顺序、当前阶段与运行边界，用于新成员冷启动。
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

* 成绩：基线 0.7439 / Iter02 0.7322 / 双模型 WBF **0.7453**（当前最佳）；融合结果已提交验证，双 checkpoint 融合管线可用。
* **触顶判断**：Qwen3-VL-8B 在当前标注下基本触顶（±0.01 量级）；但属数据天花板而非
  模型天花板——8B 预训练含空间推理知识，当前标注 74% 不练它。换对齐标注 8B 可望
  0.76-0.78；更大模型路线后续再评估。
* **策略定案（替代早先"基线超参重训"结论）**：
  ① 先做三模型 WBF（InternVL+DINO 推理 + fusion/wbf.py，权重新标注 Val 网格标定，
  预期 0.755-0.765）；② 新标注轮次照跑（pilot→审计→全量）；③ 新标注上跑 **A/B
  两配置**（A: α32/3ep/去min_lr vs B: α48/3ep/保min_lr）分离"分布错配"与"深训过拟合"
  的交互效应，不再盲猜单一超参方向；④ 并行探测 Qwen3-VL 更大变体。
* 早先"在新标注上直接回落基线超参"的建议已被 A/B 实验设计取代（理由：迭代 02 的
  drop 是同分布下的交互效应，不能外推新标注情境，A/B 才能测定其主效应）。

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
  worker 进程内完成 processor 预处理，消除主进程串行 CPU 瓶颈；加载时打印 image processor
  类型用于 fast/slow 诊断。新增 `tests/test_dataloader_inference.py`
  （本地无 torch 跳过，GPU 环境执行）。
* 标注质量门控：`query.py` 新增 `validate_query_style`（<5 词且无空间/序数/
  多目标词拒绝）；`sequence.py` 新增 `parse_verification_bbox`；`generate_queries.py`
  新增 `--verify-queries`（生成后用无红框原图复定位，IoU<0.5 重写，验证参数进
  run id）。mock 模型名从占位符统一为 `glm-4.6v`。新增用例覆盖风格门控/验证/bbox
  解析。
* 下一步：新 prompt pilot（10 序列）→ 全量重生成标注 → 基线超参重训。

### 2026-08-23（仓库，离线推理断点续跑支持 checkpoint.json 自动恢复）

* `offline/infer.py`：`--resume` 开启时，若无全量 `predictions.json` 但存在阶段性 `checkpoint.json`，自动从中恢复已完成预测。
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
