# 交接文档

静态规则见根 `AGENTS.md`；架构见 `architecture.md`，GPU 操作见 `sop.md`，
数据布局见 `data-contract.md`，赛题说明见 `research.md`。
较早日志及旧当前状态已移至 `handoff-archive.md`，原文保留。

## 当前状态（2026-09-13）

- 用户确认魔搭的 `data.tar` 包含已生成的 `Processed/`；SOP 改为解压后直接使用，
  不再将重复预处理列为常规部署步骤。
- 当前标注为 `annot_asm-r6`：train 2,730 条 / 318 序列，val 736 条 / 80 序列。
  使用修复后的 apply 和 package 重建，assembly 内容相同，approved 文件逐字节相同。
  当前日志与快照一致；17 个既有标注、审查及官方模板文件的 SHA-256 未变化。
- 两仓修复范围：生产索引衔接、普查恢复、人审保存/合并、打包检查、推理分片/恢复、
  坐标边界与 DINO 精度记录。模型名称、revision、提示词、R6 标注和已有权重未改。
- 执行策略：训练和推理只读预下载模型，实际执行须传 `--model-path`；CPU 数据
  预检和 `mock` 例外。跨目录续训后置，已有训练记录继续使用原路径。
- 模型阵容（8 个注册适配器）：`qwen3vl` (8B)、`qwen3_5` (9B)、`qwen36_27b` (27B, 
  Modal HF 专属)、`mimo_vl` (7B)、`glm46v` (Flash)、`internvl35` (8B)、
  `groundingdino` (Base, 仅推理)、`mock` (测试)。Youtu-VL-4B 已移除。
- Modal 专为 Qwen3.6-27B 部署（HF `Qwen/Qwen3.6-27B`，需 `[kernels]` 依赖）。
  其他模型均在魔搭 DSW 本地环境训练。
- 本地 `qwen_vg`：Python 3.12.13。主仓 196 项测试通过，0 skip；原生 processor 
  的默认训练输入和四样本推理输入构建通过，没有加载模型权重。
- 用户提供的实验状态：两个 Qwen 按 SOP 执行，GroundingDINO 官方 Test 为 0.576。
  本轮未读取 DSW 结果或复验榜单成绩；不据代码审查判断模型能力上限。
- GPU 待验证：所有新接入模型（`mimo_vl`、`qwen36_27b`）需 DSW/Modal 环境冒烟。
  运行入口存在不等于 GPU 验证通过。
- 后续运行：按 SOP 做实际模型冒烟；涉及新解析/参数的实验使用新标签，不混入旧结果。

## 交接日志（追加式，新的写最上面）

### 2026-09-13（模型阵容调整：移除 Youtu-VL，接入 MiMo-VL 与 Qwen3.6-27B）

- **动因**：按用户指示移除 Youtu-VL-4B（需独立依赖环境），接入 MiMo-VL-7B-RL（魔搭）
  和 Qwen3.6-27B（HF，Modal 专属），扩展 WBF 成员与换代底座。
- **改动**：
  1. 移除 `youtu_vl`：删除适配器文件（204 行）、注册表条目、6 个测试用例、
     `training_core` 白名单、文档章节（`sop.md`、`architecture.md`、`handoff.md`）。
  2. 接入 `mimo_vl`（MiMo-VL-7B-RL）：新建适配器（392 行），架构
     `Qwen2_5_VLForConditionalGeneration`，输出 JSON `{"bbox_2d": [x1,y1,x2,y2]}`，
     超参对齐 Qwen3-VL-8B（batch_size=1, lr=1e-4, lora_rank=16, lora_alpha=32）。
     魔搭来源 `XiaomiMiMo/MiMo-VL-7B-RL`，支持 LoRA 训练。
  3. 接入 `qwen36_27b`（Qwen3.6-27B）：新建适配器（392 行），架构
     `Qwen3_5ForConditionalGeneration`（与 Qwen3.5-9B 相同），HF 直连
     `Qwen/Qwen3.6-27B` (revision `"main"`)，唯一非魔搭模型。Modal H100 专属，
     需 `[kernels]` 依赖（flash-linear-attention、causal-conv1d），训练 48GB / 推理 32GB。
  4. 注册表与白名单：两模型加入 `__init__.py` 和 `training_core.py`，测试期望
     更新为 8 个模型（字母序）。
  5. 文档更新：`sop.md` 添加 MiMo-VL 下载/训练/推理命令；`cloud/README.md` 重构，
     明确 Modal 专为 Qwen3.6-27B 服务，删除"其他模型"冗余描述；`architecture.md`
     模型表同步更新。
- **验证**：196 项单测全绿 + compileall 全过；所有新适配器通过契约测试（identity、
  训练超参、LoRA 目标层）；文档简洁专业，无冗余描述。
- **下一步**：DSW 端 MiMo-VL 冒烟（zero-shot 探针 + LoRA smoke）；Modal 端
  Qwen3.6-27B 冒烟（需先上传 HF 权重到 Volume）；WBF 成员扩容后全量推理。

### 2026-09-12（数据包已含 Processed，删除重复部署步骤）

- 动因：用户确认供下载的 `data.tar` 已包含 `Processed/`，无需在 DSW 再次生成。
- 改动：SOP 删除常规流程中的预处理命令，同步数据合同与离线端说明。
- 验证：仅修改文档，差异检查通过；未执行预处理，未改代码、数据或权重。
- 下一步：按 SOP 使用包内已有处理结果，继续实际模型的小样本验证与实验。

### 2026-09-12（全量工程审查后的修复）

- 动因：独立源码审查与临时复现发现跨模块接口和恢复分支错误；管理员要求全面修复，
  定案关闭自动下载，跨目录续训后置。
- 改动：模型仅本地加载；修正 GLM 负坐标解析和边缘框量化，DINO 使用实例阈值并记录
  float32；推理先校验已有文件再恢复，分片保存分配列表并修正进程层级及完成后续跑。
  数据仓索引根与图片根分离，兼容已有哈希；完成帧和属性可恢复；人审按日志统一合并，
  保存响应绑定条目；打包先全量检查再写出。修正文档入口，删除未生效的重复配置。
- 验证：主仓 202 项、数据仓 156 项测试通过；数据仓包含本机 HTTP 和 Node 状态测试。
  R6 重新 apply/打包与现存内容一致；官方模板校验通过；冻结模型协议不变。未执行
  真实 API、模型下载、GPU 训练或全量推理。
- 下一步：在实际要使用的 GPU 环境按 SOP 冒烟，使用新标签记录修复后实验。

### 2026-09-11（依赖对齐：transformers 5.15.1 与 peft 0.20.0）

- **动因**：对齐 DSW 镜像系统预装版本，避免虚拟环境安装时的降级与卸载告警。
- **改动**：
  1. `pyproject.toml`：`transformers==5.14.1` → `5.15.1`，`peft==0.19.1` → `0.20.0`。
  2. `aicomp_grounding/config.py`：`MODAL_GPU_PACKAGES` 同步更新版本号。
  3. `docs/sop.md`：版本号描述同步。
- **验证**：182 项单测全绿 + compileall + ruff 全绿；`tests/test_env_contract.py` 契约校验通过。
- **下一步**：DSW 端执行 `pip install -e .` 验证透传命中后启动 GPU 训练。

### 2026-09-11（R6 黄金标注定稿并入库：annot_asm-r6）

- **动因**：完成三人协作分片人工审查结果合并，消除标注缺陷与同帧碰撞，固化为最新黄金标注集 `annot_asm-r6` 并随仓库分发，为下一阶段模型重训与评测提供基准。
- **改动**：
  1. 副仓合并与流水账收敛：`query-foundry` 完成 Part 1（344 query 修订 / 129 调框）、Part 2（190 query 修订 / 52 调框）、Part 3（Train 234 query 修订 + 147 调框，Val 169 query 修订 + 165 调框）事务日志与快照合并至 `outputs/review/asm-train-r5` 与 `outputs/review/asm-val-r5`；清理临时分片清单 `outputs/asm-train-r5-part*.json` 与临时 `-part` review 目录。
  2. 标注缺陷清洗：修正 `150_00000143#04` 脚手架用词（`'The biggest rock next to the grass'`）；修正 `137_00000113#03` 消除同帧碰撞（`'The third crane from left to right'`）；完善 `apply_review.py` 逻辑使后续分片正常解锁 3 条历史 `:todo`。
  3. 资产打包与导出：通过 `package_approved.py` 严格质检打包为 `annot_asm-r6`，自动导出至主仓 `outputs/annotations/annot_asm-r6/`（Train 2,730 样本 / 318 序列 / `dataset_fingerprint` eca27a56...；Val 736 样本 / 80 序列 / `dataset_fingerprint` 6c8f9b86...）。
  4. 仓库白名单分发：`.gitignore` 配置 `annot_asm-r6` 白名单规则，使其随主仓 Git 分发。
- **验证**：主仓 `validate_approved_artifact` 契约校验 100% 通过；`query-foundry` 142 项单测全绿；主仓 188 项单测全绿（耗时 1.8s）+ compileall 全过。
- **下一步**：使用 `annot_asm-r6` 启动 GPU 训练（Qwen3-VL-8B 重训、Qwen3.5-9B / GLM-4.6V-Flash 训练）。

### 2026-09-09（双模型适配层接入：qwen3_5 / glm46v）

- **动因**：按已定案阵容接入下一阶段两个模型适配层，为 WBF 成员扩容与底座
  换代铺路；复用现役训练核心，超参与推理参数对齐已验证的 Qwen3-VL 设置。
- **改动**：
  1. `qwen3_5`：复活 git 存档 `84ce34f` 的 Qwen3.8-27B 适配器（同
     `Qwen3_5ForConditionalGeneration` 类、同 Qwen-VL 协议），改 `MODEL_NAME`/
     `MODEL_REVISION`（ModelScope `460979c3`）、α48→32，保留 `enable_thinking=False`
     与 `mm_token_type_ids` collate 修复。`offline/train`+`offline/infer` 的 max_pixels
     元组纳入 `qwen3_5`。
  2. `glm46v`：新写 GLM-4.6V-Flash 训练型适配器。`Glm4vForConditionalGeneration`
     + `enable_thinking=False`（chat template 实证支持）；box 用 tokenizer 加词
     `begin_of_box`/`end_of_box`（id 151361/151362）坐标 0-1000，自有 prompt
     协议与容错解析器（多 box 判歧义拒绝）；processor 两步路径输出 schema
     （`mm_token_type_ids`/`image_grid_thw`/`pixel_values`）与 qwen 同构，collate
     复用；ModelScope pin `a4ec61fc`。本地下载 processor（非权重）dry-run 验证
     三图模板渲染与训练目标前缀匹配；forward 不吃 `token_type_ids` 故不传。
  3. 注册表 + `offline/train` choices + `training_core`（白名单/adapter 派发/
     下载子路径表）同步；`docs/sop.md` 第 4/5/6 节按既有格式补两模型下载、
     训练、推理指令。
- **验证**：182 项单测全绿（+4 qwen3_5、+7 glm46v）+ compileall
  + ruff 全过；`qwen3_5`/`glm46v` CPU preflight 通过（无需权重）。两模型 GPU
  路径无执行史，列入真机冒烟待办。
- **下一步**：GPU 冒烟（qwen3_5 训练 smoke、glm46v zero-shot 探针）；
  其后按执行序推进 α32 重训 / WBF 成员扩容。
