# 交接文档（query-foundry）

> 本仓 = 主仓 v5 query 生产线。边界与红线见根 README。
> 上游定案见主仓 `docs/handoff.md` 与 `/tmp/handoff-2026-09-05-query-foundry.md`（2026-09-05 会话交接）。

## 当前状态（2026-09-05，管线分离完成）

- 仓已建：私有、无 remote。**主仓标注生成模块已完整迁入本仓**（`foundry/` 包 +
  `scripts/` 入口），主仓只保留产物合同校验（`annotation_state.py` 的
  `validate_approved_artifact` 链）。
- 生产线延续主仓既有行为：分片并发、限流、帧级 checkpoint、同指令重跑自动
  resume、preview 机制、终端反馈格式不变；新增 API key 池（持久化 key 文档
  `keys/api_keys.txt`，固定顺序、耗尽即退、全池耗尽快速失败）。
- 数据集仍在主仓原路径，`--data-root` 默认指向主仓 `data/`；产物落本仓
  `outputs/annotations/`，由管理员手动复制回主仓。
- Phase 0 完成：`spec/style_spec.json` 已冻结（frozen-2026-09-05，口径与 gt-analysis 逐桶对齐）。Phase 1 普查协议未开始。
- v5 新普查/组装/规划 pass 未实现（属 Phase 1+，待管理员通知开工）。

## 交接日志

### 2026-09-06（普查试点事故与修复：坐标约定 + 进程管理事故）

- **根因确诊**：56% 帧失败的根因是坐标约定错配——GLM-4.6V 返回 0-1000 千分比
  框（家族通用约定，主仓 bbox.parse_bbox_from_text 即 qwen_1000），解析器当成
  所示图像像素 ÷1536，x 被压缩 1.5 倍。原始回答 + 覆盖图证实教师找全找对
  （鹿 IoU 0.93、双鞋全中），模型能力、注意力、跟踪框均无辜。
- **修复**：约定自动识别（全 ≤1.02 直用；否则千分比与像素双候选都试，
  金丝雀充当判定器，先过全部质量门者胜）+ 失败帧保留 api_calls 与
  last_raw。验证：5 序列批跑失败率 56%→约 8%（3/39，且为真实测量）。
  run-id 指纹加入 requested_num_shards（修并发变更撞 plan 冲突）。
- **事故**：我的 3 并发后台进程未清干净与管理员 8 并发进程撞车（plan 冲突、
  日志混流）；管理员规则：**一切调用 8 并发**、API 指令由管理员手动执行、
  未经明确指令禁止任何 API 操作——均已记 README 与本文件。
- **遗留**：pilot2 日志被双进程混流污染，结论以 report.json/checkpoint 为准；
  已完成序列 peers=0——跨帧 IoU≥0.5 阈值疑对帧间微动过严，威胁伪样本原料
  供给，第三批前需核查；三个 census run 目录待管理员定夺清理。
- **状态**：全部进程已停，盘子干净；第二批目标（失败率 <10%）已达成，
  第三批（20 序列重跑出能力卡）等管理员指令。

### 2026-09-05（quality-smoke-20：老管线质量基线量化）

- 20 序列 / 181 帧全完成（run annot_73dc03522f6242c6，限跑不可发布）。
- 基线 vs test（冻结口径）：逐字重复 **40.3%** vs 7.6%（比 golden v4 的
  24.9% 更差；070 序列 10 帧同句 9 次）；序数 187.8‰ vs 335.0（欠 147）；
  属性动作 386.7‰ vs 256.4（超 130）；距离 149.2 vs 150.6、空间 276.2 vs
  258.0、词数 10.44 vs 10.33——距离/空间/词数自然贴 test，错位集中在
  序数欠配与属性动作超配。starts-The 100%。
- v5 组装器验收线：重复率 ≤10%、序数 ≈335‰、距离/空间/词数维持。
- 工具修复：audit_query_style 的 extract_queries 支持 merged.json 嵌套结构
  （限跑不可发布，merged 是试点的可审计产物）；2 个新测试，51 项 OK。

### 2026-09-05（管理员新红线：API 调用须逐次明确指令）

- 规则：任何真实产生 API 调用的命令，未经管理员当次明确指令禁止执行；
  零调用的 preflight-only 不受限制。已写入 README 用法段。
- 现状：key 池就绪（api_keys.txt 11 把，0600，gitignored；模板文件已删除——
  曾诱导管理员把真实 key 填入被 git 追踪的模板，已纠正并清扫 stash/git 对象）。
  冒烟命令就绪待管理员放行：`python scripts/generate_queries.py --split train
  --limit-sequences 2 --concurrency 2 --run-tag smoke-sep`（预检已过，
  run_id=annot_d2a2d09fcdc36654，18 帧）。

### 2026-09-05 主仓 sequence/config 生成侧残留并入本仓

- 动因：主仓分离收尾——`sequence.py` 教师响应解析三件套与 `config.py`
  12 个生成侧常量在主仓已零消费者（本仓副本为唯一实体）。
- 改动：新增 `tests/test_sequence_parsing.py`（3 个 parse 测试方法自主仓
  test_artifacts_sharding_sequence.py 迁入）。
- 验证：本仓 49 项 OK；主仓 171 项 OK，golden 复验通过。

### 2026-09-05 style_spec 冻结（口径对齐 + 三项裁决落档）

- 动因：管理员审阅 Phase 0 草案后三项裁决——①阈值下错误一概不学（邻接重复
  14 次不特批）；②中文夹杂不主动制造也不刻意回避，交由底座能力处理；
  ③桶分类口径与 gt-analysis 已发布计数对齐。
- 改动：`scripts/phase0_mine_test_style.py` 四桶分类器对齐
  （序数 > 距离 > 空间 > 属性动作；空间词表不含 foreground/background/
  opposite/rows，即 139 条差额来源）；`spec/style_spec.json` 状态翻为
  `frozen-2026-09-05`，`freeze_decisions` 记录三项裁决。
- 验证：重跑后四桶 3201/2465/2450/1439 与 gt-analysis 已发布计数逐桶
  精确一致；其余基线（重复率 7.6%、词数 10.33、词表 2615）不变。
- 备注：对齐仅取分类正则口径（管理员指示），gt-analysis 的 GT 数据与
  成绩结论未进本仓，防污染边界不破。
- 下一步：Phase 1 普查协议（findall/attr/review 三类 pass）+ 20 序列试点，
  待管理员开工指令。

### 2026-09-05 标注管线自主仓分离 + key 池

- 动因：管理员决定把自动标注模块从实验主仓彻底分离到本仓（含依赖与测试），
  主仓此后只手动接收标注产物；同时接入 API key 池应对 v5 万次级调用量。
- 改动：
  - 自主仓迁入：`generate_queries.py`、`audit_query_style.py`、
    `query_style.py`、`api_client.py`、`annotation_views.py`、
    `annotation_state.py`（全量含生成状态机）与 5 个对应测试文件；
    共享工具 `io/artifacts/bbox/images/paths/config/sequence/query/sharding`
    为复制件（import 已改指 `foundry.*`）。
  - 新增 `foundry/keys.py`：key 文档解析（`#` 注释、行序即调用序）+
    `APIKeyPool`（固定顺序、退役幂等、全池耗尽抛 `APIKeyPoolExhausted`）。
  - `api_client.py` 增 `key_pool` 参数：401/402/403 立即换号；429 同号重试
    一次后换号（瞬时限流不误杀）；5xx/网络错误沿用既有同号退避，不退役；
    非池模式行为与主仓原版逐行一致。`_annotate_frame` 对池耗尽直接上抛
    （不烧帧级 attempt），shard 异常路径照旧落 checkpoint。
  - `generate_queries.py`：`API_KEY` 环境变量逻辑替换为 key 池加载；
    `--data-root` 默认指向主仓 `data/`；池状态与换号打印接入终端输出。
- 验证：本仓 `python -m unittest discover -s tests` 46 项 OK
  （39 项自主仓迁移 + 7 项新 key 池测试）；train/val 双 split
  `--preflight-only` 冒烟通过（读主仓数据、校验 split_manifest 指纹、
  产出合法 plan.json 与 run-id）。
- 主仓侧同步事实：删除 5 个管线文件 + 5 个测试文件（共 213→174 项测试），
  `annotation_state.py` 缩为合同校验；golden v4 train(2875)/val(719) 经保留
  校验器端到端通过，主仓测试全绿。
- 下一步：管理员审阅冻结 `style_spec.json` → Phase 1 普查协议（新 pass
  类型挂入本管线：findall/attr/review）+ 20 序列试点。

### 2026-09-05 Phase 0：test 句式挖掘（草案待审）

- 动因：v5 定案的 Phase 0——语法克隆的事实基础；只读 queries.json 文本，不触 GT。
- 改动：`scripts/phase0_mine_test_style.py`（纯 stdlib，可复现）、
  `spec/style_spec.json`、`spec/vocab_freq.json`。
- 验证：总数 9555 与既有读数一致；重复率 0.076、词数均值 10.33、序数桶 335.6‰
  均与既往 handoff 基线互洽（7.6% / 10.3 / 33.5%）。
- 关键读数（供管理员审阅 spec 时对照）：
  - 框架集中：`from left to right` 496 / `from right to left` 449 近对称成对；
    `on the left/right side of the` 家族 ~600；`closest to the camera` 354 对
    `farthest from the camera` 39（5.5:1 不对称）；`the far right/left` 288/255
    是 extreme 位置的独立主力框架；`of the image` 303 作画幅锚点；camera 锚定 691。
  - 深度行句式在 test 几乎不存在：`front row`/`back row` 各 1 次，`row` 43 次全是
    物理行（a row of X）。现行 prompt 的 `depth row` cue 不属官方方言。
  - 桶级（草案文本分类器）：序数 335.6‰、空间 329.0‰、属性动作 232.8‰、距离 102.6‰。
  - 骨架多样性：unique 5353，top60 覆盖仅 27%——长尾在 <X> 槽位填充，框架本身集中。
  - MT 错误低频：非 ASCII 中文夹杂 32 次（唯一过 ≥20 阈值的错误怪癖）、邻接重复词
    14、撇号脱落（mans）8、冠词误用 4、双空格 3；高频层的 MT 特征在框架结构本身，
    不在语法错误。
- 下一步：管理员审阅冻结 style_spec.json → Phase 1 普查协议 + 20 序列试点方案。

### 2026-09-05 建仓

- 动因：v5 生成策略定案（三权分立），训练全流程只吃标注产物，生产线拆出主仓。
- 改动：README（边界与红线）、docs/handoff.md、scripts/、spec/ 骨架；git init（main，无 remote）。
- 验证：git init 成功；尚无代码，无测试。
- 下一步：Phase 0 挖掘 → style_spec.json 草案 → 管理员审阅冻结 → Phase 1 普查协议草案。
