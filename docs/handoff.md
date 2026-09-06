# 交接文档（query-foundry）

> 本仓 = 主仓 v5 query 生产线。边界与红线见根 README。
> 上游定案见主仓 `docs/handoff.md` 与 `/tmp/handoff-2026-09-05-query-foundry.md`（2026-09-05 会话交接）。

## 当前状态（2026-09-06，census v2 全景协议落地 + 试点关账 + val 预检就绪，仓库已入库）

- census v2 全景协议已实现并入库：findall 跨类别全景枚举（每目标带类别名），
  金丝雀（找回 GT 目标 IoU≥0.5）为唯一帧级失败判据；乱序→代码排序重编号、
  自重复（IoU≥0.95）→去重、零面积框→丢弃，均不再杀帧。key 池 v2：逐请求
  轮转（12 key = 12 账号）；401/402/403 立即退役并注释回写 key 文件；429
  只退避（至多 8 次，Retry-After 优先），累计 30 次持续限流才退役；传输失败
  连续 2 次挂起（本 run 生效，不写文件）；测活 `scripts/check_keys.py`。
- 试点 `census_7108aac8c25365a9` 关账：178/181 帧完成（98.3%）、mean
  jaccard 0.911、attr 成功率 95%（57/60）、金丝雀失败 3 帧废弃；60 张选中帧
  能力卡人审通过；供给 = 教师目标 215 + 真框 60 = 275（≈16 条/序列，320
  序列外推 ≈1.7 倍于 2880 需求）；序数素材 39/60 选中帧。
- 第四批（train 全量 320 序列）已预检未执行：`census_8d8e87efac3f5bb3`
  （16 分片，run-tag `census-full-1`），命令待管理员放行。
- **val 普查已预检**：`census_63bca023afdc208e`（80 序列 / 719 帧，16 分片，
  run-tag `census-val-1`，零调用预检通过）。管理员定案：val 与 train 分开
  跑、同协议同命令形状、串行接力（train 完成后 val 接力），不并行。外推
  成本 ≈1,900 调用 / ≈400 万 token / ≈2-3 小时。val 组装时
  `--max-teacher-per-frame` 放开（红线：过门目标全进 val，密度 ≈3 条/帧）。
- Phase 2 组装器已实现并入库（`foundry/assembly.py` + CLI + 23 项测试），
  试点试跑 133 条、逐字重复 0.0、桶占比贴合 test、词数 7.26 vs 10.33 留
  Phase 3 规划器补。规划器与双 run 消融随后。
- 管理员定案（2026-09-06）：总量 ≈2880 为刻意约束（挑帧去相似 → 丢量 →
  3 帧/序列 × 3 目标补回旧训练量级 2875）；全比赛大模型链路不用推理模式，
  v5 训练 target = 纯 bbox（主仓待决分叉 ③ 就此关闭）。

## 交接日志

### 2026-09-06（R1 执行：census v3 + 原始毫米深度事实，冒烟预检就绪待管理员放行）

- 动因：按冻结的 `spec/census_protocol.md` v3 落代码。管理员两次方向纠正
  已吸收：① 深度事实读**原始 uint16 毫米图**（`Train/<seq>/depth/`，由
  visible 路径 color→depth 段推导），不解码 depth_jet 伪彩图（逐帧归一化
  跨帧不可比 + 无效像素涂黑与 jet 蓝端混淆两坑）——毫米值精确，无降级阶梯；
  ② 全程零图片落仓，census 的 preview 卡片渲染路径整体删除（审查器 R2 落地，
  直读 merged.json + 源图渲染）。
- 改动：
  - `foundry/depth.py` 新增：对象 bbox 内有效像素（>0）深度中值 → 帧内排名
    （1=近）+ 前景/背景三分带；`frame["depth"]` 以 JSON 数字进 merged.json。
  - `foundry/census.py`：findall 提示词 v3（上限 6 个最有把握目标 + 红框同类
    先列满）；`draw_census_card` 删除。
  - `scripts/run_census.py`：协议版本 2→3（run-id 指纹随之更新）；每帧集成
    深度事实（文件缺失帧记 `depth=unavailable`，组装器退回 y2 代理）；每帧
    状态行增 conv/depth/latency 字段；`--concurrency` 默认 48、
    `MAX_API_CONCURRENCY` 16→96（12 账号 × 官方 8 并发上限）；peers/preview
    残余清理。
  - `foundry/assembly.py`：消费深度记录——closest/farthest 改毫米余量
    （≥200mm）判定替代 y2 代理；前景/背景分带实现 + 句族
    （"The swan in the foreground"）；前景/背景主张的唯一性门 = 同头名词
    目标在带内唯一（分带非独占，不能按构造唯一）。
  - 真实数据 sanity：070_00000001 GT 框深度 12,671mm（帧跨度 4,167-19,999mm，
    物理合理）；val 索引路径推导同样成立（Train/004/...）。
- 验证：本仓 82 项测试 OK（+11 项深度测试）；compileall 通过；v3 冒烟预检
  通过：run `census_3038171f8c20c6bd`（20 序列/16 分片，零调用）。
- 注意：v2 时代的第四批/val 预检 run（`census_8d8e87efac3f5bb3`、
  `census_63bca023afdc208e`）指纹已过时作废，v3 全量命令以新预检为准。
- 执行点（管理员定）：冒烟真跑命令 =
  `python scripts/run_census.py --split train --limit-sequences 20 --concurrency 48 --num-shards 16 --run-tag census-v3-smoke`

### 2026-09-06（R0 架构瘦身执行：v4 链删除 + 协议 v3 冻结 + 定案落档）

- 动因：管理员定案「仓库专业化规范化、消除过度设计」——弃用代码不冻结直接删
  （git 可溯）；副作用类指令执行前停下由管理员定谁来跑；常规本地验证直接执行。
- 定案落档（本轮对齐会话全部拍板项）：普查上限 6 目标 +「红框同类先列满」措辞；
  并发 48（4/账号，官方上限 8 的一半），逐请求轮转；R0-R7 轮次序批准；消融
  双 run 对照砍除（旧标注成绩已存在即基线，出身字段保留可随时切对照臂）；
  深度事实加进 R1（jet 解码三档粗粒度，教师不参与，服务前景/背景消歧）；
  审查器搬 gt-annotator 机制进本仓（census 60 张冒烟 + 组装 400 张全量两模式，
  框只拖不删、实时写回）。
- 改动：
  - 删 v4 链（44 项测试随链）：`scripts/generate_queries.py`、
    `foundry/annotation_state.py`、`foundry/query_style.py`、
    `foundry/query.py`、`scripts/audit_query_style.py` + 6 个测试文件；
    `foundry/sequence.py` 瘦身至 `source_fingerprint`（v4 解析三件套随链删）。
  - 新增 `foundry/source.py`：自 v4 入口吸收 3 个数据装载/指纹函数
    （load_annotation_source / preparation_fingerprint / image_fingerprint），
    `run_census.py` 导入改指 foundry，v4 依赖清零。
  - 新增 `docs/architecture.md`（模块地图 + 数据流 + 删除清单）、
    `spec/census_protocol.md`（协议 v3 冻结：findall v3 提示词、门清单、
    深度事实规则、并发预算）；README 用法段重写为 census/组装/check_keys
    三入口。
- 验证：本仓 74 项测试 OK（118 − 44 项 v4 测试）、compileall 通过。
- 下一步：R1 = census v3 代码落地（findall v3 提示词 + 深度事实 + 每帧状态行）
  → 20 序列冒烟预检交管理员放行。

### 2026-09-06（管理员三项定案：val 分开跑串行、2880 刻意约束、全链无推理模式；仓库入库）

- 动因：管理员方向对齐轮拍板三项，并指令将工作区改动全部提交。
- 定案一：**val 普查与 train 分开跑**——同协议、同命令形状、串行接力
  （train 完成后 val 接力），不并行。理由：主仓入库合同按 split 分目录、
  run-id 按 split 溯源；checkpoint/resume 按 run 隔离；避免双 run 叠加
  并发压力（历史撞车教训）。val 预检已通过：`census_63bca023afdc208e`
  （80 序列 / 719 帧 / 16 分片 / run-tag `census-val-1`），外推 ≈1,900
  调用 / ≈400 万 token / ≈2-3 小时。val 组装放开
  `--max-teacher-per-frame`（红线：过门目标全进 val）。
- 定案二：**总量 ≈2880 为刻意约束**——挑帧去除相似帧丢失训练量，3 帧/序列
  × 3 目标 = 2,880 刻意补回旧训练量级（golden 2,875），维持与 v4 基线
  run 的可比性。接管方核对：试点教师供给 ≈3.6 目标/选中帧 vs 需求 2/
  帧，320 序列供给充足；方法学提示（供 Phase 4 设计）：消融双 run 的
  「仅真框」臂 ≈960 条与「含扩充目标」臂 ≈2,880 条同时差在内容与体量，
  分差归因时体量是混杂变量，届时定对照口径。
- 定案三：**全比赛大模型链路不用推理模式**——v5 训练 target = 纯 bbox，
  主仓「v5 训练 target 是否含枚举推理链」待决分叉就此关闭。
- 改动：`.gitignore` 增 `.vscode/`；仓库按 4 个提交入库（keys 池 v2 /
  census v2 / 组装器 / 文档）；主仓 handoff 同步分叉关闭。
- 验证：提交前本仓 118 项测试 OK、主仓 171 项 OK。

### 2026-09-06（Phase 2 组装器实现 + 试点试跑，未提交）

- 动因：/tmp 交接指定的第四批空档期工作——「代码只说」：query 文本由本地
  组装器从 census 事实拼装，教师不写句子。
- 改动：新增 `foundry/assembly.py`（事实提取 + 骨架实现 + 配额分配 + 验收
  审计）、`scripts/assemble_queries.py`（CLI）、`tests/test_assembly.py`
  （23 项测试）；README 增组装器用法。桶分类器直接 import 冻结的
  `phase0_mine_test_style` 正则，口径与已发布计数逐字节同源。
- 机制：每帧 1 真框目标（记录带组织者 GT 框）+ 最多 2 教师目标（attr 质量
  分排序、面积门 0.002-0.6）；每条句子双门——事实支撑 + 代码可验证唯一性
  （描述在本帧只解析到一个对象；attr 缺失的同头名词对象按潜在匹配保守
  处理）。事实类型：head 名词组序数（相邻秩中心距 ≥0.02）、极值/超级词
  （center-x/y、底边 y2 邻近代理，余量 ≥0.04）、画幅 thirds 侧位、唯一头
  名词锚点方位（全间隔 ≥0.01）、attr 颜色/特征词。防歧义硬规则：swan 与
  black swan 按 head 合并去歧义组；person/child/man/woman 等头名词抑制裸
  色词（"the white person" 类 racial 歧义表述不产出）；features 自带
  "{head} with" 前缀剥离；任意 -ing 开头特征走动作句式。分配器：冻结份额始
  终化 → 配额耗尽记 overshoot（保留供给不浪费）→ 无解记 shortfall，均不伪造。
- 试点试跑（`asm-pilot-trial`，输入 census_7108aac8c25365a9，零 API）：133
  条（real 50 / teacher 83）、17 序列；逐字重复 0.0（验收 ≤0.10，test
  0.076）；桶占比 vs test：序数 406/335、距离 135/151、空间 233/258、属性
  动作 226/256（per-mille）；词数均值 7.26 vs test 10.33。shortfall 12 =
  9 失败序列帧 + 3 无唯一实现（含 1 条 2 词裸类别被词窗 3-18 过滤）；
  overshoot 8。真框记录 50/51（1 条无 ≥3 词唯一实现）。
- 验证：本仓 118 项测试 OK（95 + 23 新增）。
- 遗留（Phase 3 规划器杠杆）：词数 7.26 vs 10.33 差距靠更丰富骨架实现
  （双属性栈、"wearing a {color} {garment}" 模式）与配额偏好调整补，不硬编；
  距离桶 y2 底边邻近代理对高树类目标有先验风险，人审卡重点抽查。

### 2026-09-06（接管会话核查：试点数字逐项复算一致；peers=0 根因修正）

- 动因：新会话接管，按管理员要求对 /tmp 桥接文档与既有 handoff 声明逐项
  实测核查（不轻信文档）。
- 核查结果（数据源 `outputs/census/census_7108aac8c25365a9/merged.json` +
  `report.json`）：178/181 帧、jaccard 0.911、attr 57/60、471 调用、
  金丝雀失败帧（178_00000147 / 281_00000438 / 350_00000595）全部与文档
  一致；供给账复算 = agreed 对象重建后 266 − 51 金丝雀 = 215 教师 + 60 真
  = 275，与「215 + 60 = 275」一致；序数帧 39/60 = 65%（口径 = 全部 20 序列
  各 3 选中帧）；preview 78 张中 60 张为选中帧卡（20×3）。
- peers=0 根因修正：上一条目疑因「跨帧 IoU≥0.5 对帧间微动过严」，实测
  否定——peers=0 序列的跨帧最佳 IoU 分布为 0.0 ×67 / 0.01-0.3 ×73 /
  0.3-0.5 ×12（共 152 对），目标框中心位移 0.12-0.44（归一化），帧号跨度
  64-754。是运动目标（swan/deer/duck/lemur/person 等）在相距数百帧的
  选中帧间宏观位移，非微动；降阈值救不回 0.0 重叠。census v2 已删除跨帧
  身份依赖（每样本只由本帧普查事实支撑），该威胁随协议重设计消除，无需
  调阈值。
- 主仓 171 项测试 OK（4 skip 为 GPU 项）；本仓 95 项 OK；无残留生成进程；
  key 文件 12 把全活（仅 2 行文件头注释）。

### 2026-09-06（census v2 全景协议 + key 池 v2 + 试点关账，未提交）

- **动因**：管理员 2026-09-06 定案 v5 标注哲学演进——「伪框」概念废除：
  教师框过确定性门（findall×2 一致 + 金丝雀 + 人审卡）即为正式标注，与
  组织者真框同等直接入数据；val 与 train 同生产线同结构（挑帧 + 普查 +
  过门目标 + v5 文本）；跳帧挑 3 帧/序列，总量对齐旧训练量 ≈2880。
- **改动**（全部在工作区，未提交）：
  - `foundry/census.py`：findall 从「红框同类枚举」改「跨类别全景枚举」
    （每目标带类别名）；乱序门在全景口径下曾杀 27/146 帧，改为代码排序
    重编号；自重复（IoU≥0.95）改为去重；零面积框直接丢弃；三者均不再
    杀帧，金丝雀成为唯一帧级失败判据。类名漂移（swan/duck）留给组装器。
    协议版本 1→2。
  - `foundry/keys.py` + `foundry/api_client.py`：固定顺序改逐请求轮转；
    401/402/403 立即退役并自动注释回写 key 文件（原子写、保 0600）；
    429 只退避（Retry-After 优先，至多 8 次）绝不退役，累计 30 次持续
    限流才退役；传输失败连续 2 次挂起该 key（本 run 生效，不写文件）。
    旧逻辑两次 429 即永久退役且固定顺序池把全部并发压在第一把 key 上，
    管理员实测 key#1 被 HTTP 429 误退役。
  - 新增 `scripts/check_keys.py` + 测试（realistic 真实形状调用 / minimal
    1-token 探针）；`scripts/run_census.py` 新增独立 `--num-shards`（run
    指纹含 requested_num_shards，续跑须与首跑一致；曾因 `--concurrency 16`
    隐式改分片数导致 checkpoint 未接上）。
  - README 红线段改写对齐新定案。
- **试点终报**（run `census_7108aac8c25365a9`，20 序列）：178/181 帧完成
  （98.3%）；count_agree 81.0%、mean jaccard 0.911、attr 95%（57/60）；
  单遍兜底 0 帧；3 帧金丝雀失败废弃（各 6 次独立采样未找回红框目标）；
  471 调用 / 105 万 token。60 张选中帧能力卡人审通过（唯一发现：169 序列
  地面杂物被标 shoe → findall 提示词加「只列能确信命名、轮廓清晰的目标」
  指引，已改已测）。管理员结论：金丝雀 98.3% + jaccard 0.911 + 人审
  60/60 三层数据支撑「现在能产生合格框」。
- **验证**：本仓 95 项测试 OK（此前 51 项）；`outputs/census/` 清理后仅存
  试点与已预检未跑的 `census_8d8e87efac3f5bb3`。
- **下一步**：第四批全量（320 序列，`--concurrency 16 --num-shards 16
  --run-tag census-full-1`，已预检，命令待管理员放行）→ Phase 2 组装器
  （空档期开工）→ 规划器 → 双 run 消融。

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
