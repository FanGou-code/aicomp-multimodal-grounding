# 交接文档（query-foundry）

> 本仓 = 主仓 v5 query 生产线。边界与红线见根 README。
> 上游定案见主仓 `docs/handoff.md` 与 `/tmp/handoff-2026-09-05-query-foundry.md`（2026-09-05 会话交接）。

## 当前状态（2026-09-07，R0-R4 全链路完成，全量人审进行中，R6 打包链未建）

- **生产线全链已通**：主普查（v3，train 320/320 + val 80/80）→ 补数 pass（v3.1，
  train 549 + val 134 顶满帧双遍无上限重枚举，双遍计数一致率 82.8%/82.3%）→
  重排（幻影框淘汰/场景真值序数/不一致帧禁序数）→ 颜色仲裁（同头 ≥2 同色剥除）
  → 组装（教师目标不限量全收，`--max-teacher-per-frame -1`）→ 文本 QC（冠词/
  回声裁决表重放）→ 全量人审（审查器，query 可编辑）。
- **当前语料**：train 2,730 条 / 913 帧（`asm-train-r4`）+ val 736 条 / 233 帧
  （`asm-val-r4`）= 3,466 条（设计量 3,600 的 96%）。四桶 368/100/248/284 与
  344/140/253/264（test 335/151/258/256）；逐字重复 0.11%/0%；帧内唯一 100%；
  96 项测试全绿；46 个提交。
- **人审进度**（outputs/review/）：train 65 条已核验 + 5 条待办（reviewer 署名
  真实有效），val 未开始；种子全部 glm-4.6v 署名。审查器 = vendor gt-annotator
  （MIT）+ census/assembly 双模式 + 全量帧模式 + query 可编辑（journal 持久化 +
  声明行同步）+ 需消歧待办（:todo 后缀）。
- **两个既有 census run 的用途**：`census_da571f4a3c91eb22`（train，含
  enumeration.json）与 `census_c85c9d9b1c74bf85`（val，含 enumeration.json）——
  merged.json + enumeration.json 是组装器的事实源，**不许删**。
- **审查器启动**（管理员自管进程）：
  `python scripts/review_server.py --assembly outputs/assembly/asm-train-r4/assembly.json --port 8788`
  （val 用 asm-val-r4 + 8789）。声明行 = 方向箭头 ◀▶ · 序数词 · 主体（从 query
  逐字摘取）。Enter 核验 / P 进待办 / query 输入框直接改。
- **下一步（R6）**：人审完 → 处理 :todo 清单（补数计数上下文消歧或废弃）→
  人工 query 修改合并 → 快照 → approved 产物包 → 主仓合同校验 → α32 重训
  （主仓 SOP）。
- **已知债务**：① 文本 QC 裁决表在 /tmp（易失），重组装前应折进组装器实现层；
  ② 拥挤帧序数可信域问题（教师自一致性 ~83%）——当前每条序数都过补数重排，
  但 test 侧模型数数能力仍是最大变量；③ shell 的 http_proxy 无 127 豁免——
  测试已内置 no_proxy 覆盖，管理员浏览器访问 localhost 需注意代理 TUN 模式。

## 交接日志

### 2026-09-07（人审工具完全体：全量模式 + 可编辑 query；语料放量至 3,466）

- 语料放量：`--max-teacher-per-frame -1`（教师目标按质量排序全收）→
  train 2,730 + val 736 = 3,466 条（设计量 96%），全部门禁不变。
- 审查器三轮迭代：① 全量帧模式（`--all-frames`，1,146 帧全上架、同序列
  相邻排列、人工裁决保留不重播种）；② query 可编辑（`PUT /api/item/<id>/query`
  + journal 持久化 + 重启保留 + 声明行主体实时重摘——主体改为从 query 逐字
  摘取，修掉 attr 肤色污染与序数重复两处声明失真）；③ 声明行 = 方向箭头
  ◀▶ · 序数词 · 主体，参照物文字保留。
- 事故两起：① run_enumeration 初版把 ENUMERATION_PROMPT 定义了却仍发
  findall 的 6-cap 提示词（315 帧空转报废，管理员抓出）；② store 快照重放
  元组升 4 元后漏改 3 处解包（测试全捕获）。测试侧补 localhost no_proxy
  覆盖（管理员 shell 的 http_proxy 无 127 豁免，会 502）。
- 审查服务由管理员自管（8788/8789）；train 已审 65 核验 + 5 待办。
- 管理员裁决记录：拥挤场景序数可信域问题采「人审终裁」路线（补数计数作
  上下文，不设机械阈值）；落选目标「补属性消歧 → 终审 → 废弃」三级流程；
  骨架多样性挂起为训练后迭代项；短语级超供只要正确即无害，不做 rebalance。

### 2026-09-06（语料审计 + 文本 QC 三轮：669 处修复，残留 6 条留人审）

- 动因：管理员要求对照 11 版口径复审语料（重复率/四桶/词数/骨架多样性/
  方向枚举/extreme/nearest-farthest），审计暴露语法层缺陷后授权修复。
- 新工具：`scripts/compare_distributions.py`（四库对照：test/旧v3/golden v4/
  v5）、`scripts/audit_corpus.py`（五组审计：schema 一致性/文本层/分布/
  11 版尺子/帧内唯一性，产物 corpus_audit.json）。
- 审计结论：11 版四个结构性病症全部治愈（序数 43→363‰、属性动作 717→276‰、
  方向枚举 0→223‰、extreme 157→36‰ 贴平 test）；暴露两个 v5 自身短板——
  骨架多样性（train 183 unique / top60 覆盖 74%，test 5353/27%）与词数
  7.54 vs 10.33。管理员裁决：test 长尾大半是中译英噪声，频率视角 v5 模式
  覆盖 test 语句 21.4%（test 自身 top140 = 33.7%），正确性优先，骨架扩展
  按「训练后迭代」挂起；短语级超供（from left to right 112 vs 52 等）
  只要句子正确即无害，不做 rebalance。
- 文本 QC（三轮，train 445 + val 134 = 579 处编辑，全程审计日志）：
  ① 冠词/回声 566 处（裁决表：单数可数插 a/an、复数物质裸形、with-in/
  perched/on 链条掉坑、身体部位 its、类名下划线归一 31 处）；② 头词回声
  逐条裁决 131 处（特征复述目标 → 折叠回主句；指向另一物体 → another/it
  语法化）。残留 6 条为诚实的双物体事实句或教师噪声，保留给抽帧人审。
- 终态（asm-train-r4 2,522 条 / asm-val-r4 671 条）：逐字重复 0.36%/0.15%、
  帧内唯一 100%、双冠词/复数误加/下划线 0、桶占比 363/107/255/276 与
  346/136/253/265（test 335/151/258/256）。
- 备注：修复脚本在 /tmp（一次性）；若将来重组装，裁决表应折进实现层。
  下一步：抽帧人审（400 帧）→ R6 打包链。

### 2026-09-06（val 全量收官 80/80 + R4 规划器落地；key 池二次风机制）

- val 全量（run `census_c85c9d9b1c74bf85`，80 序列 / 719 帧）：序列 **80/80**
  （新门生效，175 序列带 4 个未选中帧失败仍全收）；帧 690/719、count_agree
  77.7%、jaccard 0.836、attr 92.9%；序数素材帧 189/240 = 79%；供给 = 教师目标
  1188（5.0/帧）+ 真框 240 = 1428 → val 需求 720 的 **1.98 倍**；1,998 调用 /
  4.5M token。事故：服务端抖动致全部在飞 key 集体传输挂起、池子抽干
  （659/800 处崩）→ keys.py 加「二次风」：挂起 key 在池子枯竭时自动复活
  恰好一轮（硬退役如 401/429×30 永不复活），回归测试双路径（6143a97）。
- R4 规划器落地（`foundry/planner.py` + 分层重构）：
  - 分层：数据类型（ObjectFacts/Realization）下沉 `foundry/facts.py`，
    冻结桶分类器与配额下沉 `foundry/buckets.py`（消除 assembly↔planner 循环
    导入）；assemble_run 改为「供料 → 规划器分配 → 记录」三段。
  - 策略：冻结配额始终化 + 溢出标记；拥挤帧规则（同头名词 ≥3 且深度可用 →
    分带句所属桶优先；实测 foreground/background 落 attribute_action 桶，
    动态解析不假设）；面积比较级事实（同头名词面积比 ≥1.5 → "the larger X"，
    ≥3 对象 → "the largest X"，无绝对阈值）；长句优先 + 序列内句族多样性
    惩罚；逐字去重全 run 唯一。
  - train 试组装（`asm-train-r4`）：2,522 条（real 907 / teacher 1,615），
    逐字重复 0.0，桶占比 363/107/255/276 vs test 335/151/258/256（per-mille），
    词数均值 7.54（受实现丰富度上限约束，诚实留档）；shortfall 226。
- 验证：96 项测试 OK（+9 规划器测试）。
- 下一步：抽帧人审（审查器直接可用，train+val 数据都在）→ 打包入库链
  （R6）→ 主仓 α32 重训。

### 2026-09-06（第四批 train 全量收官：320/320 序列，供给 1.96 倍）

- 执行：管理员放行，agent 代跑（tmux 会话 census，跑完自动退出）。中途两次
  事故均修复：① 服务端瞬断（RemoteDisconnected）穿透重试层炸 run → api_client
  传输重试清单补 ConnectionResetError/HTTPException，附回归测试（fcd6c4e）；
  ② 序列门过严（113 个失败帧全在未选中帧、47 个序列的选中帧全部健康却被
  整序列废弃）→ 门改为「3 选中帧全完成 = 序列完成」（帧废弃不连坐序列，
  28602e9），checkpoint 迁移平反 45 个序列。
- 终报（run `census_da571f4a3c91eb22`，train 320 序列 / 2875 帧）：
  序列 **320/320** 完成；帧 2763/2875（112 个失败帧全部为未选中帧，按红线
  废弃）；count_agree 79.6%、jaccard 0.837、attr 93.8%、单遍兜底 27 帧；
  目标数 1-6 严格截断（6 目标帧 531 个）；序数素材帧 80%（762/956）；
  供给 = 教师目标 4675（4.9/帧）+ 真框 956 = 5631 → 2880 需求的 **1.96 倍**；
  调用 7830 / 17.8M token。key 池 1 把因 sustained 429×30 自动退役
  （11/11 alive 余量充足），key 文件已回写注释。
- 下一步：val 全量（`census_c85c9d9b1c74bf85` 已预检 80 分片，待管理员
  放行）→ R4 规划器 → 组装 → 抽帧人审（400 张）→ 打包入库。

### 2026-09-06（冒烟人审完成 + 清残留 + 96 分片重预检；R4 记两条新规则）

- 冒烟人审关账（管理员）：240/240 目标、51/51 整帧全人核，署名入审计链；
  管理员评价对照其 test 人工逆向经验「绝大多数非常完美」。审查服务已关闭。
- 清理：`outputs/assembly/asm-v3-smoke`（审查前试组装）、冒烟 census run
  `census_3038171f8c20c6bd` 及其审查快照 `outputs/review/`（其 20 序列将被
  第四批重跑覆盖，全部数字已存档本文件）；48 分片的两个作废预检目录。
  outputs 现仅存两个 96 分片预检。
- 尾巴效应处理：分片数改 2 倍超订（96 分片 / 48 工人），工人完成即领下一
  分片，动态均衡；val 80 序列每序列 1 分片。分片数进指纹，重新预检：
  train = `census_da571f4a3c91eb22`（96 分片）、val = `census_c85c9d9b1c74bf85`
  （80 分片），命令待管理员执行。
- R4 规划器新规则（管理员审查反馈，两条）：① 同头名词目标多的帧优先选
  前景/背景深度分带句式消歧（草地人群场景）；② 同头名词面积比 ≥1.5 倍
  推导「the larger X」比较级事实（test 比较级 3‰ + 最高级 2‰ + large/small
  定语 57‰），绝对尺寸不做硬编码阈值。
- 审查口径定案（管理员）：组装后审查 = 每序列抽 1 帧 × 400（test 侧 320 +
  val 80），帧内抽量理由 = 同序列三帧同场景差距小 + 确定性门全量兜底；
  发现问题随审随改（工具实时写回）。

### 2026-09-06（R2 执行：census 人审工具落地，冒烟审查就绪待管理员）

- 动因：管理员纠正流程——冒烟的 60 张必须先人审再放行全量，审查器是前置件；
  并明确要求搬 gt-annotator 现有机制、深度用原始图不解码伪彩、审查零图片落仓。
- 改动：新增 `foundry/review/`（vendor gt-annotator MIT 核心：stdlib 服务端 +
  崩溃安全 journal/snapshot 存储 + canvas 前端，出处已注）+
  `foundry/review/census_session.py`（census run → 审查条目：选中帧全部
  交集目标，左到右排序）+ `scripts/review_server.py` 入口。教师框以 AI 预标
  （`glm-4.6v` 署名）幂等播种（重启不覆盖已完成人审）；人工拖/缩调整实时
  写回 `outputs/review/<run_id>/`；框不可删（判空/清除按钮与快捷键在审查
  模式禁用）；帧内全部目标人核即计入「整帧核验」；前端多框同图叠加 +
  序数标签 + 红虚线 GT 参照框。
- 验证：86 项测试 OK（+4 项审查测试：会话形状/播种幂等/人工调整持久化/
  无署名拒绝）；HTTP 端点实测——会话内图片 200（3.4MB）、目录穿越与会话外
  图片均 404；app.js `node --check` 通过。真实服务已在 8788 端口运行
  （冒烟 run，240 目标 / 51 帧）。
- 事故记录：审查器测试首版未启动 `serve_forever()` accept 循环，HTTP 请求
  在内核 backlog 永久排队导致两轮会话卡死；测试补服务线程 + 超时熔断后修复。
- 执行点（管理员）：浏览器打开 `http://127.0.0.1:8788/` 审查冒烟 51 帧
  （v3 冒烟实际选中帧数，20 序列 × 3 减失败序列缺口）；审查完成后放行
  第四批 train 全量（`census_7269e45d575d123e` 已预检）→ 串行 val
  （`census_51e2c51a8d79724e`）。

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
