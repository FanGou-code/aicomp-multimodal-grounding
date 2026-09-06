# query-foundry

主仓 `aicomp-multimodal-grounding` 的 v5 query 生产线。私有仓，无 remote，永不 push。

## 职责

装整条 v5 生成线（三权分立：教师只看 / 代码只说 / 规划器只分）：

- Phase 1：普查协议（教师 GLM-4.6V 只输出结构化观察，不写 query）——`spec/census_protocol.md`
- Phase 2：本地组装器（query 文本由代码拼装，不由教师采样）
- Phase 3：规划器（按 style_spec 配额分配句式，未实现）
- 人审：浏览器审查器（搬 gt-annotator 机制，R2 落地）
- Phase 4/5：全量 run 与重训在主仓执行，本仓只交产物包

结构约定与数据流见 `docs/architecture.md`。

## 边界

- 可读主仓 `data/`：`Train` 图序列与 `data/Test/queries/queries.json` 文本（官方下发件）。
- 永不读 `gt-analysis`（其脚本、数据、结论一概不进本仓）。
- 产物包 = query + bbox + 出身字段 + 分布审计报告 + 语法版本 hash，交回主仓薄入库口；
  主仓校验后编 annotation run-id 落 `outputs/annotations/`。
- 语法版本 hash 进主仓 run 指纹（替代原 prompt hash），溯源链不断。
- API_KEY 只走环境变量，永不入库。

## 红线（v5 定案，改动须管理员确认）

- 教师框目标过确定性门（findall×2 交集 + 金丝雀 + 人审卡）即为正式标注，与组织者真框同等直接入数据；无"伪框"概念，标不对的帧整帧废弃换备选帧。
- 挑帧扩容：跳帧挑 3 帧/序列，每帧 1 条真框 + 教师新目标若干，总量对齐旧训练量（≈2880，刻意约束：挑帧去相似丢量，3×3 补回 golden 量级 2875）。
- 普查为跨类别全景枚举且上限 6 目标（教师最有把握者，红框同类先列满）；每个目标的句子只由其所在帧的普查事实支撑。
- val 与 train 同生产线同结构：跳帧挑帧 + 普查 + 过门目标全进 val + v5 文本重写，密度 ≈3 条/帧；val 普查与 train 分开跑、串行接力。
- 归因不以消融双 run 执行（管理员 2026-09-06 定案砍除）；每条样本保留出身字段（真框/教师目标），将来需要时可零成本切出对照臂。
- 只造事实支撑的句子；凑不够的句式缺口用别的句式补，不硬编（封顶只是保险丝）。
- MT 怪癖只学高频（机械阈值，如 test 中出现 ≥20 次），低频语病不学；全比赛大模型链路不启用推理模式，v5 训练 target = 纯 bbox。

## 结构

```
foundry/  生产线包：census 协议、组装器、API 客户端（含 key 池）、
          数据装载/指纹、红框渲染、复制的共享工具（io/artifacts/bbox/
          images/sharding/config）
scripts/  run_census.py（普查入口）/ assemble_queries.py（组装入口，纯本地
          零调用）/ check_keys.py（key 测活）/ phase0_mine_test_style.py
          （句式挖掘 + 冻结四桶分类器）
spec/     style_spec.json（test 句式规范）/ census_protocol.md（普查协议 v3）
keys/     api_keys.txt（gitignored，key 池文档，一行一把，行序=调用序）
tests/    离线单测
docs/     handoff.md（状态与日志）/ architecture.md（结构与数据流）
```

## 管线用法

**标注源索引在本仓**：`data/indexes/`（train/val.json + split_manifest.json，git 追踪）
与 `data/audits/`（SHA-256 剔除审计）。图像不搬运——`data/Train`、`data/Processed`
是指向主仓 `data/` 的符号链接，管线经 `--data-root`（默认本仓 `data/`）取图。
指纹校验按内容哈希进行，文件迁移不影响 golden 身世链。产物落在本仓
`outputs/`，由管理员手动复制进主仓同路径。

### Phase 1 普查（API 调用须管理员放行）

```bash
# key 池：一把钥匙一行，按行序固定调用顺序，耗尽自动换下一把
nano keys/api_keys.txt   # 该文件被 gitignore，永不入库

# 预检（不花调用）：校验索引指纹、图引用，产出 plan.json
python scripts/run_census.py --split train --limit-sequences 20 \
  --concurrency 48 --num-shards 16 --run-tag census-smoke --preflight-only

# 真跑（须管理员明确指令；续跑 = 同一命令重跑，--num-shards 须与首跑一致）
python scripts/run_census.py --split train --limit-sequences 320 \
  --concurrency 48 --num-shards 16 --run-tag census-full-1

# key 测活（realistic 真实形状调用 / minimal 1-token 探针，均真调用）
python scripts/check_keys.py
```

### 人审（census/组装/合并 train+val，纯本地零 API 调用）

```bash
python scripts/review_server.py --census-run outputs/census/census_<run_id> --port 8788
python scripts/review_server.py --assembly outputs/assembly/asm-train-r5/assembly.json --port 8788
# train+val 同时审（合并会话，进度目录仍按语料分开）:
python scripts/review_server.py --assembly outputs/assembly/asm-train-r5/assembly.json \
  outputs/assembly/asm-val-r5/assembly.json --port 8788
# 浏览器打开 http://127.0.0.1:8788/
```

教师框以 AI 预标（`glm-4.6v` 署名）预载，人工拖动/缩放调整后实时写回
`outputs/review/<run_id>/`（journal + 快照，崩溃可恢复）；框不可删除；
当前帧全部目标经人工核验后计入「整帧核验」。顶栏切换审查范围
[全部|train|val]，进度/跳转/搜索按所选范围计算。快捷键：`Enter` 核验并跳
下一条待审、`E` 进 query 编辑框、`H/L` 前后翻页、`J/K` 跳 AI 待审、`N` 跳
未标注、`G/Shift+G` 首尾、`P` 加入待办、`/` 图号跳转、滚轮缩放、拖拽平移；
编辑框内 `Ctrl+F/B/A/E/K` 为 linux 光标键（前移/后移/行首/行尾/删到行尾），
`Enter` 保存并退出编辑，`Esc` 放弃修改。红虚线 = GT 参照框。

### Phase 2 组装（纯本地，零 API 调用）

```bash
python scripts/assemble_queries.py --census-run outputs/census/census_<run_id> \
  --run-tag asm-<tag>
# 输出目录已存在时拒绝写入（防误覆写人审中的语料）；确要重写加 --force
```

从 census run 的 merged.json 取事实（教师目标 + attr 名片 + 几何关系），每帧
选 1 真框目标 + 最多 2 教师目标（val 放开，过门目标全进），代码拼装 query。
每条句子双门：事实支撑（本帧普查事实）+ 代码可验证唯一性（描述在本帧只解析
到一个对象，杜绝歧义监督）。桶配额按冻结 style_spec 份额始终化，配额耗尽记
overshoot、无解记 shortfall，均不伪造。组装后跑文本 QC（`foundry/text_qc.py`：
冠词/回声引擎 + `spec/text_qc_echo_table.json` 冻结裁决表，逐条落盘
`text_edits.json`）。产物 `outputs/assembly/<tag>/assembly.json`
（含出身字段与事实溯源）+ `audit.json`（重复率/桶占比/词数 vs test 参照）。

key 池行为（12 key = 12 个独立账号，平台按账号维度限流，官方上限每账号 8 并发）：
运行并发 48 = 每账号在飞 ≈4，低于上限一半。逐请求轮转换下一把 key。
401/402/403 立即退役并**自动注释回写 key 文件**（行首加 `#` + 日期原因，原子写、
保留 0600）；429 只退避重试（至多 8 次）绝不退役，累计 30 次持续限流才退役并注释
回写；传输超时/网络错误连续 2 次挂起该 key——本 run 生效、不写文件（可能复活，
事后用 scripts/check_keys.py 复查）。全池耗尽快速失败退出并在终端明示，checkpoint
已落盘，补 key 后重跑同一指令即续。终端只显示 key 编号，永不显示明文。
也可用环境变量 `ANNOTATION_API_KEY_FILE`（指定文件）或 `ANNOTATION_API_KEYS`
（逗号分隔）替代仓库 key 文件。
