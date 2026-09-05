# query-foundry

主仓 `aicomp-multimodal-grounding` 的 v5 query 生产线。私有仓，无 remote，永不 push。

## 职责

装整条 v5 生成线（三权分立：教师只看 / 代码只说 / 规划器只分）：

- Phase 0：test 句式挖掘 → `spec/style_spec.json`（语法规范）
- Phase 1：普查协议（教师 GLM-4.6V 只输出结构化观察，不写 query）
- Phase 2：本地组装器（query 文本由代码拼装，不由教师采样）
- Phase 3：规划器（按 style_spec 配额分配句式，分布审计）
- Phase 4/5：全量 run 与重训在主仓执行，本仓只交产物包

## 边界

- 可读主仓 `data/`：`Train` 图序列与 `data/Test/queries/queries.json` 文本（官方下发件）。
- 永不读 `gt-analysis`（其脚本、数据、结论一概不进本仓）。
- 产物包 = query + bbox + 出身字段 + 分布审计报告 + 语法版本 hash，交回主仓薄入库口；
  主仓校验后编 annotation run-id 落 `outputs/annotations/`。
- 语法版本 hash 进主仓 run 指纹（替代原 prompt hash），溯源链不断。
- API_KEY 只走环境变量，永不入库。

## 红线（v5 定案，改动须管理员确认）

- 教师框目标过确定性门（findall×2 一致 + 金丝雀 + 排序 + 无自重复 + 人审卡）即为正式标注，与组织者真框同等直接入数据；无"伪框"概念，标不对的帧整帧废弃换备选帧。
- 挑帧扩容：跳帧挑 3 帧/序列，每帧 1 条真框 + 教师新目标若干条，总量对齐旧训练量（≈2880）。
- 普查为跨类别全景枚举（帧内某类别 ≥2 实例即序数可行）；每个目标的句子只由其所在帧的普查事实支撑。
- val 与 train 同生产线同结构：跳帧挑帧 + 普查 + 过门目标全进 val + v5 文本重写，密度 ≈3 条/帧。
- 归因 = 双 run 对照（含扩充目标 / 仅真框，其余全同）。
- 只造事实支撑的句子；凑不够的句式缺口用别的句式补，不硬编（封顶只是保险丝）。
- MT 怪癖只学高频（机械阈值，如 test 中出现 ≥20 次），低频语病不学。

## 结构

```
foundry/  生产线包：管线状态机、API 客户端（含 key 池）、QC、提示词合同、
          复制的共享工具（io/artifacts/bbox/images/sequence/query/sharding/config）
scripts/  generate_queries.py（生成入口）/ run_census.py（census 普查入口）/
          assemble_queries.py（Phase 2 组装器，纯本地零调用）/
          audit_query_style.py（样式审计）/ check_keys.py（key 测活）
spec/     style_spec.json（test 句式规范）与后续语法版本
keys/     api_keys.txt（gitignored，key 池文档，一行一把，行序=调用序）
tests/    离线单测
docs/     handoff.md（状态与日志）
```

## 生成管线用法

**标注源索引在本仓**：`data/indexes/`（train/val.json + split_manifest.json，git 追踪）
与 `data/audits/`（SHA-256 剔除审计）。图像不搬运——`data/Train`、`data/Processed`
是指向主仓 `data/` 的符号链接，管线经 `--data-root`（默认本仓 `data/`）取图。
指纹校验按内容哈希进行，文件迁移不影响 golden 身世链。产物落在本仓
`outputs/annotations/`，由管理员手动复制进主仓同路径。

```bash
# key 池：一把钥匙一行，按行序固定调用顺序，耗尽自动换下一把
nano keys/api_keys.txt   # 一行一个 key；该文件被 gitignore，永不入库

# 预检（不花调用）：校验索引指纹、图引用，产出 plan.json
python scripts/generate_queries.py --split train --limit-sequences 20 \
  --run-tag pilot --preflight-only

# 试点（写 preview 图，人工审）
python scripts/generate_queries.py --split train --limit-sequences 20 \
  --run-tag pilot

# 全量 + 发布 approved.json；中断后重跑同一指令自动从分片 checkpoint 续跑
#
# 红线：凡真实产生 API 调用的命令（生成/审查/普查），必须持有管理员的明确
# 指令才可执行；preflight-only（零调用）不受此限。
python scripts/generate_queries.py --split train --publish --run-tag v5-train

# 样式审计
python scripts/audit_query_style.py --queries outputs/annotations/annot_<run_id>/train/approved.json --full
```

### Phase 2 组装器（纯本地，零 API 调用）

```bash
python scripts/assemble_queries.py --census-run outputs/census/census_<run_id> \
  --run-tag asm-<tag>
```

从 census run 的 merged.json 取事实（教师目标 + attr 属性 + 几何关系），每帧
选 1 真框目标 + 最多 2 教师目标，代码拼装 query 文本。每条句子双门：事实
支撑（本帧普查事实）+ 代码可验证唯一性（描述在本帧只解析到一个对象，杜绝
歧义监督）。桶配额按冻结 style_spec 份额始终化，配额耗尽记 overshoot、
无解记 shortfall，均不上报伪造。产物 `outputs/assembly/<tag>/assembly.json`
（含出身字段与事实溯源）+ `audit.json`（重复率/桶占比/词数 vs test 参照）。
人审与 Phase 3 规划器在其后。

key 池行为（12 key = 12 个独立账号，平台按账号维度限流，文档无 per-key 数字）：
每个请求轮转换下一把 key，各账号在飞请求 ≈1，远低于任何档位并发上限。
401/402/403 立即退役并**自动注释回写 key 文件**（行首加 `#` + 日期原因，原子写、
保留 0600）；429 只退避重试（至多 8 次）绝不退役，累计 30 次持续限流才退役并注释
回写；传输超时/网络错误连续 2 次挂起该 key——本 run 生效、不写文件（可能复活，
事后用 scripts/check_keys.py 复查）。全池耗尽快速失败退出并在终端明示，checkpoint
已落盘，补 key 后重跑同一指令即续。终端只显示 key 编号，永不显示明文。
也可用环境变量 `ANNOTATION_API_KEY_FILE`（指定文件）或 `ANNOTATION_API_KEYS`
（逗号分隔）替代仓库 key 文件。
