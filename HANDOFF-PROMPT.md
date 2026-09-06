# 交接提示词 — query-foundry 专业化审阅（写给下一个 agent）

> 使用方式：把本文件全文作为新会话的开场输入。本文件只陈述事实、位置与已发生
> 的裁决，不预设审阅结论——你自己的判断优先于文中一切表述。

## 任务

管理员指派：**审阅 `/home/fang0/dev/projects/query-foundry/`（v5 标注生产线），
将其专业化、解耦、规范化**。这是一个已经全链跑通并在产数据的活仓库，不是从零
开始的脚手架——先理解它为什么长这样，再决定什么该改。

## 仓库关系（三仓，先读每个仓的 AGENTS/README/handoff）

| 仓库 | 性质 | 与本仓关系 |
|---|---|---|
| `~/dev/projects/aicomp-multimodal-grounding` | 竞赛主仓（RGBDT 三模态定位，ACC@0.5 唯一指标） | 上游数据源 + 下游产物接收方；训练/评测/平台壳全在那里 |
| `~/dev/projects/query-foundry` | **本次审阅对象**：v5 query 生成生产线（私有、无 remote、永不 push） | 读主仓 `data/`，产物手动送回主仓 `outputs/annotations/` |
| `~/dev/projects/gt-analysis` | 灰色分析仓（test 侧 GT 逆向） | **禁止读取其数据文件**；本仓只允许引用其已发布统计口径；其内容永不写入任何其他仓 |

红线条文见 query-foundry 根 `README.md`（三权分立、API 调用须管理员逐次放行、
产物手动回主仓、key 永不入库）。静态规则见主仓 `AGENTS.md`。

## 现状快照（2026-09-07，数字以 handoff 与产物为准）

- 数据链：主普查 v3（train 320/320 + val 80/80 序列）→ 补数 pass v3.1
  （顶满帧双遍无上限重枚举，教师自一致率 82.8%/82.3%）→ 重排 → 颜色仲裁 →
  组装（不限量）→ 文本 QC → 全量人审（进行中，train 已 65 核验 + 5 待办）。
- 语料：train 2,730 条 / 913 帧 + val 736 / 233 帧（`outputs/assembly/asm-*-r4/`）。
- 两个 census run 目录（含 `merged.json` 与 `enumeration.json`）是组装器的事实
  源，**删除即断链**。
- 测试 96 项全绿；46 提交；两仓 handoff 已补至本轮。

## 已知的真实问题（管理员与上轮 agent 已确认的，供你核实的起点）

1. **文本 QC 裁决表是临时脚本**（`/tmp/fix_articles.py`，tmpfs 易失）：568+131
   处冠词/回声修复靠它重放。规范化解法 = 裁决表数据化入库 + 折进组装器实现层。
2. **审查器是 vendor 来的**（gt-annotator，MIT）：`foundry/review/` 里
   server/store/web 三件套带着源仓库的通用打标器语义（absent 判空、多人协作、
   翻译层）——本仓只用到其中一部分，死代码与审查模式开关（:todo、:needs 残留
   迁移）值得清理。
3. **census.py 与 assembly.py 边界**：组装器直接 import 普查的
   `pass_agreement`；事实提取（extract_frame_facts）与句族实现（realize_*）
   同文件 900+ 行——数据层已抽到 `foundry/facts.py`、桶分类器在
   `foundry/buckets.py`，但 policy 层仍可再分。
4. **scripts/ 入口幂等性不一致**：`run_census.py`/`run_enumeration.py` 有完整
   分片 checkpoint，`assemble_queries.py`/`review_report.py` 是单发脚本——
   规范化时统一约定。
5. **协议文档与代码的双源风险**：`spec/census_protocol.md`（v3.1 冻结）与
   `foundry/census.py` 内嵌提示词需人工保持同步，无机器校验。
6. shell 环境的 `http_proxy` 无 127 豁免（测试已内置覆盖；管理员浏览器走
   TUN/系统代理，访问 localhost 服务需其自行开关）。

## 管理员已裁决的事项（不要重新翻案，除非有新证据）

- 「伪框」概念已废除：教师框过确定性门即正式标注。
- 拥挤帧序数可信域：人审终裁路线（不设机械阈值）。
- 骨架多样性：挂起，训练后迭代项。
- 短语级超供：只要句子正确即无害，不做 rebalance。
- 全比赛大模型链路不启用推理模式；v5 训练 target = 纯 bbox。
- 消融双 run 对照：砍除（旧标注成绩即基线）。

## 行为约束（违反即事故）

1. **任何真实产生 API 调用的命令**（run_census / run_enumeration /
   check_keys 的 realistic 模式）必须有管理员当次明确指令；preflight-only
   （零调用）不受限。
2. 管理员自管审查服务进程（8788/8789）——**不要 pkill review_server**，那是
   他正在用的会话。
3. 冻结合同：`outputs/annotations/annot_dc189f029d962b27/**/approved.json`、
   主仓 `data/Test/queries/queries.json`（SHA 被钉死）、各 adapter 的
   MODEL_NAME/REVISION/prompt 常量。
4. 改任何代码后 `python -m unittest discover -s tests`（96 项）必须全绿。
5. 模型权重/大数据/key 明文永不入库；`outputs/` 与 `keys/api_keys.txt`
   已 gitignore，不得绕过。
6. 审查存储 `outputs/review/` 里的 journal 记录管理员的人工裁决——**只读，
   除非他明确要求重置**。
7. 本轮工作结束时按主仓 AGENTS.md 铁律 6 更新两仓 handoff 并提交。

## 环境事实

- 本地 Python：`~/miniconda3/envs/qwen_vg/bin/python`（3.12）；shell 里裸
  `python` 不存在。
- 无 GPU 任务（训练在魔搭 DSW / Modal，由管理员执行）。
- 典型耗时：组装+QC 全量 <1 分钟；补数 pass ~30 分钟（1,200 调用）；
  全量普查数小时（需放行）。

## 验收建议（供你自证审阅完成）

- 能在 30 分钟内向管理员说清：一条 query 从图片到训练样本经过哪些门、
  哪些数据结构承载它、回溯一条记录的出身需要查哪几个文件。
- 你的规范化改动有测试护航、有 handoff 记录、没有打断管理员正在进行的
  人审会话。
