# query-foundry 架构（2026-09-06，v5 生产线定型版）

> 本文件是模块布局与数据流的唯一真相；改布局先改本文件。
> v4 管线（generate_queries / annotation_state / query_style / query /
> audit_query_style 及其测试）已按管理员指令删除，历史在 git。

## 数据流（一条线，无分叉）

```
主仓 data/indexes/{train,val}.json ──→ run_census.py（Phase 1 普查）
        │ 教师只看：findall×2 交集 + 选中帧 attr 名片          │ 零 query 文本
        ▼
outputs/census/<run_id>/merged.json ──→ 人审（review 工具，R2 落地）
        │
        ▼
assemble_queries.py（Phase 2 组装，代码只说）
        │ 事实支撑 + 代码可验证唯一性 双门
        ▼
outputs/assembly/<tag>/assembly.json ──→ plan_queries.py（Phase 3 规划器，未实现）
        │
        ▼
打包（R6）→ 人审快照 → approved 产物包 → 管理员手动放回主仓
        → 主仓 validate_approved_artifact 校验 → 编 annotation run-id
```

## 模块地图

| 模块 | 职责 | 状态 |
| --- | --- | --- |
| `foundry/census.py` | 普查协议：提示词、响应解析、确定性门、坐标约定识别 | 冻结于 `spec/census_protocol.md` |
| `foundry/depth.py` | 深度事实：原始 uint16 毫米图 → 对象中值/排名/前后景分带（零图片落仓） | 冻结于 `spec/census_protocol.md` |
| `foundry/review/` | 人审工具（vendor 自 gt-annotator，MIT）：崩溃安全 journal/snapshot 存储 + 审查服务；
  教师框以 AI 预标播种，人拖框调整实时写回，框不可删，全帧人核即 reviewed | `scripts/review_server.py` 入口 |
| `foundry/assembly.py` | 组装器：事实提取 + 句族实现 + 唯一性门 + 临时配额器 | 配额器待 Phase 3 规划器替换 |
| `foundry/source.py` | split 索引装载 + 三类指纹（preparation/image/source） | 稳定 |
| `foundry/keys.py` `api_client.py` | key 池（逐请求轮转、分级退役）+ API 客户端 | 稳定 |
| `foundry/annotation_views.py` | 红框/编号框 PIL 渲染（census 取图与 key 测活共用） | 稳定 |
| `foundry/io.py` `bbox.py` `images.py` `artifacts.py` `sharding.py` `config.py` | 共享工具（自主仓复制的冻结副本，改前先看主仓原件是否同步） | 冻结副本 |
| `scripts/run_census.py` | 普查入口：分片并发、checkpoint、resume、report | 稳定 |
| `scripts/assemble_queries.py` | 组装入口：纯本地零 API | 稳定 |
| `scripts/check_keys.py` | key 测活（realistic / minimal 两档） | 稳定 |
| `scripts/phase0_mine_test_style.py` | test 句式挖掘 + **冻结四桶分类器**（assembly import 其正则） | 只读事实源 |
| `spec/style_spec.json` `spec/vocab_freq.json` | test 方言规范（frozen-2026-09-05） | 冻结 |
| `spec/census_protocol.md` | 普查协议 v3：提示词、门清单、深度事实规则 | 冻结，改协议=改文档+版本号 |

## 红线摘要（全文见根 README）

- 教师只看，代码只说，规划器只分；教师永不写 query 文本。
- 真实产生 API 调用的命令须持管理员当次明确指令；preflight-only（零调用）不受限。
- 副作用类指令（API、安装、push、平台操作）执行前停下，由管理员决定谁来跑。
- val 与 train 同生产线同结构；产物手动放回主仓，永不自动推送。
- key 明文永不入库、永不进日志。

## 已删除（git 可溯）

`scripts/generate_queries.py`、`foundry/annotation_state.py`、
`foundry/query_style.py`、`foundry/query.py`、`scripts/audit_query_style.py`
及 6 个对应测试文件——v4「教师写 query」管线的全部残余；
`foundry/sequence.py` 瘦身至 `source_fingerprint`（v4 响应解析三件套随链删除）。
