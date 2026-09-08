# query-foundry 架构（2026-09-08，工具化定型版）

## 分层

| 层 | 位置 | 职责 | 依赖 |
|---|---|---|---|
| 工具层 | `foundry/review/` | 审查器（server + store + web） | 零 pip 依赖 |
| 管线层 | `foundry/pipeline/` | 普查 / 组装 / 规划 / QC | PIL, numpy |
| 共享层 | `foundry/bbox.py` `foundry/utils.py` | 坐标工具 / IO / 指纹 | 零 pip 依赖 |

## 数据流

```
prepare_split → census → assembly → text_qc → review → apply → package_approved
```

## 模块地图

| 模块 | 职责 |
|---|---|
| `foundry/utils.py` | IO（load_json / atomic_write_json）+ 文件指纹 + 管线常量 |
| `foundry/bbox.py` | 坐标计算（IoU / 归一化） |
| `foundry/review/server.py` | 审查服务 HTTP 后端（--manifest / --assembly / --census-run 三模式） |
| `foundry/review/store.py` | 崩溃安全 journal + snapshot 存储 |
| `foundry/review/census_session.py` | Census / assembly 会话构建器 |
| `foundry/pipeline/census.py` | 普查协议：提示词、响应解析、确定性门 |
| `foundry/pipeline/assembly.py` | 组装器：句族实现 + 唯一性门 + 目标选择 |
| `foundry/pipeline/planner.py` | 配额分配 + 句族多样性 |
| `foundry/pipeline/facts.py` | 帧级事实提取（ObjectFacts / Realization） |
| `foundry/pipeline/text_qc.py` | 文本 QC（冠词引擎 + echo 表） |
| `foundry/pipeline/contract.py` | 训练合同校验 + 指纹计算（协议版本 12，零 pip 依赖） |
| `foundry/pipeline/buckets.py` | 冻结四桶分类器 + 配额份额 |
| `foundry/pipeline/depth.py` | 深度事实提取 |
| `foundry/pipeline/api.py` | Key 池 + API 客户端 |
| `foundry/pipeline/views.py` | 红框渲染 + 图片指纹 |
| `foundry/pipeline/sharding.py` | 分片 + 序列解析 |
| `foundry/pipeline/source.py` | 标注源索引加载 + 指纹 |
| `scripts/review_server.py` | 审查服务入口 |
| `scripts/make_manifest.py` | 审查清单生成（--assembly / --source 双模式） |
| `scripts/apply_review.py` | 人审结果合并烘焙 |
| `scripts/package_approved.py` | 打包发布 approved.json（自动算 4 个 SHA-256 指纹，直通主仓） |
| `scripts/run_census.py` | 普查入口 |
| `scripts/assemble_queries.py` | 组装入口 |
| `scripts/prepare_split.py` | 数据划分（--seed --train-ratio） |
| `scripts/check_keys.py` | Key 测活 |
| `scripts/review_report.py` | 审查报告 |
| `configs/default/` | 默认风格配置（prompts + rules） |

## 已删除（git 可溯）

`run_enumeration.py`、`rerank.py`（枚举 pass 全链路）、`run_ai_review.py`（AI 预审）、
`generate_queries.py`、`annotation_state.py`、`query_style.py`、`query.py`、
`audit_query_style.py`（v4 管线）、`phase0_mine_test_style.py`、`audit_corpus.py`、
`compare_distributions.py`（test 分析）。
