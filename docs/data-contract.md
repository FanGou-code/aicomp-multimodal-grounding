# 数据合同

本文件定义 `data/` 布局、标注产物与提交格式的字段级约定。可执行版本在
`aicomp_grounding/annotation_state.py`（产物准入）、`test_data.py`（官方模板与
Test 准备）、`scripts/prepare_rgbdt.py`（预处理）。

## 目录布局

```text
data/                         运行输入，不入库
  Train/<sequence>/           color/  infrared/  depth/  groundtruth.txt
  Test/                       Images/{visible,infrared,depth}/  queries/queries.json
  Processed/                  Train/<seq>/depth_jet/  Test/depth_jet/（JET 伪彩深度）
outputs/                      运行产物，不入库
  annotations/<run_id>/       train/approved.json  val/approved.json
  training/<run_id>/          plan.json  checkpoints/  best/  last/  completed.json
  inference/<run_id>/         metadata.json  predictions.json  checkpoint.json
  ordinal_enum/<run_id>/      metadata.json  parse.json  instances.json  thinking.jsonl
  ordinal_resolve/<run_id>/   metadata.json  predictions_<model>.json
  fusion/<run_id>/            融合产物
  submission/<tag>/           submission.zip
```

`Train/` 与 `Test/` 是原始数据，`Processed/` 由 `tools/prepare_rgbdt.py` 生成。
划分与查重留痕在 `data/indexes/`（见「产物与责任边界」）。

## 三模态输入格式

| 模态 | 原始格式 | 读取方式 | 约束 |
| --- | --- | --- | --- |
| `visible` | 三通道 8 位无符号 | 按 RGB 读入 | 已时间同步、空间对齐，三路尺寸一致 |
| `infrared` | 三通道 8 位无符号（单通道热辐射灰度堆叠，无彩色语义） | 按 RGB 读入 | 同上 |
| `depth` | 单通道 16 位无符号，单位毫米，`0` 为无效读数 | 经 `Processed/*/depth_jet/` 的 JET 伪彩图读入 | 固定标定 `depth_scaling=fixed`、`min_depth_mm=300`、`max_depth_mm=20000` |

序数后处理的深度轴读**原始 16 位毫米图**（`Test/Images/depth/`、`Train/<seq>/depth/`），
不走 `Processed/` 的 JET 伪彩。

同一场景的三张图按 `visible → infrared → depth` 固定顺序送入模型。`Processed/` 缺失
或损坏时由 `scripts/prepare_rgbdt.py` 重新生成；深度源已是 8 位三通道伪彩图时按原样
复制，不做二次着色。

## 产物与责任边界

| 产物 | 生产者 | 消费者 | 交接方式 |
| --- | --- | --- | --- |
| `data/Train`、`data/Test`、`data/Processed` | 数据交付方 + `scripts/prepare_rgbdt.py` | 本仓训练/推理；上游标注流水线 | 本地目录 |
| `query-foundry/data/indexes/{train,val}.json`、`split_manifest.json`、`excluded_overlap.json` | `query-foundry/scripts/prepare_split.py` | 上游标注流水线（本仓不读） | 上游仓内，纳入其 Git |
| `outputs/annotations/<run_id>/{train,val}/approved.json` | `query-foundry/scripts/package_approved.py` | 本仓训练与验证集推理 | 4 重 SHA-256 指纹（协议 12，见下） |
| `outputs/inference/<run_id>/predictions.json` | 本仓 `offline/infer.py` | 融合 `fusion.wbf` | `{query_id: bbox 或 null}` |
| `outputs/ordinal_enum/<run_id>/{parse,instances}.json` | 本仓 `ordinal.enumerate` | `ordinal.resolve` | 每题的解析意图与一次枚举清单（`thinking.jsonl` 存思考文本） |
| `outputs/ordinal_resolve/<run_id>/predictions_<model>.json` | 本仓 `ordinal.resolve` | 融合 `fusion.wbf` | 键与推理产物一致，只改动采纳的序数题 |
| `submission.zip` | 本仓 `submission.py`（显式调用；推理与融合均不自动打包） | 赛事提交 | 官方模板 + `bbox` |

上游以 `--data-root` 指向本仓 `data/`、以自己的 `--index-dir` 指向
`query-foundry/data/indexes/`；两侧不建符号链接。本仓训练只消费 `approved.json`，
不导入上游代码。

## 标注产物 `approved.json`

`{metadata, data}` 两个顶层键，缺一或多一即拒收。样本键形如
`<sequence>_<frame>#<object_index>`（同一帧可有多条描述，序列号取 `_` 之前的前缀）。

### `metadata`

| 字段 | 取值约束 |
| --- | --- |
| `status` | 固定 `"approved"` |
| `protocol_version` | 固定 `12`（`config.ANNOTATION_PROTOCOL_VERSION`） |
| `run_id` | `annot_<tag>`，须与所在目录名及 `--annotation-run-id` 一致 |
| `split` | `"train"` 或 `"val"`，须与所在目录一致 |
| `source_fingerprint` | 64 位十六进制；由三模态路径 + `bbox` + 尺寸复算（忽略 `query`） |
| `preparation_fingerprint` | 64 位十六进制；划分清单 `split_manifest.json` 的内容哈希 |
| `image_fingerprint` | `manifest_` 前缀 + 64 位十六进制；图像引用与记录尺寸的清单指纹（不含图像字节） |
| `dataset_fingerprint` | 64 位十六进制；完整 `data` 字典的内容哈希 |
| `sample_count` | 正整数，须等于 `len(data)` |
| `sequence_count` | 正整数，须等于 `data` 中不同序列数 |
| `prompt_hash` | 非空字符串；上游提示词哈希，进入训练运行身份 |
| `provenance` | 对象，字段见下 |
| `qc` | 对象，固定值见下 |

`provenance`（11 个字段，全部必填）：

| 字段 | 取值约束 |
| --- | --- |
| `source_type` | 固定 `"hosted_open_weights"` |
| `provider`、`annotator_model`、`annotator_revision`、`model_license`、`render_protocol` | 非空字符串 |
| `api_base_url`、`model_weights_url` | 非空字符串且以 `https://` 开头 |
| `mode` | 固定 `"single_marked_frame_generate"` |
| `assignment_policy` | 固定 `"single_marked_rgb_query_generate"` |
| `generation_config` | 非空对象（标注时的生成参数） |

`qc`：必须严格等于
`{complete: true, failed_sequences: 0, failed_frames: 0, invalid_queries: 0, generated_samples: <sample_count>}`。

### `data`

| 字段 | 取值约束 |
| --- | --- |
| `visible`、`infrared`、`depth` | 相对 POSIX 路径字符串；不以 `/` 开头、不含 `.` / `..` 段、三路互不相同 |
| `query` | 非空单行英文描述（1–55 词），且通过脚手架词、坐标样文本、控制符与模型词表检查 |
| `bbox` | 归一化 XYXY，`0 ≤ x1 < x2 ≤ 1`、`0 ≤ y1 < y2 ≤ 1` |
| `width`、`height` | 正整数，为该样本图像的记录尺寸 |

train 与 val 必须来自同一 `run_id`，且两侧 `prompt_hash` 与 `provenance` 一致、
样本 ID 与序列 ID 均不重叠。

## 校验点

| 校验点 | 位置 | 校验内容 | 失败行为 |
| --- | --- | --- | --- |
| 产物准入 | `annotation_state.validate_approved_artifact` | schema、状态与协议、`split`/`run_id`、`provenance`、`qc`、`source`/`dataset` 指纹复算、逐样本 query QC 与 `bbox` 合法性、序列数、结构检查 | 抛 `ValueError`，训练不启动 |
| train/val 配对 | `training_state.validate_training_artifacts` | 两侧 `prompt_hash` 与 `provenance` 一致；样本 ID 与序列 ID 不重叠 | 抛 `ValueError` |
| 图像引用 | `training_core.prepare_training_plan` | 记录值为 `manifest_` 前缀时，复算路径与记录尺寸并与产物比对（不解码图像） | 抛 `ValueError` |
| 官方模板 | `test_data.validate_official_test_template`（由 `submission.build_submission` 调用） | 条数 9555、查询 ID 形态、字段集、路径形态、内容哈希 | 抛 `ValueError`，不出包 |
| Test 深度引用 | `scripts/prepare_rgbdt.validate_test_depth_references` | 官方模板 → `Processed/Test/depth_jet` 逐条映射、深度文件集合指纹、三模态尺寸一致性 | 返回错误列表，脚本中止 |
| 推理续跑 | `inference_state.validate_checkpoint_payload`、`load_resume_predictions` | 分片分配与键集合、框与分数合法性、跨来源冲突结果 | 抛 `ValueError`，拒绝合并 |
| 上游侧 | `query-foundry`：`prepare_split`、`package_approved` | 测试集同帧哈希去重；同帧描述唯一性、QC、双 split 联合校验 | 见上游仓 |

## 提交格式

预测必须是官方模板 JSON 的逐条副本：只替换 `bbox` 字段，`visible`、`infrared`、
`depth`、`query` 逐字节保留；打包为 `submission.zip`，内含单个 `result.json`。
评测按归一化 XYXY 计算 IoU，`IoU ≥ 0.5` 记为命中；反向坐标、越界坐标、NaN 与空框
判为无效预测。`--allow-fallback` 会把无效框填成占位框，用于诊断不完整的包，正式提交
不使用。

## 数据集来源

数据包与官方 Test 的获取方式、许可与引用见根 `README.md` 的「数据来源」一节。
