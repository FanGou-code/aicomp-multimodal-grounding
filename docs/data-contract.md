# 数据契约

本文档定义仓库中数据文件的格式、字段约束与校验规则。
运行时校验由 `aicomp_grounding/contract.py` 执行。

## 1. 文件系统布局

```text
data/
  Raw/<seq>/color/             可见光图像（PNG, 3×8bit RGB）
  Raw/<seq>/infrared/          红外图像（PNG, 3×8bit 灰度堆叠）
  Raw/<seq>/depth/             深度图像（PNG, 1×16bit, 单位 mm, 0 = 无效）
  Raw/<seq>/groundtruth.txt    真值框（逗号分隔: x1,y1,w,h, 像素坐标）
  Test/Images/visible/         测试集可见光
  Test/Images/infrared/        测试集红外
  Test/Images/depth/           测试集深度
  Test/queries/queries.json    官方测试查询模板

outputs/
  annotations/
    train.json                 训练集活动数据集（approved 格式）
    val.json                   验证集活动数据集（approved 格式）
  training/<run_id>/           训练产物（检查点、adapter、状态）
  inference/<run_id>/          推理产物（predictions.json、summary.json）
  fusion/<run_id>/             融合产物
  submission/<tag>/            提交包（submission.zip）
```

## 2. approved.json 格式

训练引擎消费的唯一数据格式。由 `tools/package_approved.py` 生成。

```json
{
  "metadata": { ... },
  "data": {
    "<query_id>": {
      "visible": "Images/visible/000000.jpg",
      "infrared": "Images/infrared/000000.jpg",
      "depth": "Images/depth/000000.png",
      "query": "The person on the left side of the bench.",
      "bbox": [0.123, 0.456, 0.789, 0.901],
      "width": 1920,
      "height": 1080
    }
  }
}
```

### 2.1 metadata 字段

| 字段 | 类型 | 必需 | 说明 |
| --- | --- | --- | --- |
| `status` | string | 是 | 固定值 `"approved"` |
| `protocol_version` | int | 是 | 当前值 `13` |
| `split` | string | 是 | `"train"` 或 `"val"` |
| `sample_count` | int | 是 | `data` 中的条目数，正整数 |
| `sequence_count` | int | 是 | 去重后的序列数，正整数 |
| `provenance` | object | 是 | `{"source_type": "human_annotated"}` |
| `run_id` | string | 否 | 标注运行 ID |
| `run_tag` | string | 否 | 标注运行标签 |
| `source_fingerprint` | string | 否 | 不可变输入的 SHA-256（不含 query） |
| `dataset_fingerprint` | string | 否 | 完整 data dict 的 SHA-256 |
| `image_fingerprint` | string | 否 | 图像绑定指纹（前缀 `manifest_`） |
| `preparation_fingerprint` | string | 否 | split_manifest.json 的 SHA-256 |
| `prompt_hash` | string | 否 | prompt 文本哈希 |
| `qc` | object | 否 | `{"complete": true}` |

### 2.2 data 条目字段

| 字段 | 类型 | 可空 | 约束 |
| --- | --- | --- | --- |
| `visible` | string | 否 | 可见光图像相对路径 |
| `infrared` | string | 否 | 红外图像相对路径 |
| `depth` | string | 否 | 深度图像相对路径 |
| `query` | string | 条件 | 英文目标描述；`allow_pending=true` 时可为空 |
| `bbox` | `[float, float, float, float]` | 条件 | 归一化坐标；`allow_pending=true` 时可为 `null` |
| `width` | int | 否 | 图像物理宽度（像素） |
| `height` | int | 否 | 图像物理高度（像素） |

固定字段集合：每个条目必须且仅包含以上 7 个字段。

## 3. 坐标系契约

- **格式**：`[x1, y1, x2, y2]`，左上角 `(x1, y1)`，右下角 `(x2, y2)`。
- **单位**：归一化到 `[0, 1]`，x 除以图像宽度，y 除以图像高度。
- **不变式**：
  - `0 ≤ x1 < x2 ≤ 1`
  - `0 ≤ y1 < y2 ≤ 1`
  - 面积 `(x2 - x1) × (y2 - y1) > 0`
- **精度**：浮点数，无固定小数位数。
- **无效框判定**：反向坐标（`x1 ≥ x2` 或 `y1 ≥ y2`）、越界（超出 `[0, 1]`）、
  NaN、空框均判定为无效预测（IoU 计 0）。

## 4. 官方测试模板

`data/Test/queries/queries.json` 为官方提供的查询模板。格式：

```json
{
  "<query_id>": {
    "visible": "Images/visible/NNNNNN.jpg",
    "infrared": "Images/infrared/NNNNNN.jpg",
    "depth": "Images/depth/NNNNNN.png",
    "query": "English target description.",
    "bbox": [0.0, 0.0, 0.0, 0.0]
  }
}
```

提交时只修改 `bbox` 字段，其余字段保持不变。`testset.py` 在提交前校验模板完整性。

## 5. 提交包格式

`tools/submission.py` 生成 `submission.zip`，内含单个 JSON 文件。

- 编码：UTF-8
- query_id 集合必须与官方模板完全一致，无遗漏、无多余
- 每个 bbox 必须通过坐标系契约（第 3 节）的不变式校验
- `--allow-fallback` 会将缺失或无效框填为占位框，仅用于诊断

## 6. 指纹体系

`contract.py` 实施 4 重 SHA-256 指纹：

| 指纹 | 输入 | 用途 |
| --- | --- | --- |
| `source_fingerprint` | data dict 中除 query 外的不可变字段 | 检测图像/bbox 篡改 |
| `dataset_fingerprint` | 完整 data dict | 检测任何字段变更 |
| `image_fingerprint` | 图像文件路径绑定 | 绑定图像版本 |
| `preparation_fingerprint` | split_manifest.json | 绑定划分版本 |

所有指纹由 `artifacts.stable_json_hash()` 计算：对 dict 做 key 排序后取
UTF-8 编码的 SHA-256 hex digest。

## 7. query 文本约束

`query.py` 在封包与训练时校验 query 文本：

- 非空字符串
- 仅包含合法英文词汇字符（`validate_generated_query`）
- 不含标注术语（`target`、`object`、`bounding box` 等 `BANNED_WORDS`）
- 不含泛类别词（`validate_annotation_query` 附加规则）

`preflight_check_dataset` 在训练前对整个数据集执行上述检查，
违规条目会触发警告（非阻断）。

## 8. 标注存储格式

标注服务器（`annotator/store.py`）使用 JSONL 日志 + 原子快照：

- **日志**：`journal.jsonl`，每行一条 PUT/DELETE 记录
- **bbox 快照**：`annotations.predictions.json` → `{item_id: [x1, y1, x2, y2]}`
- **无框快照**：`annotations.absent.json` → `{item_id: annotator}`
- **恢复**：从日志重放重建快照；末尾不完整行自动跳过

标注者标识规则：
- `seed`：预标注（待核验）
- `human`：人工确认
- `*:absent`：标记为无框（如 `human:absent`）
