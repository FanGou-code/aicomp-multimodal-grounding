# query-foundry — 视觉定位标注工具

从图片 + 文字描述到目标边界框的标注工具。浏览器端看图、画框、改 query，输出归一化 bbox + query 文本，可直接被下游评测脚本消费。

**工具层零 pip 依赖**（纯 Python 标准库），管线层可选依赖 PIL + numpy。

## 30 秒体验

```bash
# 1. 准备你的数据（JSON 格式，向下兼容）
echo '{"img_001": {"image": "photos/a.jpg", "query": "the red car"}}' > my_queries.json

# 2. 生成审查清单
python scripts/make_manifest.py --source my_queries.json --images-root /path/to/images

# 3. 启动审查服务
python scripts/review_server.py --manifest my_queries.json --data-root /path/to/images --port 8788

# 4. 浏览器打开 http://localhost:8788/，开始标注
```

## 接入你自己的数据

### 数据格式

支持两种 JSON 形状：

```json
// 映射式（推荐）
{"img_001": {"image": "photos/a.jpg", "query": "the red car"}}

// 列表式
[{"id": "img_001", "image": "photos/a.jpg", "query": "the red car"}]
```

可选字段：`bbox`（预标框，`[x1, y1, x2, y2]` 归一化 0-1 XYXY）、`category`（类别名）。

### 生成审查清单

```bash
# 从任意 query JSON 生成（通用模式）
python scripts/make_manifest.py --source my_queries.json --images-root /path/to/images

# 从 assembly.json 生成（内部管线模式）
python scripts/make_manifest.py --assembly outputs/assembly/asm-train-r5/assembly.json \
    --data-root /path/to/dataset
```

图片路径为相对路径时，自动拼接 `--images-root`。也支持绝对路径。

### 启动审查服务

```bash
python scripts/review_server.py --manifest my_queries.json --data-root /path/to/images --port 8788
```

浏览器打开 `http://localhost:8788/`。

## 配置

管线层（census / assembly）的 API 供应商和模型参数通过  配置：



修改  和  即可切换到其他 OpenAI 兼容的 API 供应商。

提示词、桶分类规则、QC 规则分别在  和  下，修改后管线运行时自动加载。

## 审查器快捷键

| 键 | 功能 |
|---|---|
| `Enter` | 核验并跳到下一条待办 |
| `E` | 进入 query 编辑框 |
| `H` / `L` | 前/后翻页 |
| `N` | 跳到未标注条目 |
| `G` / `Shift+G` | 跳到第一条 / 最后一条 |
| `/` | 图号跳转 |
| `P` | 加入待办清单 |
| 滚轮 | 缩放画布 |
| 拖拽 | 平移画布 |

编辑框内：`Ctrl+F/B` 前后移光标、`Ctrl+A/E` 行首/行尾、`Ctrl+K` 删到行尾、`Enter` 保存退出、`Esc` 放弃。

## 多用户协作

1. 管理员生成分片：
   ```bash
   python scripts/make_manifest.py --source all_queries.json --images-root /path/to/images \
       --split 3 --part 1
   ```

2. 拿到分片文件（如 `part2.json`），启动审查：
   ```bash
   python scripts/review_server.py --manifest part2.json --data-root /path/to/images --port 8789
   ```

3. 标完后交回 `outputs/review/<run_tag>/annotations.predictions.json` 和 `annotations.queries.json`

4. 管理员合并各分片：
   ```bash
   python scripts/apply_review.py --assembly assembly.json \
       --review-queries part1/annotations.queries.json part2/annotations.queries.json
   ```

## 输出说明

| 文件 | 内容 |
|---|---|
| `outputs/review/<run_tag>/annotations.predictions.json` | 标注结果：`{item_id: [x1,y1,x2,y2]}` 归一化 0-1 XYXY |
| `outputs/review/<run_tag>/annotations.queries.json` | 人工修订后的 query 文本 |
| `outputs/review/<run_tag>/annotations.jsonl` | 追加日志（崩溃恢复） |
| `outputs/assembly/<tag>/assembly.json` | apply 烘焙后的最终语料 |

## 环境

- Python 3.12+
- 工具层（审查器 + make_manifest + apply）：零 pip 依赖，纯标准库
- 管线层（census / assembly）：可选依赖 `pip install Pillow numpy`

## 结构

```
foundry/          — 工具层 review/ (审查器) + 管线层 pipeline/ (普查/组装/QC)
scripts/          — CLI 入口 (review_server / make_manifest / apply_review 等)
configs/default/  — 默认风格配置 (提示词 / 桶分类规则 / QC 规则)
tests/            — 123 项离线单测
data/indexes/     — 数据集划分索引
outputs/          — 产物 (census / assembly / review)
```

## 管线用法（内部）

以下为完整生产线用法，使用本仓库内部管线生成标注语料。

### 数据准备

```bash
python scripts/prepare_split.py --raw-root /path/to/dataset \
    --seed 42 --train-ratio 0.8 --out-dir data/indexes
```

### Phase 1 普查（API 调用）

```bash
python scripts/run_census.py --split train --limit-sequences 320 \
    --concurrency 48 --num-shards 16 --run-tag census-full-1 \
    --data-root /path/to/dataset
```

### Phase 2 组装（纯本地）

```bash
python scripts/assemble_queries.py --census-run outputs/census/census_<id> \
    --run-tag asm-<tag> --data-root /path/to/dataset
```

### 烘焙

```bash
python scripts/apply_review.py --assembly outputs/assembly/asm-train-r5/assembly.json \
    --review-queries outputs/review/asm-train-r5/annotations.queries.json
```

### Key 测活

```bash
python scripts/check_keys.py --data-root /path/to/dataset
```

## 测试

```bash
python -m unittest discover -s tests
```

## License

MIT
