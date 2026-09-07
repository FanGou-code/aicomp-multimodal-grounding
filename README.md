# query-foundry — 视觉定位标注生产线

从图片数据集到自然语言定位 query 的完整标注工具链。包含审查器（浏览器端人工画框+改句）和组装器（从普查事实自动拼装 query），输出归一化 bbox + query 文本。

## 快速开始（审查器）

审查器是主要对外接口：浏览器里看图、画框、改 query，产出标注文件。

```bash
# 1. 生成审查清单
python scripts/make_manifest.py --assembly outputs/assembly/asm-train-r5/assembly.json \
    --data-root /path/to/dataset

# 2. 启动审查服务
python scripts/review_server.py --manifest asm-train-r5.json \
    --data-root /path/to/dataset --port 8788

# 3. 浏览器打开 http://localhost:8788/
```

快捷键：`Enter` 核验跳下一条、`E` 编辑 query、`H/L` 翻页、`N` 跳未标注、`G/Shift+G` 首尾、`/` 图号跳转、滚轮缩放。编辑框内 `Ctrl+F/B/A/E/K` 为光标快捷键。

## 多用户协作

1. 管理员用 `make_manifest.py --split N --part I` 生成分片文件
2. 队友 clone 仓库，将分片文件放入仓库目录，准备图片数据
3. 队友启动审查服务：
   ```bash
   python scripts/review_server.py --manifest part2.json --data-root /path/to/dataset --port 8788
   ```
4. 完成后交回 `outputs/review/<run_tag>/annotations.predictions.json` 和 `annotations.queries.json`
5. 管理员合并各分片产出，运行 `apply_review.py` 烘焙最终语料

## 环境

Python 3.12。测试：

```bash
python -m unittest discover -s tests
```

## 结构

```
foundry/       工具层（review/）+ 管线层（pipeline/）
scripts/       CLI 入口
configs/       风格配置（prompts + rules）
tests/         离线单测
data/indexes/  数据集划分索引
outputs/       产物（census / assembly / review）
```

## 管线用法（内部）

### Phase 1 普查

```bash
python scripts/run_census.py --split train --limit-sequences 20 \
    --concurrency 48 --num-shards 16 --run-tag census-smoke --preflight-only \
    --data-root /path/to/dataset
```

### Phase 2 组装

```bash
python scripts/assemble_queries.py --census-run outputs/census/census_<id> \
    --run-tag asm-<tag> --data-root /path/to/dataset
```

### 数据准备

```bash
python scripts/prepare_split.py --raw-root /path/to/dataset \
    --test-hashes data/test_image_hashes.json --out-dir data/indexes
```

### Apply 烘焙

```bash
python scripts/apply_review.py --assembly outputs/assembly/asm-train-r5/assembly.json \
    --review-queries outputs/review/asm-train-r5/annotations.queries.json
```

### Key 测活

```bash
python scripts/check_keys.py --data-root /path/to/dataset
```

## 许可

MIT
