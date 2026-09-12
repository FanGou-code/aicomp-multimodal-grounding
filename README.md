# query-foundry — 视觉定位标注工具

从图片 + 文字描述到目标边界框的标注工具。浏览器端看图、画框、改 query，输出归一化 bbox + query 文本，可直接被下游评测脚本消费。

**工具层零 pip 依赖**（纯 Python 标准库），管线层可选依赖 PIL + numpy。

## 30 秒体验

```bash
# 1. 准备你的数据（JSON 格式，向下兼容）
echo '{"img_001": {"image": "photos/a.jpg", "query": "the red car"}}' > my_queries.json

# 2. 生成审查清单
python scripts/make_manifest.py --source my_queries.json --images-root /path/to/images --out review-manifest.json

# 3. 启动审查服务
python scripts/review_server.py --manifest review-manifest.json --data-root /path/to/images --port 8788

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
python scripts/make_manifest.py --source my_queries.json --images-root /path/to/images --out review-manifest.json

# 从 assembly.json 生成（内部管线模式）
python scripts/make_manifest.py --assembly outputs/assembly/asm-train-r5/assembly.json \
    --data-root /path/to/dataset --index-dir data/indexes
```

图片路径为相对路径时，自动拼接 `--images-root`。也支持绝对路径。

### 启动审查服务

```bash
python scripts/review_server.py --manifest review-manifest.json --data-root /path/to/images --port 8788
```

浏览器打开 `http://localhost:8788/`。

## 配置

当前比赛配方固定在代码中：教师和服务地址见 `foundry/utils.py`，四桶分类与配额
见 `foundry/pipeline/buckets.py`。没有通过 JSON 切换教师或桶规则的入口。

实际读取的外置文件为 `configs/default/prompts/` 和 `configs/default/rules/qc.json`；
后者包含冠词规则及人工裁定的 echo 表。变更配方使用新运行标签，不覆盖已有标注。

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
       --split 3 --part 1 --out part1.json
   ```

2. 拿到分片文件（如 `part2.json`），启动审查：
   ```bash
   python scripts/review_server.py --manifest part2.json --data-root /path/to/images --port 8789
   ```

3. 停止写入后，交回完整 `outputs/review/<run_tag>/`：包括 `annotations.jsonl`、
   query/bbox/absent 三份快照。只交两份快照会缺少部分裁决和恢复信息。

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
| `outputs/review/<run_tag>/annotations.jsonl` | 追加日志，apply 优先重放的状态来源 |
| `outputs/review/<run_tag>/annotations.absent.json` | 不存在裁决的快照 |
| `outputs/assembly/<tag>/assembly.json` | apply 烘焙后的最终语料 |
| `outputs/approved/<run_id>/<split>/approved.json` | 打包发布产物（自包含 4 个 SHA-256 指纹，主仓训练直接消费） |

## 环境

- Python 3.12+；审查服务支持 Linux/macOS，Windows 使用 WSL
- 工具层（审查器 + make_manifest + apply + package_approved）：零 pip 依赖，纯标准库
- 管线层（census / assembly）：可选依赖 `pip install Pillow numpy`

## 结构

```
foundry/          — 工具层 review/ (审查器) + 管线层 pipeline/ (普查/组装/QC/合同校验)
scripts/          — CLI 入口 (review_server / make_manifest / apply_review / package_approved 等)
configs/default/  — 实际读取的提示词与 QC 规则
tests/            — 离线逻辑、HTTP 服务及前端状态测试
data/indexes/     — 数据集划分索引
outputs/          — 产物 (census / assembly / review / approved)
```

## 管线用法（内部）

以下为生产线入口。`--data-root` 指图片目录，`--index-dir` 指索引目录；
索引默认取本仓 `data/indexes/`，仍兼容旧的图片根下 `indexes/` 布局。

### 数据准备

```bash
python scripts/prepare_split.py --raw-root /path/to/dataset \
    --seed 42 --train-ratio 0.8 --out-dir data/indexes
```

### Phase 1 普查（API 调用）

```bash
python scripts/run_census.py --split train --limit-sequences 320 \
    --concurrency 48 --num-shards 16 --run-tag census-full-1 \
    --data-root /path/to/dataset --index-dir data/indexes
```

### Phase 2 组装（纯本地）

```bash
python scripts/assemble_queries.py --census-run outputs/census/census_<id> \
    --run-tag asm-<tag> --data-root /path/to/dataset --index-dir data/indexes
```

### 烘焙

```bash
python scripts/apply_review.py --assembly outputs/assembly/asm-train-r5/assembly.json \
    --review-queries outputs/review/asm-train-r5/annotations.queries.json
```

### 打包发布（交付主仓训练）

```bash
python scripts/package_approved.py --assembly outputs/assembly/asm-train-r6/assembly.json \
    --run-id annot_r6 \
    --export-to-main ../aicomp-multimodal-grounding
```

### Key 测活

```bash
python scripts/check_keys.py --data-root /path/to/dataset --index-dir data/indexes
```

## 测试

```bash
python -m unittest discover -s tests
```

完整前端状态测试需要 Node.js；缺少时会明确跳过该项。HTTP 测试仅监听本机临时端口，
API 测试使用假响应，不消耗真实额度。

## 合并与交付

审查日志为恢复依据，快照是可重建导出。多轮合并按传入顺序应用人工修改，后轮
明确裁决覆盖前轮；教师初始框不撤销已有人工结果。旧的仅快照交付仍可读取，
但它不具备 journal 的完整恢复信息。

apply 在最终文本 QC 后重新检测同帧冲突。package 始终拒绝未解决的冲突，
先完成双 split 校验和全部目标检查，再写文件。`--lenient-qc` 仅放宽文本规则，
相应报告随产物导出，不绕过同帧冲突检查。

## License

MIT
