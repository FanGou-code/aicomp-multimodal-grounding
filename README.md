# query-foundry

面向多模态目标视觉定位（RGB-D-T Grounding）的数据工程与人机协同质检系统。

输入主仓预处理产出的三模态对齐图像（可见光 RGB、热红外 Infrared、16 位毫米深度
Depth-Jet），输出可直接驱动下游模型微调的自包含标注产物 `approved.json`：
每帧的英文描述（query）+ 归一化边界框 + 三模态路径 + 4 重 SHA-256 指纹。

本仓是 [aicomp-multimodal-grounding](https://github.com/FanGou-code/aicomp-multimodal-grounding)
（RGBDT 视觉定位训练与推理系统）的上游数据工程仓，两仓通过 `approved.json`
单向衔接。

```
主仓 scripts/prepare_rgbdt.py                本仓（数据工程）                      主仓（训练）
├─ 三模态存在性/尺寸对齐校验            ┌──────────────────────────┐
├─ 16 位深度 → JET 伪彩（Processed/）   │ prepare_split → census    │
└─ groundtruth.txt → 归一化 XYXY   ───▶ │ → assembly → text_qc      │ ───▶ outputs/annotations/
                                       │ → review → apply          │      <run_id>/{train,val}/
                                       │ → package_approved        │      approved.json
                                       └──────────────────────────┘            │
                                                                               ▼
                                                                    LoRA 微调 → 推理 → ACC@0.5
```

## 系统定位与痛点

视觉定位的监督数据有三个属性会直接决定最终指标的含金量：**不与评测集重叠**、
**不跨时序泄漏**、**描述指向唯一目标**。传统标注流程在这三点上失守的方式都很具体：

| 痛点 | 失效形态 | 本仓的应对 |
| --- | --- | --- |
| 训练/测试集重叠污染 | 训练帧与评测帧是同源画面，模型"见过答案"，离线指标虚高、线上崩 | 逐帧 SHA-256 与测试集全量比对，命中即剔除并留痕（`excluded_overlap.json`） |
| 视频帧时序穿越 | 同一镜头的相邻帧分别落进 train 与 val，验证集等于"背题" | 按镜头序列整段分配 train/val，序列不跨 split；分配清单与指纹冻结在 `split_manifest.json` |
| 自然语言指向歧义 | 同帧有多个同类目标（"the car"），描述无法唯一确定监督目标 | 组装期确定性唯一性门：描述必须在该帧内只指向一个被枚举物体，否则不产出 |
| 教师模型自证与幻觉 | 让大模型"直接写标注"，错误无法被机械校验 | 教师只报事实（枚举 + 属性），所有门控在代码里，教师不写 query、不自我认证 |
| 人审不可恢复 / 多人合并丢数据 | 崩溃丢进度、快照覆盖、多人分片合并时后写覆盖前写 | 追加日志（WAL）+ 进程锁 + 崩溃恢复重放；合并按传入顺序重放，后轮裁决覆盖前轮 |
| 交付产物与训练端口径漂移 | 打包后才发现 schema、坐标或重复描述不合规 | 打包前全量契约自校验（协议 12）+ 4 重指纹，校验失败不产出交付文件 |

## 核心技术特性（六项）

### 1. 跨集哈希去重与防污染（Anti-Leakage Deduplication）

`scripts/prepare_split.py` 在划分前先做去重：对 `visible`（可见光）逐帧计算
SHA-256，与测试集图像哈希集合比对，命中即从 train/val 候选中剔除，并把
`{sample_id, visible, test_images}` 明细写入 `data/indexes/excluded_overlap.json`
（含 `test_images_hashed` 计数）。测试集哈希可直接给出，或由脚本自动探测
`Test/Images/visible`、`Test/visible`、`Test/color` 生成。

当前冻结索引的实测留痕：比对 1,999 张测试图，剔除 train 246 帧、val 63 帧。

### 2. 镜头序列级防泄漏划分（Sequence-level Split）

划分单位是**镜头序列**而不是帧：`_build_split` 对全部序列做种子洗牌
（默认 `seed=42`），按 `train_ratio` 在序列数上切分，同一序列的帧永远落在同一侧。
`split_manifest.json` 记录 `train_sequences` / `val_sequences` 清单、
`index_fingerprints`（内容哈希）、`index_sample_counts` 与
`split_method="frozen-sequence-assignment"`，供下游预检复算。

当前冻结划分：400 序列 → train 320 / val 80 序列，样本数 2,875 / 719。

### 3. 三模态物理与空间事实普查（Tri-modal Census）

普查阶段（`scripts/run_census.py` + `foundry/pipeline/census.py`）对每个选中帧跑
三遍：两遍独立的 `findall` 枚举（红框参考视图，最多 12 个最可信目标，多实例同类
目标优先、从左到右排序）+ 一遍 `attr` 属性报告（编号框视图，报告颜色与一个可见
特征）。教师**只报事实**，其余全部由代码判定：

- **canary 门（唯一帧级致命门）**：返回集合必须重新找到真值框（IoU ≥ 0.5）——
  教师指错目标无法修复，只有这一种情况判帧失败。
- **非致命归一化**：到达顺序按 x1 重排并重新编号；近重合框（IoU ≥ 0.95）判为同物
  重复列举而剔除；零面积框直接丢弃。
- **双遍一致性**：两遍枚举做一对一 IoU ≥ 0.5 匹配，交集才是"可信物体集"
  （第二遍本身就是复核）。
- **坐标约定自动判别**：0–1 归一化 / 0–1000 per-mille / 所示图像像素三种约定同时
  尝试，以 canary 为唯一判据选出生效的那个。
- **深度事实**（`foundry/pipeline/depth.py`）：直接读取原始 uint16 毫米深度
  （`0` = 无效读数），按 bbox 内有效像素取**中位数**，得到最近/最远（200 mm 余量）
  与前景/后景（帧内有效深度按三分之一分带）判据——毫米语义精确，不经伪彩解码，
  不做跨帧归一化。

热红外在事实层不参与文本生成，它与 RGB、深度共同作为对齐模态随产物交付，在训练
侧作为三模态输入之一进入模型。

### 4. 无歧义指向性文本质检（Text QC & Unambiguity Engine）

两道机械门控保证"描述只指向一个目标"：

- **组装期唯一性门**（`foundry/pipeline/assembly.py: realization_is_unique`）：
  每个候选句都必须在该帧内被代码验证为唯一可解析——序数需在同类目标中有确定的
  秩位与间距、极值需对同组其他目标保持严格余量、锚点参照物需自身唯一、属性描述
  在有未知属性的目标存在时不成立（缺失属性不算作"不同"，绝不因信息缺失而放行）。
- **文本 QC**（`foundry/pipeline/text_qc.py`）：冠词引擎为
  `with/wearing/holding/carrying` 尾句补冠词（按元音字母取 a/an），命中人工裁定的
  KEEP / 复数-物质词 / 例外表则保持不变；echo 表按 `(item_id, before)` 键控，
  只有当该条目当前文本仍等于记录的 before 时才生效，因此重放幂等、不会过度套用。

此外还有脚手架词表（`red rectangle`、`annotated image`、`this frame`、
`in the image`…）、坐标样文本、模型控制符与装饰格式的拦截；人群消歧由深度带词
（`in the foreground` / `in the background`）承担，配额规划优先把这类句子分配给
同类目标 ≥ 3 的帧。

### 5. 崩溃安全的人机协同审查（Crash-safe Review Server）

审查器（`foundry/review/`）是**纯标准库实现**（`http.server` + `json` + `fcntl`），
零 pip 依赖；前端为 `foundry/review/web/` 下的原生 HTML/JS/CSS，由同一进程提供。

- **WAL 追加日志**：每次写入都是一行 JSONL（`annotations.jsonl`），快照
  （`annotations.predictions.json` / `.queries.json` / `.absent.json`）在每次变更后
  由一次完整日志重放重建并原子替换——崩溃后恢复以日志为准，撕裂的尾行直接忽略。
- **进程锁 + 线程锁**：`flock` 协调多个服务进程/实例，Python 锁保护同实例线程；
  快照始终反映同一份日志状态，多写者不会互相吞掉记录。
- **跨进程热重载**：日志文件签名（inode/mtime/size）变化即自动重放，其他进程写入的
  标注立即对读者可见。
- **前端防竞态**：保存响应只更新发起请求的条目，跳转后到达的旧响应不覆盖当前画布。
- **三种会话模式**：`--census-run`（普查框审，教师框预置为 AI 预标注）、
  `--assembly`（组装件审：改框 + 改 query）、`--manifest`（任意清单，通用模式）。
- **人工操作**：拖拽/缩放修正框（`PUT /api/item/<id>/bbox`）、在编辑框改写 query
  （`PUT /api/item/<id>/query`）。前端不提供"目标不存在"按钮，HTTP 也不提供删除入口：
  缺席裁决以 `bbox: null` 且 `annotator` 带 `:absent` 后缀写入日志，只有经
  `AnnotationStore.delete()` 的程序化调用（脚本/工具）才能产生，标注侧与
  `apply_review.py` 都按该记录形态读取。

### 6. 指纹化自包含契约交付（Contract-Driven Packaging）

`scripts/package_approved.py` 把组装产物封包为下游可直接消费的 `approved.json`，
自带 4 重 SHA-256 指纹并逐项自校验：

| 指纹 | 含义 |
| --- | --- |
| `source_fingerprint` | 不可变输入（三模态路径 + 框 + 尺寸）的内容哈希，忽略 query 文本 |
| `dataset_fingerprint` | 完整数据集字典的内容哈希 |
| `image_fingerprint` | 图像引用与记录尺寸的清单指纹（`manifest_` 前缀） |
| `preparation_fingerprint` | `split_manifest.json` 的内容哈希（划分与准备口径） |

打包前依次执行：schema 与 QC 全量校验（协议 12，`--lenient-qc` 仅放宽文本规则且
随产物导出报告）、同帧 query 碰撞门（同帧描述必须唯一）、train/val 联合校验
（跨集样本 ID 与序列不重叠、prompt 一致）。所有目标路径先整体检查再逐文件原子
写入，默认不覆盖既有文件；校验失败时不产生任何交付文件。

## 数据生产流水线

```text
[主仓] prepare_rgbdt.py ──▶ data/{Train,Test,Processed}（三模态校验 + Depth-JET 伪彩）
        │
        ▼
[1] prepare_split.py ──▶ data/indexes/{train,val}.json + split_manifest.json + excluded_overlap.json
        │                 测试集同帧哈希去重 → 序列级 train/val 冻结划分
        ▼
[2] run_census.py ─────▶ outputs/census/census_<id>/（merged.json + 分片 checkpoint）
        │                 红框/编号视图 → 双 findall + attr → 代码门控 → 深度事实
        ▼
[3] assemble_queries.py ▶ outputs/assembly/asm-<tag>/assembly.json（+ text QC 日志）
        │                 事实层 → 句族实现 → 唯一性门 → 四桶配额规划
        ▼
[4] review_server.py ──▶ outputs/review/<tag>/（annotations.jsonl + 三份快照）
        │                 人审：修正框 / 修订 query / 不存在裁决
        ▼
[5] apply_review.py ───▶ outputs/assembly/asm-…-r6/assembly.json（合并烘焙）
        │
        ▼
[6] package_approved.py ▶ outputs/approved/<run_id>/<split>/approved.json
                          （4 重指纹 + 契约自校验；--export-to-main 直交主仓）
```

`--data-root` 指图片目录，`--index-dir` 指索引目录；索引默认取本仓 `data/indexes/`，
仍兼容旧的图片根下 `indexes/` 布局。

```bash
# [1] 划分与去重（seed / 比例可调，默认 42 / 0.8）
python scripts/prepare_split.py --raw-root /path/to/dataset \
    --seed 42 --train-ratio 0.8 --out-dir data/indexes

# [2] 普查（API 调用；--resume 断点续跑、--retry-failed 重试失败项、
#     --preflight-only 只做计划与校验、--deep-verify-images 额外校验图像字节）
python scripts/run_census.py --split train --limit-sequences 320 \
    --concurrency 48 --num-shards 16 --run-tag census-full-1 \
    --data-root /path/to/dataset --index-dir data/indexes

# [3] 组装（纯本地，确定性）
python scripts/assemble_queries.py --census-run outputs/census/census_<id> \
    --run-tag asm-<tag> --data-root /path/to/dataset --index-dir data/indexes

# [4] 人审（见下节：最小会话 / 多人分片）

# [5] 合并烘焙
python scripts/apply_review.py --assembly outputs/assembly/asm-train-r5/assembly.json \
    --review-queries outputs/review/asm-train-r5/annotations.queries.json

# [6] 打包发布（自动算 4 重指纹并直交主仓）
python scripts/package_approved.py \
    --assembly outputs/assembly/asm-train-r6/assembly.json \
             outputs/assembly/asm-val-r6/assembly.json \
    --run-id annot_r6 \
    --export-to-main ../aicomp-multimodal-grounding

# 辅助入口
python scripts/check_keys.py --data-root /path/to/dataset --index-dir data/indexes
python scripts/review_report.py --census-run outputs/census/census_<id>
```

真实 API 调用不属于单元测试范围。

## 与下游训练系统的生态联动

下游是 [aicomp-multimodal-grounding](https://github.com/FanGou-code/aicomp-multimodal-grounding)：
RGB-D-T 三模态视觉定位训练与推理系统（Qwen3-VL-8B / Qwen3.5-9B / MiMo-VL-7B /
GLM-4.6V-Flash + Grounding-DINO 基线，LoRA 微调，唯一指标 ACC@0.5）。

交付一步到位：`--export-to-main` 会把 train 与 val 两个 split 同时写入主仓的
`outputs/annotations/<run_id>/{train,val}/approved.json`，主仓随后可直接启动训练：

```bash
# 本仓：封包并交付
python scripts/package_approved.py \
    --assembly outputs/assembly/asm-train-r6/assembly.json \
             outputs/assembly/asm-val-r6/assembly.json \
    --run-id annot_r6 --export-to-main ../aicomp-multimodal-grounding

# 主仓：直接训练（run_id 即上一步的 --run-id）
python offline/train.py --annotation-run-id annot_r6 \
    --model qwen3vl --model-path /path/to/Qwen3-VL-8B-Instruct \
    --data-dir data --batch-size 1 --gradient-accumulation-steps 16 \
    --learning-rate 1e-4 --epochs 3 --run-tag qwen3vl-r1
```

**两仓的合同是同一条**：产物协议 `protocol_version=12`，字段
`visible`/`infrared`/`depth`/`query`/`bbox`/`width`/`height`，坐标归一化 XYXY。
本仓 `foundry/pipeline/contract.py` 与主仓 `aicomp_grounding/annotation_state.py`
各自独立校验同一份 schema 与指纹，任一侧不通过都不进入训练——交付物一旦封包，
两侧的校验是双端成立的。

## 快速开始

### 环境

```bash
pip install -e ".[pipeline]"   # 管线层（census / assembly）需要 Pillow + numpy
```

- Python 3.12+；审查服务支持 Linux/macOS，Windows 使用 WSL。
- **工具层零依赖**：审查器、`make_manifest`、`apply_review`、`package_approved`
  纯标准库；只有普查/组装（读取图像与深度）需要 Pillow + numpy。
- 完整前端状态测试需要 Node.js，缺失时该测试项明确跳过。

### 最小审查会话（三步）

```bash
# 1. 生成审查清单（从 assembly.json）
python scripts/make_manifest.py \
    --assembly outputs/assembly/asm-train-r5/assembly.json \
    --data-root /path/to/dataset --index-dir data/indexes

# 或从任意 query JSON 生成（通用模式，零管线依赖）
# 支持映射式 {"img_001": {"image": "photos/a.jpg", "query": "the red car"}}
# 与列表式 [{"id": "img_001", "image": "photos/a.jpg", "query": "the red car"}]
python scripts/make_manifest.py --source my_queries.json \
    --images-root /path/to/images --out review-manifest.json

# 2. 启动审查服务
python scripts/review_server.py --manifest review-manifest.json \
    --data-root /path/to/images --port 8788

# 3. 浏览器打开 http://localhost:8788/
```

### 多人协作分片

```bash
# 管理员切分清单
python scripts/make_manifest.py --source all_queries.json --images-root /path/to/images \
    --split 3 --part 1 --out part1.json

# 每位审查者启动自己的分片与端口
python scripts/review_server.py --manifest part2.json --data-root /path/to/images --port 8789

# 停止写入后交回完整 outputs/review/<run_tag>/ 目录
# （annotations.jsonl + query/bbox/absent 三份快照；只交两份快照会缺少部分裁决与恢复信息）

# 管理员按传入顺序合并各分片
python scripts/apply_review.py --assembly assembly.json \
    --review-queries part1/annotations.queries.json part2/annotations.queries.json
```

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

编辑框内：`Ctrl+F/B` 前后移光标、`Ctrl+A/E` 行首/行尾、`Ctrl+K` 删到行尾、
`Enter` 保存退出、`Esc` 放弃。

HTTP 接口：`GET /api/session`、`GET /api/progress`、`GET /image`、
`PUT /api/item/<id>/bbox`、`PUT /api/item/<id>/query`，其余路径回落到
`foundry/review/web/` 静态前端。

## 产物说明

| 文件 | 内容 |
|---|---|
| `outputs/census/census_<id>/` | 普查结果 `merged.json`、运行计划与分片 checkpoint |
| `outputs/assembly/<tag>/assembly.json` | 组装后的语料（每条含 sample_id、object_index、query、bbox、family、bucket、facts） |
| `outputs/review/<run_tag>/annotations.jsonl` | 追加日志，apply 优先重放的状态来源 |
| `outputs/review/<run_tag>/annotations.predictions.json` | 框结果：`{item_id: [x1,y1,x2,y2]}` 归一化 0–1 XYXY |
| `outputs/review/<run_tag>/annotations.queries.json` | 人工修订后的 query 文本 |
| `outputs/review/<run_tag>/annotations.absent.json` | 不存在裁决快照：`{item_id: annotator}` |
| `data/indexes/{train,val}.json` | 划分索引：`sample_id → {visible, infrared, depth, bbox, width, height}`（无 query） |
| `data/indexes/split_manifest.json` | 划分清单：序列清单 + 索引指纹 + 样本数 |
| `data/indexes/excluded_overlap.json` | 跨集去重留痕：命中测试集的帧明细 |
| `outputs/approved/<run_id>/<split>/approved.json` | 打包发布产物（4 重 SHA-256 指纹，主仓训练直接消费） |

## 预置生产配方与规则

教师身份、服务地址与限流参数固化在 `foundry/utils.py`（教师为托管开源权重模型
GLM-4.6V，经 OpenAI 协议端点调用，account 级并发受服务方限制，可用
`--requests-per-minute` / `--tokens-per-minute` 对齐账号配额）；四桶分类与配额份额
固化在 `foundry/pipeline/buckets.py`：分类优先序为 `ordinal` → `distance` → `spatial`
→ `attribute_action`，配额份额（per-mille）分别为 `ordinal` 335 / `spatial` 258 /
`attribute_action` 256 / `distance` 151（即份额最大值在 `spatial`，`distance` 最小）。
这两处没有通过 JSON 切换教师或桶规则的入口。

实际读取的外置文件只有 `configs/default/prompts/`（findall / attr 提示词）与
`configs/default/rules/qc.json`（冠词规则、KEEP 表与人工裁定的 echo 表）。
提示词按设计不含示例 query、不含风格选项——教师没有可模仿的风格样本。
变更配方使用新的运行标签，不覆盖已有标注。

密钥只走环境变量或 `keys/`（已 ignore），仓库内不含任何凭据。

## 合并与交付

审查日志是恢复依据，快照是可重建的导出。多轮合并按传入顺序应用人工修改，后轮明确
裁决覆盖前轮；教师初始框不撤销已有人工结果。旧的仅快照交付仍可读取，但不具备
journal 的完整恢复信息。

`apply_review.py` 在最终文本 QC 后重新检测同帧冲突；`package_approved.py` 始终拒绝
未解决的冲突，先完成双 split 校验与全部目标检查，再逐文件原子写入。
`--lenient-qc` 仅放宽文本规则，相应报告随产物导出，不绕过同帧冲突检查。

## 目录结构

```
foundry/
  review/           审查器（工具层，零 pip 依赖）
    server.py         HTTP 服务与三种会话模式
    store.py          追加日志 + 快照 + flock 崩溃安全存储
    census_session.py census / assembly 会话构建
    bbox.py           审查器坐标工具
    web/              原生 HTML/JS/CSS 前端
  pipeline/         数据管线层（Pillow + numpy）
    api.py            OpenAI 协议客户端 + Key 池 + 限流 + 有界重试
    census.py         普查协议：提示词、响应解析、确定性门
    facts.py          帧级事实提取（ObjectFacts / Realization）
    planner.py        四桶配额分配 + 句族多样性
    assembly.py       句族实现 + 唯一性门 + 目标选择
    buckets.py        冻结四桶分类与份额
    depth.py          16 位毫米深度事实
    text_qc.py        冠词引擎 + echo 表裁定
    contract.py       下游训练合同校验与指纹（协议 12）
    views.py          标注视图渲染 + 图像指纹
    sharding.py       镜头感知的选择与分片
    source.py         标注源索引加载与校验
  bbox.py / utils.py  共享层：坐标与 IO/指纹（零 pip 依赖）
scripts/              CLI 入口（prepare_split / run_census / assemble_queries /
                      review_server / make_manifest / apply_review /
                      package_approved / check_keys / review_report）
configs/default/      实际读取的提示词与 QC 规则
data/indexes/         划分索引与清单（纳入 Git 追踪）
tests/                离线逻辑、HTTP 服务与前端状态测试
outputs/              产物（census / assembly / review / approved，已 ignore）
```

## 测试

```bash
python -m unittest discover -s tests
```

156 项测试，覆盖划分与去重、普查门控、事实与规划、组装唯一性、文本 QC、契约打包、
审查器 HTTP 与存储恢复、Key 池。HTTP 测试只监听本机临时端口，API 测试使用假响应，
不消耗真实额度。

## License

MIT，见 `LICENSE`。
