# 数据合同（data/ 布局与派生产物）

`data/` 整体 gitignore。训练数据包由用户在魔搭分发；官方 Test 由赛事渠道
单独获取，按下列布局放入本地目录。
下载的数据包已包含 `Processed/`，常规部署解压后直接使用，不重复生成。

## 目录布局

```text
data/
  Train/<sequence>/            不可变本体：color/ infrared/ depth/ groundtruth.txt
  Test/                        不可变本体：Images/{visible,infrared,depth}/ + queries/queries.json
  Processed/                   派生：Train/<seq>/depth_jet/ + Test/depth_jet/（JET 伪彩深度）
```

**原则**：Train/Test 是不可变本体；`Processed/` 是确定性构建产物。
派生索引与审计日志自 2026-09-05 起随标注生产线移居伴生仓
`query-foundry/data/`（见下），本仓不再存放。

## 三模态输入格式

| 模态 | 原始格式 | 训练/推理读取方式 |
| --- | --- | --- |
| 可见光 `visible` | 三通道 8 位无符号，取值 `[0, 255]` | 直接按 RGB 读入 |
| 热红外 `infrared` | 三通道 8 位无符号（三个单通道热辐射灰度堆叠，无彩色语义） | 直接按 RGB 读入 |
| 深度 `depth` | 单通道 16 位无符号，单位毫米，0 表示无效深度；相机量程约 300–20000 mm | 经 `Processed/` 的 JET 伪彩图读入（`depth_scaling=fixed`，`min_depth_mm=300`、`max_depth_mm=20000`） |

三模态图像要求已时间同步、空间对齐，尺寸一致；同一场景的三张图按
`visible → infrared → depth` 固定顺序送入模型。`Processed/` 缺失时
`scripts/prepare_rgbdt.py` 可按上述口径重新生成。

## 标注产物协议（approved.json）

训练唯一消费的产物，布局 `outputs/annotations/<run_id>/<split>/approved.json`：

```text
metadata: {status, protocol_version, run_id, split, source_fingerprint,
           preparation_fingerprint, image_fingerprint, dataset_fingerprint,
           sample_count, sequence_count, prompt_hash, provenance, qc}
data:     {sample_id: {visible, infrared, depth, query, bbox, width, height}}
```

- `bbox` 为归一化 XYXY，须满足 `0 ≤ x1 < x2 ≤ 1`、`0 ≤ y1 < y2 ≤ 1`；
  训练前逐样本校验，任一非法即中止。
- `image_fingerprint` 绑定图像路径与记录尺寸（`manifest_` 前缀），训练前重算比对，
  不解码图像字节。
- train/val 必须来自同一 `run_id` 且样本 ID 与序列 ID 均不重叠。

## 提交格式

预测结果必须是官方模板 JSON 的逐条副本，只替换 `bbox` 字段，其余字段（`visible`、
`infrared`、`depth`、`query`）逐字节保留；打包为 `submission.zip`，内含单个
`result.json`。评测按归一化 XYXY 计算 IoU，`IoU ≥ 0.5` 记为命中；反向坐标、越界
坐标、NaN 与空框直接判为无效预测。官方模板 `data/Test/queries/queries.json` 的
条数（9555）与内容哈希被 `tests/test_data.py` 钉死，改为其它形状即报错。

## 派生产物清单（存放于 query-foundry/data/）

| 文件 | 生成器 | 下游依赖 |
| --- | --- | --- |
| `indexes/train.json` / `indexes/val.json` | `query-foundry/scripts/prepare_split.py` | 标注生产线（标注源索引；训练不读） |
| `indexes/split_manifest.json` | `query-foundry/scripts/prepare_split.py` | 标注生产线（预检校验锚） |
| `indexes/excluded_overlap.json` | `query-foundry/scripts/prepare_split.py` | 查重留痕（防测试集泄漏） |

`query-foundry/scripts/prepare_split.py` 整合了 BBox 异常清洗与测试集 SHA-256 查重，是索引与留痕的唯一生成器。产物位于 `query-foundry/data/indexes/`，纳入 Git 追踪。

## 迁移说明（2026-09-05 至 2026-09-08）

- 标注生产线分离后，`indexes/` 与 `excluded_overlap.json` 完整收口在 query-foundry
  （训练只消费 `approved.json`，其数据与指纹自包含）；
  图像仍在主仓，foundry 的 `--data-root` 指向主仓 `data/`，
  `--index-dir` 指向 foundry 的 `data/indexes/`。
- 2026-09-08：副仓升级 `prepare_split.py` 支持 Test 图像自动探测并生成
  `data/indexes/excluded_overlap.json`，清理主仓未追踪临时目录 `data/audits_from_foundry`。
- golden `approved.json` 的 `source_fingerprint` / `preparation_fingerprint`
  是内容哈希，文件迁移不影响校验（preflight 复算 run-id 逐字节一致）。

## 魔搭数据集

- Dataset Repo：`Fang001/rgbdt-grounding-dataset`（打包 `data.tar`）
- 组装命令（首次，见 `docs/sop.md` 第 2 节）：

```bash
modelscope download --dataset Fang001/rgbdt-grounding-dataset data.tar \
  --local_dir /root/rgbdt-download
tar -xf /root/rgbdt-download/data.tar -C data --no-same-owner
rm -rf /root/rgbdt-download
```

## 校验锚

标注生产线预检会比对 `split_manifest.json` 的
`index_fingerprints[split]`（内容哈希）与 `index_sample_counts[split]`；
因此索引文件移动位置不影响指纹，但内容必须由生成器产出。协议 2 的既有两种
JSON 序列化哈希均可读取；生成器继续使用已提交索引的格式，不回写历史文件。

`approved.json` 的 `manifest_` 图像指纹绑定路径和尺寸，不包含图像字节。
训练前会重算该指纹并与审批产物比对（纯内存校验，不解码图像）；历史图像
内容版本由数据交付方另外固定。打包后的同帧描述必须唯一，跨 split 检查通过后才发布。
