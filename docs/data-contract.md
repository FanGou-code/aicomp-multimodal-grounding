# 数据合同（data/ 布局与派生产物）

`data/` 整体 gitignore。训练数据包由用户在魔搭分发；官方 Test 由赛事渠道
单独获取，按下列布局放入本地目录。

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
训练的 `--deep-verify-images` 会检查图片文件与尺寸并计算字节哈希；历史图像
内容版本由数据交付方另外固定。打包后的同帧描述必须唯一，跨 split 检查通过后才发布。
