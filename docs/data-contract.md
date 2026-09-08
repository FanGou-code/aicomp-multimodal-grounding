# 数据合同（data/ 布局与派生产物）

`data/` 整体 gitignore，通过魔搭公开数据集分发。任何人克隆仓库后按本合同
组装数据，不需要记忆任何约定。

## 目录布局

```text
data/
  Train/<sequence>/            不可变本体：color/ infrared/ depth/ groundtruth.txt
  Test/                        不可变本体：Images/{visible,infrared,depth}/ + queries/queries.json
  Processed/                   派生：Train/<seq>/depth_jet/ + Test/depth_jet/（JET 伪彩深度）
```

**原则**：Train/Test 是不可变本体；`Processed/` 是确定性构建产物。
派生索引与审计日志自 2026-09-05 起随标注生产线移居外部私有仓
`query-foundry/data/`（见下），本仓不再存放。

## 派生产物清单（存放于 query-foundry/data/）

| 文件 | 生成器 | 下游依赖 |
| --- | --- | --- |
| `indexes/train.json` / `indexes/val.json` | `query-foundry/scripts/prepare_split.py` | 标注生产线（标注源索引；训练不读） |
| `indexes/split_manifest.json` | `query-foundry/scripts/prepare_split.py` | 标注生产线（预检校验锚） |
| `indexes/excluded_overlap.json` | `query-foundry/scripts/prepare_split.py` | 查重留痕（防泄漏审计凭据） |

`query-foundry/scripts/prepare_split.py` 整合了 BBox 异常清洗与测试集 SHA-256 查重，是索引与留痕的唯一生成器。产物由 `query-foundry/data/indexes/` 进行 Git 追踪。

## 迁移说明（2026-09-05 至 2026-09-08）

- 标注生产线分离后，`indexes/` 与 `excluded_overlap.json` 完整收口在 query-foundry
  （训练只消费 `approved.json`，其数据与指纹自包含）；
  图像不搬运，foundry 经 `data/Train`、`data/Processed` 读本仓。
- 2026-09-08 闭环：副仓升级 `prepare_split.py` 支持 Test 图像自动探测并生成
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
因此索引文件**移动位置不影响指纹**，但内容必须由生成器产出。
