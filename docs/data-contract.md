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
| `indexes/train.json` / `indexes/val.json` | `scripts/prepare_rgbdt.py`（本仓） | 标注生产线（标注源索引；训练不读） |
| `indexes/split_manifest.json` | `scripts/prepare_rgbdt.py`（本仓） | 标注生产线（预检校验锚） |
| `audits/excluded_overlap.json` | `scripts/filter_overlap.py`（本仓） | 无（纯留痕） |

`prepare_rgbdt.py` 是索引的唯一生成器；不要手改这些 JSON。若需重建索引，
生成器仍在本仓运行，产物输出到 `query-foundry/data/`。

## 迁移说明（2026-09-05）

- 标注生产线分离后，`indexes/` 与 `audits/excluded_overlap.json` 的唯一运行时
  消费者是 query-foundry（训练只消费 `approved.json`，其数据与指纹自包含），
  故产物文件移入 `query-foundry/data/`（git 追踪）；
  图像不搬运，foundry 经 `data/Train`、`data/Processed` 符号链接读本仓。
- golden `approved.json` 的 `source_fingerprint` / `preparation_fingerprint`
  是内容哈希，文件迁移不影响校验（迁移当日 preflight 复算 run-id 逐字节一致）。

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
