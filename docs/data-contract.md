# 数据合同（data/ 布局与派生产物）

`data/` 整体 gitignore，通过魔搭公开数据集分发。任何人克隆仓库后按本合同
组装数据，不需要记忆任何约定。

## 目录布局

```text
data/
  Train/<sequence>/            不可变本体：color/ infrared/ depth/ groundtruth.txt
  Test/                        不可变本体：Images/{visible,infrared,depth}/ + queries/queries.json
  Processed/                   派生：Train/<seq>/depth_jet/ + Test/depth_jet/（JET 伪彩深度）
  indexes/                     派生索引（下方三件，随数据集一起分发）
  audits/                      审计日志（excluded_overlap.json）
```

**原则**：Train/Test 是不可变本体；indexes 与 audits 是确定性构建产物，
与数据同生命周期，随同一个魔搭数据仓分发。

## 派生产物清单

| 文件 | 生成器 | 下游依赖 |
| --- | --- | --- |
| `indexes/train.json` / `indexes/val.json` | `scripts/prepare_rgbdt.py` | 外部标注生产线 query-foundry（标注源索引） |
| `indexes/split_manifest.json` | `scripts/prepare_rgbdt.py` | 外部标注生产线（预检校验锚） |
| `audits/excluded_overlap.json` | `scripts/filter_overlap.py` | 无（纯留痕） |

`prepare_rgbdt.py` 是索引的唯一生成器；不要手改这些 JSON。

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

外部标注生产线（query-foundry）预检会比对 `split_manifest.json` 的
`index_fingerprints[split]`（内容哈希）与 `index_sample_counts[split]`；
因此索引文件**移动位置不影响指纹**，但内容必须由生成器产出。
