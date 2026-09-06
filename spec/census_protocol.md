# 普查协议 v3（census protocol，2026-09-06 冻结）

> 改协议 = 改本文件 + 升版本号 + 提示词 hash 随之进 run 指纹。
> 原则：任务清晰简单则教师不出低级错误；门只做机械校验，不怀疑教师能力；
> 正确率第一，不造难目标。

## 调用结构（每序列 ≈23 次）

| 步骤 | 调用 | 内容 |
| --- | --- | --- |
| findall ×2 遍 | 每帧 2 次 | 独立枚举，红框图为锚 |
| 双遍交集 | 0（代码） | 1:1 IoU≥0.5 匹配取交集 = 二审 |
| 挑帧 | 0（代码） | 按交集目标数 + 帧间距挑 3 帧 |
| attr 名片 | 每选中帧 1 次 | 每编号对象 color + 一条显著可见特征 |

## findall 提示词 v3(本节文本 = `foundry/census.py` 的 `FINDALL_PROMPT` 逐字副本)

```text
The red rectangle marks one object in the scene.
Task: list up to 6 objects you are MOST CONFIDENT about — clear outline, nameable at a glance — regardless of category, ordered from left to right. If multiple instances of the red-boxed category exist, include them all first, then fill the remaining slots with other confident objects. Skip tiny clutter, blurry ground debris, and anything you cannot identify precisely.
You must include the object inside the red rectangle. Number them 1..N (N is the total count).
For each object give a short common category name and its bounding box as normalized coordinates [x1, y1, x2, y2]: four decimal fractions where 0 is the left/top edge of the image and 1 is the right/bottom edge. NEVER use pixel values.
Output JSON only:
{"objects": [{"i": 1, "category": "<category name>", "bbox": [x1, y1, x2, y2]}, ...]}
```

## attr 提示词(不变;本节文本 = `foundry/census.py` 的 `ATTR_PROMPT` 逐字副本)

```text
The image shows numbered boxes around objects in the scene.
For each numbered object report only what is directly visible: its color and one notable visible feature.
Do not guess occluded or unclear properties.
Output JSON only:
{"1": {"color": "...", "features": "..."}, ...}
```

提示词指纹(`tests/test_census.py` 机器校验文档与代码一致,改任一侧必须同步):
FINDALL `b9f33a9cd0492ccce4a7b1c7f516bdd4ecd070226078a0e3d9f1b552592906a5`、
ATTR `d41666252810fa4b6747fbe291d6fedb8b06b1e84416c2ab5b6e9ea63e313785`。

上限 6 = test 均值 4.78 query/图、众数 5 的现实;「同类先列满」保序数供给。

## 门清单（全部机械校验）

| 门 | 致命性 | 规则 |
| --- | --- | --- |
| 坐标约定识别 | 致命（全部候选都失败才杀帧） | 全 ≤1.02 直用；否则归一化/千分比/像素三候选，金丝雀当裁判 |
| 金丝雀 | 唯一帧级致命门 | 红框目标 IoU≥0.5 被找回 |
| 乱序 | 代码排序重编号 | 不杀帧 |
| 自重复 IoU≥0.95 | 代码去重 | 不杀帧 |
| 零面积框 | 代码丢弃 | 不杀帧 |
| 双遍交集 | 事实裁决 | 交集 = 可信对象集，多标少标不重跑 |
| 序列门 | 选中帧裁决 | 3 个选中帧全部完成 = 序列完成；未选中帧失败只废弃该帧不连坐序列（管理员 2026-09-06） |

## 深度事实生成规则（R1 实现，服务距离/前景桶）

- 输入：**原始 uint16 毫米深度图** `data/Train/<seq>/depth/<frame>.png`
  （由索引 visible 路径的 `color` 段换为 `depth` 段推导，train/val 同构）。
  不解码 depth_jet 伪彩图——伪彩存在逐帧归一化跨帧不可比、无效像素涂黑与
  jet 蓝端混淆两个坑；原始毫米值无此问题，无解码风险即无降级阶梯。
- 计算：本地进程内完成（PIL 读 uint16 + numpy 中值），对象 bbox 内有效像素
  （>0）中值 = 该对象深度；毫米语义精确：值小 = 近。
- 事实三档（存储进 census merged.json 的 `frame["depth"]`，**零图片落仓**）：
  - `closest to the camera` / `farthest from the camera`：帧内毫米余量
    ≥200mm 的极值（唯一性天然成立）；
  - `in the foreground` / `in the background`：帧有效深度跨度的近端/远端
    三分带（非独占，唯一性门要求同头名词目标在带内唯一）；
  - 不做细粒度比较（"比 X 更近"），中景不做（test 仅 4 条）。
- 深度记录缺失的帧（文件不存在等）：组装器退回底边 y2 几何代理，run 内
  通过每帧状态行 `depth=-` 可见。
- 用途定位：消歧工具优先（多同类别目标前后景混排时锁定目标），配额其次；
  训练输入层不变，RGB+T+D 照旧喂图。

## 并发与预算（管理员定案）

- 12 key = 12 账号，官方每账号并发上限 8；运行并发 48（4/账号，上限一半），
  key 池逐请求轮转分发。
- 第四批 train（320 序列）≈7,400 调用 / 预计 1.5-3 小时；val（80 序列）
  ≈1,900 调用 / ≈400 万 token / ≈2-3 小时，串行接力。
- 质量参照（试点 census_7108aac8c25365a9）：完成率 98.3%、jaccard 0.911、
  attr 95%、金丝雀失败 1.7%。
