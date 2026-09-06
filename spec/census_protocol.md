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

## findall 提示词 v3（变更点加粗）

The red rectangle marks one object in the scene.
Task: list **up to 6** objects, **only those you are most confident about** —
clear outline, nameable at a glance. **If multiple instances of the red-boxed
category exist, include them all first, then fill the remaining slots with
other confident objects. Skip tiny clutter, blurry ground debris, and anything
you cannot identify precisely.**
You must include the object inside the red rectangle. Order from left to right.
Number them 1..N (N is the total count).
For each object give a short common category name and its bounding box as
normalized coordinates [x1, y1, x2, y2]: four decimal fractions where 0 is the
left/top edge of the image and 1 is the right/bottom edge. NEVER use pixel values.
Output JSON only:
`{"objects": [{"i": 1, "category": "<category name>", "bbox": [x1, y1, x2, y2]}, ...]}`

上限 6 = test 均值 4.78 query/图、众数 5 的现实；「同类先列满」保序数供给。

## attr 提示词（不变）

The image shows numbered boxes around objects in the scene.
For each numbered object report only what is directly visible: its color and
one notable visible feature. Do not guess occluded or unclear properties.
Output JSON only: `{"<i>": {"color": "...", "features": "..."}}`

## 门清单（全部机械校验）

| 门 | 致命性 | 规则 |
| --- | --- | --- |
| 坐标约定识别 | 致命（全部候选都失败才杀帧） | 全 ≤1.02 直用；否则归一化/千分比/像素三候选，金丝雀当裁判 |
| 金丝雀 | 唯一帧级致命门 | 红框目标 IoU≥0.5 被找回 |
| 乱序 | 代码排序重编号 | 不杀帧 |
| 自重复 IoU≥0.95 | 代码去重 | 不杀帧 |
| 零面积框 | 代码丢弃 | 不杀帧 |
| 双遍交集 | 事实裁决 | 交集 = 可信对象集，多标少标不重跑 |

## 深度事实生成规则（R1 实现，服务距离/前景桶）

- 输入：`data/Processed/<split>/<seq>/depth_jet/*.png`（本地解码，零 API）。
- 前置验证：jet 色表解码后单调性 sanity check（近景目标深值 < 远景目标）；
  验不过 → 降级为底边 y2 几何代理并在 run report 标注降级。
- 事实三档（带余量，不贴边）：
  - `in the foreground` / `in the background`：帧内深度排序近端/远端带；
  - `closest to the camera` / `farthest from the camera`：深度极值（唯一性天然成立）；
  - 不做细粒度比较（"比 X 更近"），中景不做（test 仅 4 条）。
- 用途定位：消歧工具优先（多同类别目标前后景混排时锁定目标），配额其次。

## 并发与预算（管理员定案）

- 12 key = 12 账号，官方每账号并发上限 8；运行并发 48（4/账号，上限一半），
  key 池逐请求轮转分发。
- 第四批 train（320 序列）≈7,400 调用 / 预计 1.5-3 小时；val（80 序列）
  ≈1,900 调用 / ≈400 万 token / ≈2-3 小时，串行接力。
- 质量参照（试点 census_7108aac8c25365a9）：完成率 98.3%、jaccard 0.911、
  attr 95%、金丝雀失败 1.7%。
