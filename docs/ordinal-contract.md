# 序数后处理契约（服务侧 ↔ 标注侧共用）

两个仓库共用同一条内核与同一套中间表示的语义，入口与出口相反。共用的不是报文格式：
两仓面向不同的模型、不同的提示词，各自的 JSON 信封独立，互不解析对方的产物。

## 1. 内核

```
解析 → 枚举同类 → 代码按轴排序 → 取第 k 个
```

## 2. 中间表示

```
{category, k, axis, direction, count, color, feature}
```

| 字段 | 含义 | 逆向（服务侧） | 正向（标注侧） |
|---|---|---|---|
| `category` | 目标类别 | parse 解析出 | 枚举时模型认出 |
| `k` | 第几个（1-based） | parse 解析出 | 代码排序算出 |
| `axis` | `x` / `y` / `depth` / `ir` / `area` | parse 解析出 | 代码选定维度 |
| `direction` | `asc` / `desc` | parse 解析出 | 代码选定方向 |
| `count` | 该类实例总数 | 枚举自报 | 枚举自报 |
| `color` / `feature` | 描述性属性 | — | 组句时模型认 |

轴名、`k` 的表示与方向定义，两仓逐字一致。

### 2.1 排序键与平局

```
x      左边缘 x1          asc = 左起第一个，desc = 右起第一个
y      顶边 y1            asc = 上起第一个，desc = 下起第一个
depth  框内 16 位毫米中位数  asc = 最近，desc = 最远
ir     框内红外均值         asc = 最冷，desc = 最热
area   (x2-x1)(y2-y1)     asc = 最小，desc = 最大
```

`x` / `y` 用边缘而不是中心点，正向侧的 `leftmost` / `rightmost` / `topmost` /
`bottommost` 与极值维度用同一把键，因此与推理侧 `asc k=1` / `desc k=1` 取到同一个框。

平局一律按 `(x1, y1)` 升序，与方向无关：`desc` 只反转主键，不反转平局。两仓的排序
实现（`rank_instances` 与 `pick_kth`）都必须满足这一条，否则同一场景两仓会选出不同的框。

## 3. 参数

### 3.1 主仓 · qwen3.5（本地开源）

| 环节 | thinking | temperature | top_p | top_k | presence_penalty | max_new_tokens |
|---|---|---|---|---|---|---|
| 主推理（吐坐标） | 关 | 贪心 | — | — | — | 32 |
| parse（纯文本 → JSON） | 关 | 贪心 | — | — | — | 256 |
| 枚举（看图数清单） | 开 | 0.6 | 0.95 | 20 | 0.0 | 8192 |

依据：Qwen3.5-9B 官方 model card，HF 与 ModelScope 内容一致。

```
思考-通用        1.0 / 0.95 / 20 / 1.5
思考-精确编码     0.6 / 0.95 / 20 / 0.0
非思考-通用      0.7 / 0.8  / 20 / 1.5
非思考-推理      1.0 / 1.0  / 40 / 2.0
```

同系列 0.8B 卡把 VL 与精确编码并为一档：

> Thinking mode for VL or precise coding tasks:
> temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0

官方另有一处警告：`presence_penalty` 取值过高会导致语言混杂与性能下降。

### 3.2 标注侧 · glm-4.6v（智谱 API）

| 阶段 | thinking | do_sample | temperature | top_p | response_format | max_tokens |
|---|---|---|---|---|---|---|
| 正向层 1 · 枚举 | `enabled` | `true` | 不传（=1.0） | 不传（=0.95） | 不用 | 8192 |
| 正向层 2 · 组句 | `disabled` | `true` | 不传（=1.0） | 不传 | 不用 | 256 |
| 逆向 · parse | `disabled` | `false` | 不传（被忽略） | 不传 | `json_object` | 256 |
| 逆向 · 枚举 | `enabled` | `true` | 不传 | 不传 | 不用 | 8192 |
| 逆向 · 兜底出框 | `disabled` | `false` | 不传 | 不传 | 不用 | 128 |

依据：智谱开放文档。

- `temperature` 取值 `[0.0, 1.0]`；GLM-4.6 系默认 `1.0`
- `top_p` 默认 `0.95`
- `temperature` 与 `top_p` 只调其中一个
- `do_sample=false` 为贪心，`temperature` / `top_p` 被忽略
- `glm-4.6v` 的 `max_tokens` 上限 32768，默认 16384
- 无 `seed`；采样阶段不可复现
- `thinking.type=enabled` 对 GLM-4.6 / GLM-4.6V 为模型自行判断，非强制
  （GLM-4.7 与 GLM-4.5V 强制思考；GLM-5.3 系强制且不可关闭）
- 官方未给思考 / 非思考的分档推荐值；上表用默认值

## 4. 主仓的门

以下是主仓 `ordinal.resolve` 返回的理由码，标注侧不实现这一套：它产出给的是人，
不做出包判定。

```
解析层
  1  parse-invalid        解析不出 {category, k, axis, direction}
  2  not-rank             非排序题

枚举层
  3  run-malformed        不是一份合法实例清单
  4  count-missing        自报数不是整数
  5  truncated            自报 count != 实际列出的个数
  6  count-zero           count == 0

选择层
  7  k-out-of-range       k > 清单长度
  8  axis-unsupported     该轴算不出值
```

`already-correct` 与 `replaced` 不是门，是终态：前者保留基准框，后者替换。

思考文本不参与判定：模型的推理过程只作诊断材料留存（`thinking.jsonl`），
不用来否决它自己的枚举结果。

## 5. run-id

```
do_sample=false 的阶段  →  贪心，可复现
do_sample=true  的阶段  →  采样，不可复现
```

run-id 标识一套参数配置。全部解码参数（含 `enable_thinking`）进 run-id。

## 6. 产物

主仓产物见 `data-contract.md`；标注侧输出 `outputs/{census,assembly,reverse}/<run_id>/`。
落盘原则：可重算的不落盘，模型输出与人审结果必留。
