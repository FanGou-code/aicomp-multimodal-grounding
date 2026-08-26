# 仓库架构与协作接入指南

> 本文是三模型仓库的架构约定。数据、成绩等实际状态见 `handoff.md`，
> 本文件只描述**结构不变量**：改代码前先读这里。

## 总览

```text
┌───────────────────────────── aicomp_grounding (核心库) ─────────────────────────────┐
│                                                                                     │
│  数据层    prepare/annotate/QC (annotation_state, sharding, sequence, query,       │
│                                query_style, ...)                                   │
│  身份层    指纹与 run id (artifacts, training_state, inference_state)                │
│  推理核心  inference_core (items 加载 / ACC@0.5 评估 / 分片合并)                     │
│  训练核心  training_core (adapter 驱动训练循环, 平台无关)                          │
│  模型适配  models/ (qwen3vl | internvl35 | groundingdino | mock)                    │
│  融合      fusion/wbf (多模型加权框融合)                                             │
│  提交      submission (官方模板合同 + ZIP 构建)                                      │
│                                                                                     │
├────────── cloud/ (Modal 壳) ──────────┬────────── offline/ (离线壳) ────────────────┤
│  train.py   H100 训练                  │  train.py   单机训练 (同一 training_core) │
│  infer.py   分片推理 (--model)         │  infer.py   单卡推理 (--model)             │
└────────────────────────────────────────┴─────────────────────────────────────────────┘
```

## Portable Repository Path Contract

The repository is the portable experiment unit copied to a lab workstation,
cloud workspace, or Modal-backed job. The platform-independent entrypoints in
`offline/` use these roots:

```text
PROJECT_ROOT/                 repository root
PROJECT_ROOT/data/             dataset files and generated indexes
PROJECT_ROOT/outputs/annotations/  approved query artifacts
PROJECT_ROOT/outputs/          training, inference, fusion, submission outputs
```

`data_root` is only the dataset root. It must not be used to derive
`data/outputs/annotations`; portable callers pass the repository-level
annotation root explicitly. The historical `data/outputs/annotations` layout
is retained only as a compatibility path inside the Modal adapter.

`data/test.json` is optional. The official `data/Test/queries/queries.json`
can be sent directly to inference-core workers; it is mapped in memory to
`Test/Images/...` and `Processed/Test/depth_jet/...`, so no separate processed
index file is required.

两条铁律：

1. **业务逻辑只存在于核心库**。cloud/ 与 offline/ 是平台壳，只做平台相关的事
   （Modal 装饰器/卷挂载、本地 CLI），互相不 import。
2. **平台的差异 = 环境安装的差异**。推理/训练本体平台无关；各平台的装法
   （镜像源、锁版本）见 `offline/README.md` 的平台矩阵，不新增目录或脚本。

## 模型适配层（models/）

统一接口（`models/base.py`）：

```python
adapter.name               # "qwen3vl" | "internvl35" | "groundingdino" | "mock"
adapter.model_name         # HF 模型 id（进指纹）
adapter.model_revision     # revision（进指纹；记录 run 前必须 pin 具体 commit）
adapter.supports_lora      # VLM=True, DINO=False
adapter.generation_config  # 解码参数（进 run identity）
adapter.prompt_hash()      # prompt 协议哈希（进 run identity）
adapter.load(device, lora_path=None, model_path=None)   # 惰性导入重依赖
adapter.predict([ModelInput(visible, infrared, depth, query, key)]) -> [Prediction(bbox, score)]
```

- **输出统一为归一化 XYXY + score(可 None)**：坐标换算全部封装在 adapter 内
  （Qwen `<|box_start|>` 0-1000、InternVL `<box>` 0-1000、DINO cxcywh 归一化）。
- **纯逻辑（prompt 构造/解析/坐标换算）是模块级函数**，无 GPU 也能单测；
  `load/predict` 才需要 GPU。
- **训练能力由 `TrainableGroundingAdapter` 提供**：`load_for_training`、
  `build_training_batch`、`collate_training_batch`、验证解码和 LoRA target 都由 adapter 实现。
- **指纹连续性**：`qwen3vl` 的 identity 值被 `tests/test_models.py` 钉死，
  保证 Qwen 历史 run id 永不漂移。新模型接入 = 新 identity，天然隔离。

### 各模型输入策略（已论证，勿随意改动）

| | Qwen3-VL | InternVL3.5 | GroundingDINO |
|---|---|---|---|
| 输入 | RGB+IR+Depth 三图 | 同左 | **仅 RGB**（单图架构） |
| 分辨率机制 | 整图动态分辨率 `max_pixels` | 448 tile 网格 `max_num_tiles` | 内部 resize ~800×1333 |
| prompt | 系统提示 + `Locate:` | 官方 grounding prompt + `<ref>` | 裸 query（无模板） |
| 置信度 | 无 | 无 | **原生 score**（WBF 用） |

### 验证状态（诚实标注）

- `qwen3vl`：GPU 路径与历史产出 0.7439 的代码同源，行为等价搬运。
- `internvl35` / `groundingdino`：**纯逻辑有单测，GPU 路径未冒烟**。首次使用
  必须先跑 val 小切片：InternVL 对照官方 `evaluate_grounding.py` 钉坐标序，
  DINO 验证 `post_process` 输出形状。两者的 `model_revision` 已 pin 具体 commit。

## 指纹与溯源

一切 run（训练/推理/融合）的 id = hash(数据指纹 + 模型 identity + prompt 协议 +
参数 + seed + Python runtime 版本)。三模型经 `adapter.identity()` 注入，互不污染；融合 run id =
hash(输入 predictions 文件字节 + 权重 + 阈值)，可回溯到每一次推理。

## 协作工作流

```bash
git clone                         # 代码 + 已批准标注集 (approved.json 白名单入库)
# 大数据 (data/ 43G) 与权重自行获取，见 offline/README.md 外部数据清单

# 1. zero-shot 跑通（单张 24GB GPU 即可）
python offline/infer.py --model internvl35 --test-json data/Test/queries/queries.json --limit 100 ...
python offline/infer.py --model groundingdino --test-json data/Test/queries/queries.json --limit 100 ...

# 2. 训练（通过 --model 选择已接入训练循环的 adapter；标注 run id 用当前 golden 值）
modal run cloud/train.py --model <name> --annotation-run-id YOUR_ANNOTATION_RUN_ID ...

# 3. 交回 predictions_*.json（WBF 只交换预测文件，不交换权重）

# 4. 融合
python -m aicomp_grounding.fusion.wbf --predictions qwen.json internvl.json dino.json \
    --weights 1 1 1 --scores '' '' dino_scores.json --test-json data/Test/queries/queries.json
```

分支约定：fork → 特性分支 → PR 回主仓库，由仓库管理员 review 合并。
