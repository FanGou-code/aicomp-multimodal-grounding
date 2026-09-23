# 仓库架构

本文件是代码地图与结构不变量；字段与产物格式见 `data-contract.md`。

## 分层

| 层 | 位置 | 职责 | 依赖 |
| --- | --- | --- | --- |
| 核心库 | `aicomp_grounding/` | 图像与坐标、合同校验、运行身份、训练核心、推理状态、融合与提交、序数后处理 | 标准库 + `Pillow` / `numpy` |
| 模型适配 | `aicomp_grounding/models/` | 各底座的协议适配与 LoRA 目标层声明 | 惰性导入 `torch` / `transformers` / `peft` |
| 入口 | `offline/`、`scripts/` | 训练与推理 CLI、数据预处理 | `aicomp_grounding` |
| 上游 | `query-foundry`（独立仓库） | 数据划分、标注生产、质检与封包 | 与本仓只通过 `approved.json` 交接 |

依赖方向单向：`offline/`、`scripts/` → `aicomp_grounding/`。核心库不导入入口代码与
上游仓库代码；`models/` 中的 `torch` / `transformers` / `peft` 在函数内惰性导入。

## 数据流

```text
原始图像 → scripts/prepare_rgbdt.py（三模态校验 + 深度伪彩）
        → query-foundry（划分 / 普查 / 组装 / 人审）
        → outputs/annotations/<run_id>/{train,val}/approved.json
        → offline/train.py → outputs/output_lora/<run_id>/
        → offline/infer.py → outputs/inference/<run_id>/predictions.json
        → aicomp_grounding.ordinal.{enumerate,resolve}（可选）→ outputs/{enum,ordinal}/
        → aicomp_grounding.fusion.wbf → aicomp_grounding.submission → submission.zip
```

## 模块地图

核心库（`aicomp_grounding/`）：

| 模块 | 职责 |
| --- | --- |
| `bbox.py` | 归一化 XYXY 的校验、像素框归一化、0–1000 量化与解析、IoU |
| `io.py` | JSON 读写（原子替换、拒绝重复键）、输入输出路径互斥检查 |
| `artifacts.py` | 内容哈希工具（`stable_json_hash` / `key_hash` / `file_set_fingerprint`）与元数据严格比对 |
| `config.py` | 跨模型常量：检查点版本、协议版本、运行期包清单、推理默认像素预算、split 枚举 |
| `paths.py` | 仓库相对路径解析与产物路径约定 |
| `prompts.py` | 三模态提示词协议与提示词哈希 |
| `query.py` | 查询文本的格式校验与风格门 |
| `sequence.py` | 序列级输入指纹、标注查询 QC |
| `sharding.py` | 序列感知的键分组与分片工具（生产推理分片见 `inference_state.assign_pending_shards`；本模块目前仅测试引用） |
| `images.py` | 图像引用指纹（路径 + 记录尺寸，不解码字节） |
| `test_data.py` | 官方 Test 模板合同、Test 准备合同（深度集合指纹） |
| `submission.py` | 由官方模板生成提交包（只补 `bbox`，ZIP 回读校验） |
| `annotation_state.py` | `approved.json` 产物合同校验（协议 12） |
| `training_state.py` | 训练身份构成、epoch 指标与 adapter manifest 校验、断点与完成态校验 |
| `training_core.py` | 训练计划、LoRA 注入与视觉塔中和、训练/验证循环、检查点持久化 |
| `inference_core.py` | 推理条目加载（三种输入形态）与 ACC@0.5 / mIoU 计算 |
| `inference_state.py` | 推理运行身份、分片分配、检查点校验与恢复 |
| `models/__init__.py` | 适配器注册表 |
| `models/base.py` | 适配器协议、输入/输出类型、LoRA 目标层构造、本地模型路径策略 |
| `models/qwen3vl.py` `models/qwen3_5.py` `models/glm46v.py` | 三个可训练 VLM 适配器（各自的提示词、坐标协议与超参默认值） |
| `models/mock.py` | CPU 契约测试用的最小适配器（无权重、确定性输出） |
| `fusion/wbf.py` | 多模型预测的加权框融合与融合身份 |
| `ordinal/resolve.py` | 序数后处理的门、轴排序、第 k 个选择（纯代码，无组决策） |
| `ordinal/parse.py` `ordinal/enumerate.py` | 序数后处理的提示词消息组装、思考切分与严格解码 |
| `ordinal/loader.py` | 读取 `ordinal/prompts/*.md` 并给出提示词指纹 |
| `ordinal/run.py` | 序数后处理的运行身份、产物读取、思考侧车与原始深度路径映射 |

入口与工具：

| 模块 | 职责 |
| --- | --- |
| `offline/train.py` | 训练 CLI：参数解析 → 训练计划 → 训练循环 → 产物交接 |
| `offline/infer.py` | 推理/评测 CLI：分片、断点续跑、指标与提交包判定 |
| `offline/rocm_env.sh` | ROCm 性能环境变量（BLAS 后端、硬件队列、缓存目录） |
| `scripts/prepare_rgbdt.py` | 三模态校验、深度伪彩生成、Test 引用与深度集合校验 |
| `scripts/upload_dataset.py` | 数据集发布工具（维护者用，需 `modelscope`） |
| `tests/` | CPU 单元测试：合同、身份、训练/推理状态机、mock 端到端链路 |

## 不变量

1. **冻结标识**：各适配器的 `MODEL_NAME` / `MODEL_REVISION` / 提示词常量、官方
   `data/Test/queries/queries.json`、golden `approved.json` 均不得改动。它们进入运行
   身份，改动会使既有 run 不再可复现、历史产物不可复算。
2. **LoRA 范围**：目标层一律经 `models/base.py:language_model_lora_targets()` 构造
   （锚定 `model.language_model` 的正则）；锚定共享，投影名单由各适配器自己声明。
   视觉塔恒为冻结、只做前向。裸后缀名单会被 PEFT 按后缀匹配而命中视觉塔。
3. **像素预算**：三个三模态适配器的 `min_pixels` / `max_pixels` 一律按单帧计并进入
   运行身份；处理器单位与单帧单位不一致的适配器在处理器边界换算。
4. **运行身份**：训练 id 由模型 revision、提示词哈希、数据与图像指纹、超参、种子与
   run-tag 哈希得出；推理 id 另含像素预算、分片数、limit 等。任何参数变更必须换新
   tag，不覆盖既有产物。
5. **坐标与提交**：解析器只接受各模型协议的 0–1000 或像素坐标，越界即判失败（不裁剪）；
   提交包只补 `bbox`，官方模板其余字段逐字节保留。
6. **单张量形状**：验证损失与训练共用同一 micro-batch 大小（默认 1）。
7. **离线加载**：所有 `from_pretrained` 使用 `local_files_only=True`；底座目录由调用者
   提供，本仓库不校验该目录与声明 revision 的对应关系。
8. **产物自洽**：`approved.json` 的 `source_fingerprint` 与 `dataset_fingerprint` 在
   读入时复算比对；`image_fingerprint` 在记录值带 `manifest_` 前缀时复算比对。

## 不提供

数据集、模型权重、标注产物与运行结果不在本仓库。平台相关差异集中在
`offline/rocm_env.sh`；`cloud/`、`internvl35`、`qwen36_27b`、`mimo_vl`、`groundingdino`
不在本仓库（git 历史可溯）。

## 契约边界

| 方向 | 内容 | 定义位置 |
| --- | --- | --- |
| 上游 → 本仓 | `outputs/annotations/<run_id>/{train,val}/approved.json`（协议 12） | `data-contract.md` |
| 本仓 → 提交 | `submission.zip`（官方模板 + `bbox`） | `data-contract.md` |
| 官方 → 本仓 | `data/Test/queries/queries.json`（条数与内容哈希被测试钉死） | `test_data.py` |

## 测试

`python -m unittest discover -s tests`：269 项，纯 CPU、不加载权重。真实权重加载、
生成质量、步时与显存不在覆盖内。
