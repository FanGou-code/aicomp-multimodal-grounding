# 仓库架构

本文件是代码地图与结构不变量；字段与产物格式见 `data-contract.md`。

## 分层

| 层 | 位置 | 职责 | 依赖 |
| --- | --- | --- | --- |
| 共享底座 | `aicomp_grounding/*.py` | 坐标、哈希与产物身份、协议 12 合同、图像引用、路径与运行配置 | 标准库 + `Pillow` / `numpy` |
| 服务侧 | `aicomp_grounding/serving/` | 训练、推理、融合、序数后处理、提交包、消息构造 | 惰性导入 `torch` / `transformers` / `peft` |
| 标注侧 | `aicomp_grounding/annotation/` | 普查、生成、人审、封包、逆向通路 | 共享底座 + 教师 API |
| 入口 | `tools/` | 训练 / 推理 / 预处理 / 标注流水线 CLI | 上述三层 |

依赖方向单向：`tools/` → `aicomp_grounding/`；标注侧与服务侧互不导入，共用共享底座。
`serving/models/` 中的 `torch` / `transformers` / `peft` 在函数内惰性导入。

## 数据流

```text
原始图像 → tools/prepare_split.py（按序列划分 + 跨集去重）→ data/indexes/
        → tools/run_census.py（每帧一次枚举）
        → tools/run_generation.py（教师为每个目标生成一句）
        → tools/review_server.py 人审 → tools/package_approved.py
        → outputs/annotations/<run_id>/{train,val}/approved.json
        → tools/train.py → outputs/training/<run_id>/
        → tools/infer.py → outputs/inference/<run_id>/predictions.json
        → aicomp_grounding.serving.fusion → outputs/fusion/<run_id>/predictions.json
        → aicomp_grounding.serving.ordinal.{enumerate,resolve}（可选，以融合产物为基准底座修正序数题）
          → outputs/{ordinal_enum,ordinal_resolve}/<run_id>/
        → aicomp_grounding.serving.submission → submission.zip
```

`tools/run_reverse.py` 走同一内核的反向入口：给一句 query，教师产出框，用于抽查逆向一致性。

## 模块地图

共享底座（`aicomp_grounding/`）：

| 模块 | 职责 |
| --- | --- |
| `bbox.py` | 归一化 XYXY 的校验、像素框归一化、0–1000 量化与解析、IoU |
| `io.py` | JSON 读写（原子替换、拒绝重复键）、输入输出路径互斥检查 |
| `artifacts.py` | 内容哈希工具（`stable_json_hash` / `key_hash` / `file_set_fingerprint`）与元数据严格比对 |
| `contract.py` | `approved.json` 产物合同与指纹（协议 12）、结构校验、训练产物互查 |
| `query.py` | 查询文本的格式校验、标注级 QC（脚手架/泛类词）与风格门、数据前检 |
| `images.py` | 图像引用指纹（不解码字节）与字节级校验 |
| `sharding.py` | 序列感知的键分组与分片工具（生产推理分片见 `serving.engine.inference_state`） |
| `ordinal_kernel.py` | 序数内核：轴值、排序键与平局规则（两侧共用，从原 `ordinal/` 提级） |
| `paths.py` | 仓库相对路径解析；产物布局单一来源（`OUTPUT_FAMILIES` / `output_dir()`） |
| `config.py` | 跨模型常量：检查点与协议版本、运行期包清单、推理默认像素预算、split 枚举 |
| `testset.py` | 官方 Test 模板合同与处理索引校验 |

服务侧（`aicomp_grounding/serving/`）：

| 模块 | 职责 |
| --- | --- |
| `messages.py` + `prompts/{grounding,glm46v}.md` | 可见光单图消息构造；提示词文本以 `[system]`/`[user]` 两段的文件随包，两者一起进入运行身份 |
| `prompt_files.py` | `[system]` / `[user]` 提示词文件的共享读取与拆分（缺失、空或畸形即报错） |
| `engine/training_state.py` `engine/training_core.py` | 训练身份与计划、LoRA 注入与视觉塔中和、训练/验证循环、检查点持久化 |
| `engine/inference_core.py` `engine/inference_state.py` | 推理条目加载、ACC@0.5 / mIoU 计算、分片分配与检查点校验恢复 |
| `models/base.py` | 适配器协议、输入/输出类型、LoRA 目标层构造、本地模型路径策略 |
| `models/qwen3vl.py` `models/qwen3_5.py` `models/glm46v.py` | 三个可训练 VLM 适配器（各自的坐标协议与超参默认值；提示词文本在 `prompts/`） |
| `models/mock.py` | CPU 契约测试用的最小适配器（无权重、确定性输出） |
| `fusion.py` | 多模型预测的加权框融合与融合身份 |
| `submission.py` | 由官方模板生成提交包（只补 `bbox`，ZIP 回读校验） |
| `ordinal/resolve.py` | 序数后处理的门、轴排序、第 k 个选择（纯代码，无组决策） |
| `ordinal/parse.py` `ordinal/enumerate.py` | 序数后处理的提示词消息组装、思考切分与严格解码 |
| `ordinal/loader.py` | 读取 `ordinal/prompts/*.md` 并给出提示词指纹 |
| `ordinal/run.py` | 序数后处理的运行身份、产物读取、思考侧车与原始深度路径映射 |

标注侧（`aicomp_grounding/annotation/`）：

| 模块 | 职责 |
| --- | --- |
| `config.py` | 教师模型身份（`ANNOTATION_*`）与索引目录定位 |
| `client.py` | 单 key 注入的 OpenAI 协议客户端；错误一律有限退避重试后终止 |
| `prompts/*.md` | 教师提示词配方（普查/属性/枚举/解析/直接定位/组句），运行时可改 |
| `source.py` | 划分索引加载与校验、分裂索引指纹 |
| `census.py` | 普查：每帧一次枚举，代码判定计数门 |
| `facts.py` `depth.py` | 帧级事实与深度事实提取（纯数据） |
| `selection.py` | 候选去重与选用 |
| `realize.py` `generation.py` | 单句生成与整轮生成：教师组句、全轮去重、审计 |
| `reverse.py` | 逆向通路：query → 框，与正向共用序数内核 |
| `imaging.py` | 标记图渲染与 JPEG data URL |
| `review/store.py` | 崩溃安全标注库（WAL、快照重放、进程锁） |
| `review/sessions.py` | 由普查/生成产物构建审查会话 |
| `review/server.py` `review/static/` | 人审服务与前端 |

入口与工具（`tools/`）：

| 模块 | 职责 |
| --- | --- |
| `train.py` / `infer.py` | 训练与推理评测 CLI |
| `prepare_split.py` | 按序列划分 train/val、跨集去重与索引落盘 |
| `run_census.py` / `run_generation.py` / `run_reverse.py` | 标注三阶段入口 |
| `review_server.py` / `review_report.py` / `apply_review.py` | 人审服务、审查报告、裁决落盘 |
| `make_manifest.py` | 由 generation/任意 query JSON 构建人审 manifest（可拆分多份） |
| `package_approved.py` | 组装产物封包为 `approved.json` 并做合同自校验 |
| `fusion.py` / `submission.py` / `ordinal_enumerate.py` / `ordinal_resolve.py` | 融合、提交包、序数枚举与修正（服务侧后处理） |
| `check_key.py` | 注入 key 的测活（管理员执行真实调用） |
| `setup_cuda.sh` | CUDA 环境安装：按驱动选 cu130/cu128 轮子装 torch/torchvision 并做健康自检 |

## 不变量

1. **冻结标识**：各适配器的 `MODEL_NAME` / `MODEL_REVISION` / 提示词、官方
   `data/Test/queries/queries.json`、golden `approved.json` 均不得改动。它们进入运行
   身份，改动会使既有 run 不再可复现、历史产物不可复算。
2. **LoRA 范围**：目标层一律经 `serving/models/base.py:language_model_lora_targets()` 构造
   （锚定 `model.language_model` 的正则）；锚定共享，投影名单由各适配器自己声明。
   视觉塔恒为冻结、只做前向。裸后缀名单会被 PEFT 按后缀匹配而命中视觉塔。
3. **像素预算**：三个可训练适配器的 `min_pixels` / `max_pixels` 一律按单帧计并进入
   运行身份；处理器单位与单帧单位不一致的适配器在处理器边界换算。
4. **运行身份**：训练与推理 id 由模型 revision、提示词哈希、数据与图像指纹、超参、
   种子与 run-tag 哈希得出。任何参数变更必须换新 tag，不覆盖既有产物。
5. **坐标与提交**：解析器只接受各模型协议的 0–1000 或像素坐标，越界即判失败（不裁剪）；
   提交包只补 `bbox`，官方模板其余字段逐字节保留。
6. **单张量形状**：验证损失与训练共用同一 micro-batch 大小（默认 1）。
7. **离线加载**：所有 `from_pretrained` 使用 `local_files_only=True`；底座目录由调用者
   提供，本仓库不校验该目录与声明 revision 的对应关系。
8. **产物自洽**：`approved.json` 的 `source_fingerprint` 与 `dataset_fingerprint` 在
   读入时复算比对；`image_fingerprint` 在记录值带 `manifest_` 前缀时复算比对。
9. **标注侧教师输出不校语义**：`generation` 只把事实交给教师，不校验写回的句子；
   措辞由人审逐条过。普查的计数门由代码判定（`count == len(instances)`）。

## 不提供

数据集、模型权重、标注产物与运行结果不在本仓库；`cloud/`、`internvl35`、
`qwen36_27b`、`mimo_vl`、`groundingdino` 不在本仓库（git 历史可溯）。平台专属工具
（如数据集上传）放在被 `.gitignore` 排除的 `local/`，仓库本体不含平台专属文件。

## 契约边界

| 方向 | 内容 | 定义位置 |
| --- | --- | --- |
| 标注侧 → 服务侧 | `outputs/annotations/<run_id>/{train,val}/approved.json`（协议 12） | `data-contract.md` |
| 本仓 → 提交 | `submission.zip`（官方模板 + `bbox`） | `data-contract.md` |
| 官方 → 本仓 | `data/Test/queries/queries.json`（条数与内容哈希被测试钉死） | `testset.py` |
| 两侧共用 | 序数内核：轴值、排序键与平局规则 | `ordinal_kernel.py` |

## 测试

`python -m unittest discover -s tests`：434 项，纯 CPU、不加载权重。真实权重加载、
生成质量、步时与显存不在覆盖内。
