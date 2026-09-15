# 仓库架构与协作接入指南

本文件记录模块与数据边界；GPU 操作见 `sop.md`，数据布局见 `data-contract.md`。

## 模块职责

| 位置 | 职责 |
| --- | --- |
| `aicomp_grounding/` | 图像、坐标、合同、运行身份、训练核心、模型适配、融合与提交 |
| `offline/` | 训练/推理 CLI 与运行编排、ROCm 环境脚本 |
| `scripts/` | 深度伪彩预处理、数据上传工具 |
| `query-foundry` | 切分查重、教师普查、描述组装、人审与标注交付（独立仓库） |

依赖方向：`offline → aicomp_grounding`。主仓训练只消费 `approved.json`，
不导入数据构建仓代码。

数据流：原始图像 → 深度伪彩与索引 → 普查/组装/人审 → `approved.json`
→ 训练/验证/推理 → `fusion.wbf` → 官方模板 ZIP。

## 路径

```text
PROJECT_ROOT/
  data/                       Train/、Test/、Processed/
  outputs/annotations/        标注合同产物
  outputs/output_lora/        训练记录和 LoRA
  outputs/inference/          预测与 checkpoint
```

标注源索引位于 `query-foundry/data/indexes/`。数据构建入口分别接收图片根
`--data-root` 和索引目录 `--index-dir`，不依赖跨仓符号链接。

CLI 相对路径以 `--project-root` 解析。新训练可以使用不同输出根；恢复已有训练
记录仍要求保留其记录的绝对路径，不提供跨目录迁移。已训练 LoRA 可单独用于推理。

官方 `data/Test/queries/queries.json` 在内存中映射图片路径，不另造 `test.json`。
提交文件仅补 `bbox`，保留官方模板其余字段。

## 模型适配

| 名称 | 输入 | 训练能力 | 坐标 |
| --- | --- | --- | --- |
| `qwen3vl` | RGB、IR、深度三图 | LoRA（仅语言模型） | 0–1000 整数 |
| `qwen3_5` | RGB、IR、深度三图 | LoRA（仅语言模型） | 0–1000 整数 |
| `mimo_vl` | RGB、IR、深度三图 | LoRA（仅语言模型） | JSON bbox 归一化 |
| `glm46v` | RGB、IR、深度三图 | LoRA（仅语言模型） | 0–1000 整数 |
| `groundingdino` | RGB | 仅推理，原生置信度 | 归一化 XYXY |
| `mock` | 测试输入 | CPU 流程测试 | 归一化 XYXY |

所有适配器返回归一化 XYXY 与可选 score。训练接口负责输入、监督掩码、组批、
验证解码和 LoRA 目标层；普通训练循环不猜测模型协议。

**LoRA 范围不变量**：适配器的目标层一律由 `models/base.py` 的
`language_model_lora_targets()` 构造，即锚定 `model.language_model` 的正则。
**锚定是共享的，名单是各适配器自己的**——语言模型暴露哪些投影属于该模型的
个性（`glm46v` 的 MLP 融合为 `gate_up_proj`，只声明 `q/k/v/o/down_proj`；
其余三个声明共享的 `DEFAULT_LORA_PROJECTIONS`）。视觉塔（`model.visual.*`、
`model.vision_tower.*`）恒为冻结，只做前向。

裸后缀名单（`["q_proj", ...]`）由 PEFT 按后缀匹配，会误伤复用同名投影的视觉塔：
Qwen2.5-VL 系视觉 MLP 的 `gate_proj`/`up_proj`/`down_proj` 会随 `mimo_vl` 被命中。
视觉塔一旦进入可训练集，其反向与梯度检查点重算即被强制打开，各适配器之间也不再
可比。改动构造器或任一适配器的名单，须同步 `tests/test_models.py` 的强制覆盖项、
语言模型侧正向断言与视觉侧反向断言。

**像素预算单位**：四个三模态适配器的 `min_pixels`/`max_pixels` 一律按单帧计，
并进 run 身份。`glm46v` 的处理器按 `temporal_factor × h × w` 比较，适配器在传给
处理器时统一乘 `GLM_PIXEL_UNIT_FACTOR`；其余三个处理器的 `max_pixels` 本身就是
单帧单位。改动换算或预算值属配方变更，须换新标签。

底座先下载到本地，通过 `--model-path` 指定。适配器拒绝缺少配置的目录，所有
`from_pretrained` 调用采用 `local_files_only=True`，执行阶段不下载模型。
模型名称、revision 和提示词是冻结协议；`MODEL_REVISION` 一律填来源仓库（魔搭或
HF）的 commit id，取值与 `sop.md` 下载命令的 `--revision` 相同——分支名会让身份
字符串不变而权重移动。更改训练目标、解析行为或运行参数时使用新标签。

## 身份和恢复

训练/推理身份绑定模型版本、提示词、数据与参数。LoRA 指纹读取配置、清单和权重；
底座目录使用调用者预下载的快照，不自动核验该目录与声明 revision 的对应关系。
当前 `manifest_` 图像指纹绑定路径与尺寸；图像字节版本须单独固定，深检查见数据合同。

推理恢复先验证 checkpoint、预测文件和各分片的运行身份、框、分数与键集合，
冲突结果拒绝合并。仅有预测文件时必须同时有记录元数据。新分片保存完整分配列表；
旧分片缺少列表时校验运行身份及结果键，保留原分配哈希，不伪造旧分配记录。
LoRA 的机器路径只用于追溯，内容指纹一致时允许推理端重新定位它。

单卡使用一个分片。单机多卡可用 `--num-shards`；分片进程使用 spawn，CPU 数据
加载使用 Linux fork，数据加载子进程不访问 GPU。没有剩余工作时直接汇总。

## 协作

特性分支 → PR，由仓库管理员审核合并。

特性分支提交修改；CPU 全量测试和静态检查通过后，再按 SOP 做实际模型的 GPU
小样本验证。模型和大数据不入 Git，冻结标注与历史产物不覆盖。
