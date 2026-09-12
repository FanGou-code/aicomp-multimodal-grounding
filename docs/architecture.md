# 仓库架构与协作接入指南

本文件记录模块与数据边界；运行进度见 `handoff.md`，GPU 操作见 `sop.md`。

## 两仓职责

| 位置 | 职责 |
| --- | --- |
| `aicomp_grounding/` | 图像、坐标、合同、运行身份、训练核心、模型适配、融合与提交 |
| `offline/` | 训练/推理 CLI 与共享运行编排 |
| `cloud/` | Modal 镜像、H100 与 Volume 包装，复用 `offline.run_cli` |
| `scripts/` | 深度伪彩预处理、数据上传工具 |
| `query-foundry` | 切分查重、教师普查、描述组装、人审与标注交付 |

依赖方向：`cloud → offline → aicomp_grounding`。主仓训练只消费 `approved.json`，
不导入数据构建仓代码。两仓不通过复制训练循环适配平台。

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
| `qwen3vl` | RGB、IR、深度三图 | LoRA | 0–1000 整数 |
| `qwen3_5` | RGB、IR、深度三图 | LoRA | 0–1000 整数 |
| `qwen36_27b` | RGB、IR、深度三图 | LoRA（Modal H100 专属） | 0–1000 整数 |
| `mimo_vl` | RGB、IR、深度三图 | LoRA | JSON bbox 归一化 |
| `glm46v` | RGB、IR、深度三图 | LoRA | 0–1000 整数 |
| `internvl35` | RGB、IR、深度三图 | LoRA，训练批量为 1 | 0–1000 整数 |
| `groundingdino` | RGB | 仅推理，原生置信度 | 归一化 XYXY |
| `mock` | 测试输入 | CPU 流程测试 | 归一化 XYXY |

所有适配器返回归一化 XYXY 与可选 score。训练接口负责输入、监督掩码、组批、
验证解码和 LoRA 目标层；普通训练循环不猜测模型协议。

底座先下载到本地，通过 `--model-path` 指定。适配器拒绝缺少配置的目录，所有
`from_pretrained` 调用采用 `local_files_only=True`，执行阶段不下载模型。
模型名称、revision 和提示词是冻结协议；更改训练目标、解析行为或运行参数时使用新标签。

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
