# AGENTS.md — 本仓库 Agent 操作手册

静态规则；修改须经仓库管理员（用户本人）确认。
动态状态与进度一律看 `docs/handoff.md` 顶部，不写进本文件。

## 身份与环境边界

- 竞赛实验单元：RGBDT 三模态视觉定位，唯一指标 ACC@0.5
- 本地环境 = 仅 CPU 测试与静态检查（qwen_vg conda，Python 3.12）
- GPU 训练/推理在魔搭 DSW（offline/）或 Modal（cloud/，H100 硬编码）执行；
  `modal` 命令由用户本人执行
- 模型权重与大数据永不入库（data/、*.pt、*.safetensors 已 ignore，不得绕过）

## 常用命令

- 本地全量测试：`python -m unittest discover -s tests`（约 1s；GPU 相关项自动 skip）
- 语法检查：`python -m compileall aicomp_grounding scripts offline`
- GPU 验证：魔搭只跑冒烟，不跑 unittest——训练 smoke 与推理 val 切片命令见 `docs/sop.md`
- 环境安装：pin 单一来源 `envs/gpu.txt`（transformers==5.14.1，与 DSW 镜像一致）

## 铁律（违反 = 破坏历史 run 的可复现性）

1. 未经用户明确允许，不修改任何文件
2. 冻结合同不得改动：
   - `outputs/annotations/annot_dc189f029d962b27/**/approved.json`（golden 标注）
   - `data/Test/queries/queries.json`（官方模板，SHA-256 被 `test_data.py` 钉死）
   - 各 adapter 的 `MODEL_NAME` / `MODEL_REVISION` / prompt 常量
     （进 run-id 指纹，`tests/test_models.py` 钉死）
3. 改任何代码后必须先跑全量测试，绿了才算完成
4. 训练/推理参数变更一律用新 run-tag，不覆盖旧产物
5. 不新增顶层目录、不改模块布局；结构约定见 `docs/architecture.md`

## 文档阅读顺序（冷启动）

1. `docs/handoff.md` 顶部「当前状态」——干到哪了
2. `docs/architecture.md`——结构不变量
3. `docs/sop.md`——GPU 实操唯一真相源
4. `docs/data-contract.md`——data/ 布局与派生产物合同

## 平台注意

- 魔搭 DSW：A 卡 ROCm 7.2.3 / torch 2.11 底座镜像不可升级；持久空间 `/mnt/workspace`
- API_KEY 等凭据只走环境变量，永不入库
