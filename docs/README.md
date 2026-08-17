# 📚 项目文档组织结构

本项目采用双层文档架构：**迭代级**（宏观策略）+ **任务级**（微观实施）。

---

## 📂 目录结构

```
aicomp-multimodal-grounding/
│
├── docs/                          # 📖 项目文档中心
│   ├── README.md                  # 文档中心总览与规范
│   ├── research/                  # 研究分析
│   │   ├── competition_rules.md        # 竞赛规则
│   │   └── research_summary.md         # 技术研究总结
│   │
│   ├── iterations/                # 🔄 迭代记录（宏观）
│   │   ├── README.md                   # 迭代目录索引
│   │   ├── iteration_02_conservative_optimization.md  # 完整技术报告
│   │   └── iteration_02_summary.md                    # 执行总结
│   │
│   ├── reports/                   # 📊 专项报告
│   │   └── BUG_FIX_REPORT.md           # 历史缺陷与修复报告
│   │
│   └── archive/                   # 🗄️ 归档库 (过时方案与历史草稿备查)
│       └── README.md                   # 归档库索引与规范
│
├── tasks/                         # 📋 任务清单（微观）
│   ├── README.md                       # 任务索引
│   ├── task_01_preprocessing_enhancement.md
│   ├── task_02_text_standardization.md
│   ├── task_03_training_strategy_upgrade.md
│   └── task_04_inference_postprocessing.md
│
├── README.md                      # 项目主文档
└── CLAUDE.md                      # Claude Code 工作指南
```

---

## 🎯 文档分层说明

### 第一层：迭代文档（`docs/iterations/`）
**目的**: 记录完整迭代的战略决策和整体成果

**受众**: 项目管理者、技术评审、未来维护者

**内容**:
- 为什么做这次迭代？（背景与动机）
- 整体技术方案是什么？
- 预期收益和风险评估
- 如何复现和部署？
- 最终成果和经验教训

**示例**: `iteration_02_conservative_optimization.md`

---

### 第二层：任务文档（`tasks/`）
**目的**: 记录具体任务的实施细节和代码修改

**受众**: 开发者、代码审查者

**内容**:
- 任务具体做什么？
- 修改了哪些文件的哪些行？
- 如何验证代码正确性？
- 遇到了什么问题和如何解决？
- 为什么选择这个技术方案而非其他？

**示例**: `task_02_text_standardization.md`

---

## 📖 使用场景

### 场景1: 快速了解项目当前状态
```bash
# 1. 看项目主README
cat README.md

# 2. 看最新迭代
cat docs/iterations/iteration_02_summary.md
```

### 场景2: 了解某个具体功能如何实现的
```bash
# 先看任务文档（微观）
cat tasks/task_02_text_standardization.md

# 再看代码
cat aicomp_grounding/prompts.py
```

### 场景3: 准备技术报告（半决赛/论文）
```bash
# 从迭代文档提取内容（宏观）
cat docs/iterations/iteration_02_conservative_optimization.md

# 补充实施细节
cat tasks/task_0*.md
```

### 场景4: 对比不同方案的决策过程
```bash
# 看为什么放弃Task 01
cat tasks/task_01_preprocessing_enhancement.md

# 看为什么采用Task 03保守版
cat tasks/task_03_training_strategy_upgrade.md
```

---

## 🏗️ 文档维护规范

### 新增迭代时
1. 在 `docs/iterations/` 创建新迭代文档
2. 在 `tasks/` 创建具体任务文档
3. 更新两个目录的 `README.md` 索引
4. 在项目根目录 `README.md` 更新最新成绩

### 文档命名规范
```
迭代文档: iteration_<序号>_<简短描述>.md
任务文档: task_<序号>_<任务名称>.md
```

### 文档状态标记
- ✅ 已完成
- 🚧 进行中
- ❌ 已放弃
- 📅 已计划

---

## 📊 当前项目状态

| 迭代 | 状态 | Val IoU | 文档 |
|------|------|---------|------|
| Iteration 01 | ✅ 完成 | 0.7439 | 见根目录README |
| Iteration 02 | 🚧 待训练 | TBD | [`docs/iterations/`](./docs/iterations/) |

| 任务 (Iter 02) | 状态 | 预期提升 |
|----------------|------|----------|
| Task 01 | ❌ 放弃 | - |
| Task 02 | ✅ 完成 | +0.3~0.8% |
| Task 03 | ✅ 完成 | +0.5~1.0% |
| Task 04 | ✅ 完成 | +0.3~0.6% |

---

## 🚀 快速导航

- **项目概览**: [`README.md`](./README.md)
- **最新迭代**: [`docs/iterations/iteration_02_summary.md`](./docs/iterations/iteration_02_summary.md)
- **研究总结**: [`docs/research/research_summary.md`](./docs/research/research_summary.md)
- **任务清单**: [`tasks/README.md`](./tasks/README.md)

---

**维护者**: FanGou-code  
**最后更新**: 2026-08-15
