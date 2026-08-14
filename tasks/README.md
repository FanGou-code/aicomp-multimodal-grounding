# 📋 任务清单

本目录包含本次迭代（Iteration 02）的详细任务分解和实施记录。

---

## 📂 与迭代文档的关系

```
项目文档层级：
├── docs/iterations/           # 迭代级别（宏观）
│   ├── iteration_02_conservative_optimization.md   # 完整技术报告
│   └── iteration_02_summary.md                     # 执行总结
│
└── tasks/                     # 任务级别（微观）
    ├── task_01_*.md          # 具体任务实施细节
    ├── task_02_*.md
    ├── task_03_*.md
    └── task_04_*.md
```

**区别**:
- **`docs/iterations/`**: 面向项目整体，记录完整迭代的背景、方案、成果
- **`tasks/`**: 面向开发执行，记录每个任务的具体实施步骤和代码修改

---

## 📋 Iteration 02 任务列表

### ❌ Task 01: 数据预处理增强
**状态**: 已放弃  
**理由**: 实测数据证伪（25%目标在13~20m，截断13m会损害性能）  
**文档**: [`task_01_preprocessing_enhancement.md`](./task_01_preprocessing_enhancement.md)

### ✅ Task 02: 文本标准化
**状态**: 已完成  
**改动**: `prompts.py`, `train_modal.py`, `run_inference.py`  
**预期**: +0.3~0.8%  
**文档**: [`task_02_text_standardization.md`](./task_02_text_standardization.md)

### ✅ Task 03: 训练策略升级（保守版）
**状态**: 已完成  
**改动**: `train_modal.py`  
**预期**: +0.5~1.0%  
**文档**: [`task_03_training_strategy_upgrade.md`](./task_03_training_strategy_upgrade.md)

### ✅ Task 04: 推理后处理（修正版）
**状态**: 已完成  
**改动**: `bbox.py`, `run_inference.py`  
**预期**: +0.3~0.6%  
**文档**: [`task_04_inference_postprocessing.md`](./task_04_inference_postprocessing.md)

---

## 📊 任务统计

| 指标 | 数值 |
|------|------|
| 总任务数 | 4 |
| 已完成 | 3 |
| 已放弃 | 1 |
| 修改文件数 | 4 |
| 新增代码行数 | +168 |
| 累计预期提升 | +1.1~2.4% |

---

## 🔍 任务文档结构规范

每个任务文档应包含：
1. **任务目标** - 要解决什么问题
2. **技术方案** - 如何实现
3. **实施步骤** - 具体代码修改位置
4. **验证方法** - 如何测试
5. **风险评估** - 可能的问题
6. **预期收益** - 理论提升

---

## 🎯 如何使用

### 开发时
1. 从 `task_*.md` 了解具体实施细节
2. 按文档中的步骤修改代码
3. 运行文档中的验证脚本

### 复盘时
1. 从 `docs/iterations/` 了解整体迭代思路
2. 查看 `tasks/` 了解具体技术决策过程
3. 对比不同任务的风险/收益评估

---

**最后更新**: 2026-08-15  
**对应迭代**: Iteration 02 - Conservative Optimization
