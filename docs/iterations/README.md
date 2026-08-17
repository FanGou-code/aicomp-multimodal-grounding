# 📚 迭代文档目录

本目录记录项目的所有迭代优化过程和技术决策。

---

## 📂 目录结构

```
docs/
├── research/              # 研究分析文档
│   ├── competition_rules.md
│   └── research_summary.md
└── iterations/           # 迭代优化记录（本目录）
    ├── README.md
    ├── iteration_02_conservative_optimization.md
    └── iteration_02_summary.md
```

---

## 🔄 迭代历史

### Iteration 02: Conservative Optimization (2026-08-15)
**状态**: ✅ 已完成，待训练验证

**核心改进**:
- Task 02: 文本标准化（训练+推理两端）
- Task 03: 保守版训练策略（Rank16, Alpha48, 3 轮, 余弦退火）
- Task 04: 推理后处理（BBox校准 + Selective Retry）
- Task 01: 已放弃（实测数据证伪深度13m假设）

**预期提升**: +1.1~2.4% (0.7439 → 0.7550~0.7683)

**文档**:
- [`iteration_02_conservative_optimization.md`](./iteration_02_conservative_optimization.md) - 完整技术报告
- [`iteration_02_summary.md`](./iteration_02_summary.md) - 执行总结和操作指南

---

### Iteration 01: Baseline (2026-08-14)
**状态**: ✅ 已完成

**基线配置**:
- 模型: Qwen3-VL-8B + LoRA (Rank 16, Alpha 32)
- 训练: 2 epochs, Linear LR schedule
- 数据: RGBDT500, 2875 train samples
- 硬件: Modal A100-80GB

**成绩**: Val IoU 0.7439

**文档**: 见项目根目录 `README.md`

---

## 📖 文档命名规范

### 迭代文档
```
iteration_<序号>_<简短描述>.md         # 完整技术报告
iteration_<序号>_summary.md           # 执行总结
```

### 内容要求
每个迭代的完整文档应包含：
1. **背景与动机** - 为什么做这次迭代
2. **技术方案** - 具体修改了什么
3. **实施细节** - 代码改动位置和逻辑
4. **验证结果** - 测试和验证数据
5. **预期收益** - 理论提升和风险评估
6. **操作指南** - 如何复现和部署

---

## 🎯 使用指南

### 查看最新迭代
```bash
ls -lt docs/iterations/*.md | head -5
```

### 对比不同迭代
```bash
diff docs/iterations/iteration_01_*.md docs/iterations/iteration_02_*.md
```

### 生成迭代报告模板
参考 `iteration_02_conservative_optimization.md` 的结构

---

## 📊 迭代效果追踪

| 迭代 | 日期 | Val IoU | Test IoU | 排名 | 关键改动 |
|------|------|---------|----------|------|---------|
| 01 Baseline | 2026-08-14 | 0.7439 | - | - | 基线模型 |
| 02 Conservative | 2026-08-15 | TBD | TBD | TBD | 文本标准化+保守训练+后处理 |

*TBD: 待训练完成后更新*

---

**维护者**: FanGou-code  
**最后更新**: 2026-08-15
