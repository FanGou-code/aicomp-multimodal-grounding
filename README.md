# RGBDT Multimodal Visual Grounding

基于统一 adapter 接口的 RGB / 红外 / 深度视觉定位**可移植实验单元**。
输入时间同步、空间对齐的 RGB、Infrared、Depth 图像与英文 Query，输出目标在可见光
图像中的归一化边界框 `[x1, y1, x2, y2]`；唯一评分指标 `ACC@0.5`（预测框与 GT 框
IoU ≥ 0.5）。

训练与推理核心（`aicomp_grounding/`）平台无关；`offline/` 与 `cloud/` 是薄壳，
只承载平台差异（环境安装）。模型阵容、成绩、超参与待办以 `docs/handoff.md`
「当前状态」为准，不写在本文件。

## 环境安装

- Python 3.12；依赖唯一声明源 = 根 `pyproject.toml`

```bash
pip install -e .             # GPU 算力平台：纯净核心模型与计算依赖（torch 用范围，平台已有版本自动跳过）
pip install -e ".[dev]"      # 本地开发机：完整工具链（代码检查 + Modal 调度 + ModelScope 数据管理）
pip install -e ".[kernels]"  # 仅混合线性注意力架构模型按需
```

DSW 完整实操见 `docs/sop.md`。

## 运行入口

```bash
python offline/train.py --model <name> --model-path /path/to/model --annotation-run-id <id> ...
python offline/infer.py --model <name> --model-path /path/to/model --test-json data/Test/queries/queries.json ...
python -m aicomp_grounding.fusion.wbf --predictions ... --test-json ...
python -m aicomp_grounding.submission --test-json ... --predictions ... --output-dir ...
```

`--model` 可选值与各 adapter 约定见 `docs/architecture.md`；具体模型名与 revision
以代码（`aicomp_grounding/models/`）为准。模型先按指定版本下载，训练和推理仅加载
本地目录，不自动联网补文件。`mock` 和仅做数据检查的 `--preflight-only` 不需要底座权重。

## 数据

训练图像来自 [RGBDT500](https://xuefeng-zhu5.github.io/RGBDT500/) 公开数据集
（research-only 许可），本仓库不重新分发原始数据。`data/` 布局与派生产物见
`docs/data-contract.md`。

引用：

```bibtex
@inproceedings{Zhu_RGBDT500,
  author    = {Xue-Feng Zhu and Tianyang Xu and Yifan Pan and Jinjie Gu and
               Xi Li and Jiwen Lu and Xiao-Jun Wu and Josef Kittler},
  title     = {Collaborating Vision, Depth, and Thermal Signals for
               Multi-Modal Tracking: Dataset and Algorithm},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```

## 文档导航

| 文档 | 内容 |
| --- | --- |
| `docs/handoff.md` | 当前状态、成绩、交接日志（唯一状态真相源） |
| `docs/architecture.md` | 结构不变量、adapter 约定 |
| `docs/sop.md` | GPU 实操 |
| `docs/data-contract.md` | data/ 布局与派生产物 |
| `docs/research.md` | 赛题规则 |

## 测试

```bash
python -m unittest discover -s tests
python -m compileall aicomp_grounding scripts offline cloud
```
