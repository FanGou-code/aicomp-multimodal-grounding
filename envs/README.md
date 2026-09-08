# 环境声明（依赖分层与安装说明）

依赖的**唯一声明源是根 `pyproject.toml`**（`dependencies` + `optional-dependencies`）。
本目录只解释"为什么这么分层"与"按需装什么"，不再维护任何 pin 文件。

## L0~L5 分层归属

GPU 软件栈从上到下分六层，只有 L4/L5 是仓库的；L0~L3 归平台，仓库**不装、不重装、不 pin 死版本**：

| 层 | 内容（AMD 侧为例） | 谁拥有 | 仓库处理 |
|---|---|---|---|
| L0 | 内核驱动 `amdgpu`（6.10.5） | 平台 | 碰不到；与 L1 不对齐时只能反馈平台换镜像 |
| L1 | 用户态 ROCm runtime（7.2.3） | 平台 | 同上 |
| L2 | 计算库 rocBLAS/hipBLAS/hipBLASLt | 平台随镜像 | 不装 |
| L3 | torch（ROCm 版 2.11 / CUDA 版 2.10） | 平台预装 | `--system-site-packages` 透传复用，**范围声明**（`>=2.8,<3`）让 pip 判定已满足而跳过 |
| L4 | transformers / peft / accelerate / qwen-vl-utils | **仓库** | `==` 紧 pin，跨平台一致 |
| L5 | 代码 / 适配器 | **仓库** | 随仓库自带 |

结论：`python -m venv --system-site-packages` 是唯一正确姿势——让 venv 看见并复用平台 L0~L3，仓库只声明 L4。**不要**建独立完整 venv，否则会迫使 pip 自己装 torch，拉到与平台 L1/L2 不匹配的 wheel。

## 安装（三平台同一条命令）

```bash
pip install -e .            # 通用：含 torch（范围）+ 四件套 + numpy/pillow/opencv
pip install -e ".[dev]"     # 开发/CI 再加 ruff
pip install -e ".[kernels]" # 仅 Qwen3.5-9B(GDN) 接入时按需
```

torch 是**范围**（`>=2.8,<3`），平台已有版本落在范围内即被 pip 判定已满足、自动跳过：
DSW A 卡 torch 2.11、DSW N 卡 2.10、Modal（空容器装最新 CUDA 版）、实验室 N 卡（装最新 CUDA 版）——同一份声明，平台口味在安装时决定。

验证（10 秒，激活 venv 后）：

```bash
python -c "import transformers, peft; print(transformers.__version__, transformers.__file__)"
# 应输出 5.14.1，路径落在 venv 内
```

## 加速内核（仅混合线性注意力架构，按需）

FLA + causal-conv1d 只为 **GDN 混合线性注意力**（计划接入的 Qwen3.5-9B）服务；
现有标准注意力模型（qwen3vl / internvl35 / groundingdino / glm46v）用不上，装了也不生效。

- **N 卡/CUDA 机**：`pip install -e ".[kernels]"` 一键（两者均有预编译轮）。
- **A 卡（DSW）**：FLA 有预编译，causal-conv1d 无 HIP 预编译轮，必须源码编译：

```bash
export PYTORCH_ROCM_ARCH=gfx942 MAX_JOBS=8
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
pip install causal-conv1d --no-build-isolation \
  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
# 编不过可直接跳过；FLA 是主要收益来源
```

注意：causal-conv1d 源码编译产物不进 pip 缓存，重建 venv 后需重编。
