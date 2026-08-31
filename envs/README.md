# 环境声明（单一事实源）

三层分工：**镜像层**（torch + CUDA/ROCm + 驱动）归平台，永不进 pip 声明；
**venv 层**（本目录的 pin）归仓库；**代码层**随仓库自带。

- `gpu.txt` — GPU 环境（魔搭 DSW / Modal / 实验室 N 卡通用）。
  魔搭 venv 与 Modal Image 均从此文件安装；transformers==5.14.1 与
  DSW 镜像自带版本一致，声明与现实对齐。
- `local.txt` — 本地 CPU 测试环境（由根 requirements-lock.txt 承担，暂不重复维护）。

安装（GPU 实例）：

```bash
python3 -m venv --system-site-packages /mnt/workspace/aicomp_env
source /mnt/workspace/aicomp_env/bin/activate
pip install -r envs/gpu.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

验证（10 秒）：

```bash
python -c "import transformers, peft; print(transformers.__version__, transformers.__file__)"
# 版本应为 5.14.1，路径落在 venv 内（未被 --ignore-installed 跳过时）
```

若编译 C++ 扩展类加速库（如 causal-conv1d），先设置：

```bash
export PYTORCH_ROCM_ARCH=gfx942   # 只编 MI300X，砍掉多架构编译量
export MAX_JOBS=8
export TRITON_CACHE_DIR=/mnt/workspace/.triton_cache   # Triton JIT 缓存持久化
```

A 卡加速内核（按需，为混合线性注意力架构的模型准备；对现有
标准注意力模型完全惰性，装上不生效也不破坏）：

```bash
pip install flash-linear-attention \
  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
pip install causal-conv1d --no-build-isolation \
  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
# 编不过 causal-conv1d 可直接跳过；FLA 是主要收益来源
python -c "import fla; print('fla OK')"
```

注意：每个新实例重建 venv 后需重装本节内容（包缓存持久，重装为秒级）。
