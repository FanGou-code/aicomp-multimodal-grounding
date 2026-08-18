# 🇨🇳 ModelScope DSW 环境预设

魔搭社区免费 GPU（A10 24G）上运行离线推理的专属环境预设。推理本体是
[`../infer.py`](../infer.py)，本目录只负责装环境和一键启动。

---

## 🚀 极简 3 步使用流程（均在仓库根目录执行）

### 1. 同步代码
在魔搭 DSW 的 Terminal 终端中：
```bash
git pull
```

### 2. 初始化环境（阿里云镜像 + 与云端同版本锁定）
```bash
bash offline/dsw/setup.sh
```

### 3. 一键启动推理与打包
将从云端训练导出的 LoRA 权重文件夹放入项目（如 `best/epoch_03/`），然后：
```bash
# 默认使用 best/epoch_03 作为 LoRA 路径
bash offline/dsw/run.sh

# 或者显式指定 LoRA 路径 / 数据目录 / 基础模型
bash offline/dsw/run.sh best/epoch_03 data Qwen/Qwen3-VL-8B-Instruct
```

> 基础模型在 DSW 上建议从魔搭镜像下载到本地后用 `--model-path` 指定本地目录，
> 避免直连 HuggingFace。

---

## 📦 输出结果

脚本运行完毕后，会自动在 `outputs/inference/<run_id>/` 下生成：
- `predictions.json`: 原始预测坐标字典
- `submission.zip`: **已严格校验的官方格式比赛提交压缩包（直接用于上传测评平台）**
- `metadata.json`: 完整可追溯的推理配置与指纹信息
