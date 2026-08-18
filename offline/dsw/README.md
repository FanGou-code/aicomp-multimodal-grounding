# 🇨🇳 ModelScope DSW 环境预设

DSW = **Data Science Workshop**（数据科学工作台），阿里云 PAI 的交互式建模组件；
魔搭社区的免费 GPU 实例即跑在 DSW 上。

本目录只包含**真正 DSW 特定的东西**：环境安装脚本。启动器 `../run.sh` 与推理本体
`../infer.py` 是通用的，供所有离线平台共用——平台的差异只体现在"怎么装环境"。

---

## 🚀 使用流程（均在仓库根目录执行）

### 1. 同步代码
```bash
git pull
```

### 2. 初始化环境（阿里云镜像 + 与云端同版本锁定，基于 DSW 预装 torch）
```bash
bash offline/dsw/setup.sh
```

### 3. 一键启动推理与打包（通用启动器）
将从云端训练导出的 LoRA 权重文件夹放入项目（如 `best/epoch_03/`），然后：
```bash
bash offline/run.sh                          # 默认 best/epoch_03
bash offline/run.sh best/epoch_03 data Qwen/Qwen3-VL-8B-Instruct   # 显式指定
```

> 基础模型在 DSW 上建议从魔搭镜像下载到本地后用 MODEL_PATH 指定本地目录，
> 避免直连 HuggingFace。
