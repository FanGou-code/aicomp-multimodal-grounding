# 🇨🇳 ModelScope DSW 推理部署指南

本目录包含在 **阿里云魔搭社区（ModelScope DSW）** 或国内 GPU 机器上运行离线推理与比赛自动打包的专属工具。

---

## 🚀 极简 3 步使用流程

### 1. 同步代码
在魔搭 DSW 的 Terminal 终端中：
```bash
git pull
```

### 2. 初始化环境（使用阿里云镜像源加速）
```bash
bash deploy/modelscope/setup_dsw.sh
```

### 3. 一键启动推理与打包
将从 Modal 训练导出的 LoRA 权重文件夹放入项目（如 `best/epoch_03/`），然后直接执行：
```bash
# 默认使用 best/epoch_03 作为 LoRA 路径
bash deploy/modelscope/run_dsw.sh

# 或者显式指定 LoRA 路径
bash deploy/modelscope/run_dsw.sh best/epoch_03
```

---

## 📦 输出结果

脚本运行完毕后，会自动在 `outputs/inference/<run_id>/` 下生成：
- `predictions.json`: 原始预测坐标字典
- `submission.zip`: **已严格校验的官方格式比赛提交压缩包（直接用于上传测评平台）**
- `metadata.json`: 完整可追溯的推理配置与指纹信息
