# Task 02: 全对称通用文本标准化清洗模块

## 1. 任务目标
实现在训练端与推理端全对称调用的 `standardize_query()` 公共清洗模块：
1. **解决测试集 15.03%（1,436 条）标点噪音**：剥离末尾句号（`.`）、问号、多余空白与异常特殊字符。
2. **解决训练集 2.37%（68 条）首字母小写瑕疵**：统一规范首字母大写（Sentence Case）。
3. **保证分词绝对一致（Zero-Shift）**：确保训练与测试送入 Qwen3-VL Tokenizer 的 Token ID 路径 100% 对齐。

---

## 2. 目标文件
- **核心定义文件**：[`aicomp_grounding/prompts.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/aicomp_grounding/prompts.py)
- **集成调用文件**：
  - [`train_modal.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/train_modal.py)
  - [`run_inference.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/run_inference.py)

---

## 3. 具体修改规范

### 3.1 在 `prompts.py` 中新增 `standardize_query`
在 [`aicomp_grounding/prompts.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/aicomp_grounding/prompts.py) 中导出该函数：

```python
def standardize_query(query: str) -> str:
    """
    对视觉定位 Query 进行文本标准化清洗（训练与推理两端全对称使用）。
    
    清洗规则：
    1. 剥离首尾空白字符；
    2. 剥离末尾多余的标点符号 (. ? ! : ;)；
    3. 压缩内部连续多余空格/制表符为单个空格；
    4. 统一中英文特殊引号（“”’’）为 ASCII 标准半角符号；
    5. 规范首字母大写（Sentence Case）。
    """
    if not query:
        return ""
    q = query.strip()
    # 剥离末尾句号等标点
    q = q.rstrip(".?!:;").strip()
    # 压缩连续空格
    q = " ".join(q.split())
    # 规范单双引号
    q = q.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
    # 首字母大写
    if q and q[0].islower():
        q = q[0].upper() + q[1:]
    return q
```

### 3.2 训练端集成 ([`train_modal.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/train_modal.py))
在 `RGBDTGroundingDataset.__getitem__` 中：
```python
raw_query = self.records[index].get("query", "")
clean_query = standardize_query(raw_query)
# 使用 clean_query 构建 prompt
```

### 3.3 推理端集成 ([`run_inference.py`](file:///home/fang0/dev/projects/aicomp-multimodal-grounding/run_inference.py))
在循环读取测试样本构建输入时：
```python
raw_query = item.get("query", "")
clean_query = standardize_query(raw_query)
messages = build_grounding_messages(vis_img, ir_img, depth_img, clean_query)
```
*注：最终写入提交文件时仍需保留官方原始 `raw_query` 字段，确保赛规合规。*

---

## 4. 验证命令与验收标准
执行以下脚本，验证全量 9,555 条测试集与 2,875 条训练集的清洗准确度：

```bash
/home/fang0/miniconda3/envs/qwen_vg/bin/python -c "
import json
from pathlib import Path
from aicomp_grounding.prompts import standardize_query

# 1. 基础单测
assert standardize_query('  the brown bear.  ') == 'The brown bear'
assert standardize_query('a red car?') == 'A red car'
assert standardize_query('“The drone”') == '\"The drone\"'

# 2. 全量测试集清洗测试
with open('data/test.json', 'r', encoding='utf-8') as f:
    test_raw = json.load(f)

test_queries = [v['query'] for v in test_raw.values() if 'query' in v]
cleaned_test = [standardize_query(q) for q in test_queries]

assert all(not q.endswith('.') for q in cleaned_test), 'Still contains trailing dot!'
assert all(q[0].isupper() for q in cleaned_test if q), 'Not all capitalized!'
print(f'Task 02: Cleaned {len(test_queries)} queries successfully without errors!')
"
```
