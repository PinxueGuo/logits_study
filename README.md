# Logits Study 使用指南

## 快速开始

### 1. 环境设置

```bash
./setup.sh
uv sync
```

## 详细使用说明

### 配置模型路径

编辑 `config.py` 文件，修改模型路径：

```python
MODELS = {
    'baseline': {
        'path': '/path/to/your/baseline/model',
        # ...
    },
    # ...
}
```

### 准备数据

数据格式为JSONL，每行包含：
```json
{
  "query": "问题内容",
  "answer": "回答内容",
  "level": 1-5,
}
```

### 运行选项

#### 1. 命令行运行

```bash
# 完整流程
python run_analysis.py --all

# 仅运行分析
python run_analysis.py --run

# 指定模型
python run_analysis.py --run --models baseline,sft

# 指定数据文件
python run_analysis.py --run --data /path/to/data.jsonl
```

## 高级用法

### 添加新的目标词汇

在 `config.py` 中修改：

```python
TARGET_TOKENS = ['wait', 'aha', 'check', 'your_new_token']
```
