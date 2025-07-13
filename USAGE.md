# Logits Study 使用指南

## 快速开始

### 1. 环境设置

```bash
# 方法1: 使用setup脚本（推荐）
./setup.sh

# 方法2: 使用Makefile
make setup

# 方法3: 手动设置
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
  "is_correct": true/false
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

#### 2. 程序化运行

```python
from analyzer import LogitsAnalyzer

analyzer = LogitsAnalyzer()
analyzer.run_full_analysis(['baseline', 'sft'])
```

#### 3. Jupyter Notebook

打开 `logits_analysis_demo.ipynb` 进行交互式分析。

## 输出结果

### 1. 可视化图表

- `results/model_comparison_*.png` - 模型对比图
- `results/level_analysis_*.png` - 难度级别分析
- `results/correctness_comparison_*.png` - 正确性对比
- `results/logits_heatmap.png` - Logits热力图

### 2. 交互式图表

- `results/target_token_probabilities.html` - 交互式概率图
- `results/dashboard.html` - 综合仪表板

### 3. 分析结果

- `results/analysis_results.json` - 详细分析数据
- `results/comprehensive_analysis_results.json` - 统计分析结果

## 高级用法

### 自定义分析

```python
from analyzer import LogitsAnalyzer
from config import TARGET_TOKENS

analyzer = LogitsAnalyzer()

# 加载特定模型
analyzer.load_models(['baseline'])

# 加载数据
analyzer.load_data('your_data.jsonl')

# 提取logits
all_logits = analyzer.extract_all_logits()

# 自定义分析
custom_results = analyzer.analyze_target_tokens(all_logits)
```

### 添加新的目标词汇

在 `config.py` 中修改：

```python
TARGET_TOKENS = ['wait', 'aha', 'check', 'your_new_token']
```

### 自定义可视化

```python
from visualizer import LogitsVisualizer

viz = LogitsVisualizer('custom_output_dir')
viz.plot_model_comparison(your_data)
```

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型路径是否正确
   - 确认有足够的GPU内存
   - 尝试使用CPU：修改 `config.py` 中的 `DEVICE = 'cpu'`

2. **内存不足**
   - 减少批次大小：`BATCH_SIZE = 1`
   - 减少序列长度：`MAX_LENGTH = 1024`
   - 使用模型并行：在模型配置中添加 `device_map="auto"`

3. **数据格式错误**
   - 确认JSONL文件格式正确
   - 检查必需字段：query, answer, level
   - 使用 `create_sample_data()` 生成示例数据参考

### 调试模式

```bash
# 使用详细输出
python run_analysis.py --run --verbose

# 使用单个样本测试
python -c "
from analyzer import LogitsAnalyzer
analyzer = LogitsAnalyzer()
analyzer.load_data()
print('数据加载成功')
"
```

### 性能优化

1. **使用缓存**
   - 首次运行后，logits会被缓存
   - 清理缓存：`make clean`

2. **批处理优化**
   - 增加批次大小（如果内存允许）
   - 使用GPU加速

3. **并行处理**
   - 多个模型可以顺序加载以节省内存
   - 考虑使用模型量化

## 扩展功能

### 添加新模型

在 `config.py` 中添加：

```python
MODELS['your_model'] = {
    'name': 'Your Model Name',
    'path': '/path/to/your/model',
    'description': 'Model Description'
}
```

### 自定义分析指标

继承 `LogitsAnalyzer` 类：

```python
class CustomAnalyzer(LogitsAnalyzer):
    def custom_metric(self, logits, tokens):
        # 实现自定义分析
        pass
```

### 集成到其他项目

```python
# 作为库使用
from logits_study.analyzer import LogitsAnalyzer
from logits_study.config import TARGET_TOKENS

# 在您的项目中使用
analyzer = LogitsAnalyzer()
results = analyzer.run_full_analysis(['your_model'])
```

## 引用和参考

如果您在研究中使用此工具，请考虑引用：

```bibtex
@software{logits_study,
  title={Logits Study: LLM Internal Representation Analysis Tool},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/logits_study}
}
```

## 支持和贡献

- 问题报告：请在GitHub Issues中提交
- 功能请求：欢迎提交Pull Request
- 文档改进：帮助完善使用说明

## 许可证

本项目采用MIT许可证，详见LICENSE文件。
