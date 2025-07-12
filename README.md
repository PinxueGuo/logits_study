# Logits Study

这是一个用于分析大语言模型（LLM）logits的研究工具，专门针对Qwen架构模型设计。

## 功能特点

- 支持多个模型对比分析（Baseline、SFT、RL模型）
- 分析特定词汇的logits变化（如 'wait', 'aha', 'check' 等推理词汇）
- 多维度比较：
  - 不同模型间的比较
  - 不同难度级别题目的比较
  - 答对与答错题目的比较
- 丰富的可视化功能（热力图、折线图、交互式图表）

## 模型配置

当前配置的模型路径：

1. **Baseline Model**: Qwen2.5-Math-7B
   - 路径: `/gpfs/models/huggingface.co/Qwen/Qwen2.5-Math-7B`

2. **SFT Model**: DeepSeek-R1-Distill-Qwen-7B  
   - 路径: `/gpfs/models/huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/`

3. **RL Model**: RL训练的DeepSeek-R1-Distill-Qwen-7B
   - 路径: `/gpfs/users/xizhiheng/qiji_projects/NorthRL/checkpoints/...`

## 环境设置

项目使用 `uv` 进行包管理：

```bash
# 安装 uv（如果未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 设置环境
uv sync
```

## 数据格式

输入数据应为JSONL格式，每行包含：
```json
{
  "query": "问题内容",
  "answer": "回答内容", 
  "level": 1-5,
}
```

## 使用方法

### 1. 快速开始（完整流程）

```bash
python run_analysis.py --all
```

### 2. 分步骤运行

```bash
# 设置环境
python run_analysis.py --setup

# 创建示例数据（如果没有真实数据）
python run_analysis.py --create-sample

# 运行分析
python run_analysis.py --run
```

### 3. 指定模型和数据

```bash
# 只分析特定模型
python run_analysis.py --run --models baseline,sft

# 使用自定义数据文件
python run_analysis.py --run --data /path/to/your/data.jsonl
```

## 项目结构

```
logits_study/
├── config.py              # 配置文件
├── model_loader.py         # 模型加载和logits提取
├── data_processor.py       # 数据处理工具
├── visualizer.py          # 可视化工具
├── analyzer.py            # 主分析流程
├── run_analysis.py        # 快速运行脚本
├── data/                  # 数据目录
├── results/               # 结果输出目录
├── cache/                 # 缓存目录
└── pyproject.toml         # 项目依赖配置
```

## 输出结果

分析完成后，将在 `results/` 目录下生成：

1. **可视化图表**：
   - `model_comparison_*.png` - 模型对比图
   - `level_analysis_*.png` - 难度级别分析图
   - `correctness_comparison_*.png` - 正确性对比图
   - `logits_heatmap.png` - logits热力图

2. **交互式图表**：
   - `target_token_probabilities.html` - 目标词汇概率交互图

3. **综合仪表板**：
   - `dashboard.html` - 包含所有可视化的综合仪表板

4. **分析结果**：
   - `analysis_results.json` - 详细的分析数据

## 自定义配置

修改 `config.py` 文件来调整：

- 模型路径
- 目标分析词汇
- 分析参数
- 可视化设置

## 目标分析词汇

当前配置分析的推理相关词汇：
- wait, aha, check, think, hmm
- let, actually, however, so, therefore

可在 `config.py` 中的 `TARGET_TOKENS` 列表中修改。

## 注意事项

1. 确保模型路径正确且可访问
2. 如果在本地环境运行，可能需要调整模型路径
3. 大模型加载需要足够的GPU内存
4. 首次运行会缓存结果以加速后续分析

## 依赖项

主要依赖包括：
- torch, transformers - 模型加载和推理
- matplotlib, seaborn, plotly - 可视化
- pandas, numpy - 数据处理
- jsonlines - JSONL文件处理

## 故障排除

如果遇到模型加载问题：
1. 检查模型路径是否正确
2. 确认有足够的GPU内存
3. 可以先用单个模型测试：`--models baseline`

如果遇到内存不足：
1. 减少 `batch_size` 在 `config.py` 中
2. 减少 `max_length` 参数
3. 使用更小的数据集进行测试
