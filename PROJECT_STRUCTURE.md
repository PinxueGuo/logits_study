# 项目文件说明

## 核心文件

### 🔧 配置和设置
- `pyproject.toml` - 项目依赖和uv配置
- `config.py` - 模型路径、分析参数配置
- `setup.sh` - 自动化环境设置脚本
- `Makefile` - 常用操作的快捷命令

### 📊 核心分析模块
- `model_loader.py` - 模型加载和logits提取
- `data_processor.py` - 数据加载和预处理
- `analyzer.py` - 主分析流程
- `visualizer.py` - 可视化工具

### 🚀 运行脚本
- `run_analysis.py` - 命令行运行脚本
- `logits_analysis_demo.ipynb` - Jupyter演示notebook

### 📖 文档
- `README.md` - 项目介绍和快速开始
- `USAGE.md` - 详细使用指南
- `PROJECT_STRUCTURE.md` - 本文件，项目结构说明

### 🧪 测试
- `test_environment.py` - 环境验证测试

## 目录结构（运行后生成）

```
logits_study/
├── data/                    # 数据目录
│   └── queries.jsonl       # 输入数据文件
├── results/                 # 分析结果
│   ├── *.png               # 可视化图表
│   ├── *.html              # 交互式图表
│   ├── dashboard.html      # 综合仪表板
│   └── *.json              # 分析结果数据
├── cache/                   # 缓存目录
│   └── *_logits.pkl        # 模型logits缓存
└── ...核心文件...
```

## 使用流程

1. **环境设置**: `./setup.sh` 或 `make setup`
2. **配置模型**: 编辑 `config.py`
3. **准备数据**: 准备JSONL格式数据
4. **运行分析**: `make run` 或 `make jupyter`
5. **查看结果**: 打开 `results/dashboard.html`

## 扩展点

- **新模型**: 在 `config.py` 中添加模型配置
- **新指标**: 在 `analyzer.py` 中添加分析方法
- **新可视化**: 在 `visualizer.py` 中添加图表类型
- **新数据格式**: 在 `data_processor.py` 中添加加载器

## 依赖关系

```
run_analysis.py → analyzer.py → {model_loader.py, data_processor.py, visualizer.py}
                              ↗
config.py ────────────────────┘
```

## 关键设计

- **模块化**: 每个功能独立模块，便于维护
- **缓存机制**: 自动缓存logits以加速重复分析
- **多模型支持**: 统一接口支持不同模型对比
- **可视化丰富**: 静态图表 + 交互式图表
- **统计严谨**: 包含显著性检验和效应量分析
