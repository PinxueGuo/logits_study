#!/bin/bash

# Logits Study Setup Script
# 使用uv管理Python环境和依赖

set -e  # 遇到错误时退出

echo "🚀 开始设置Logits Study研究环境..."

# 检查uv是否安装
if ! command -v uv &> /dev/null; then
    echo "❌ uv未安装，正在安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
    echo "✅ uv安装完成"
else
    echo "✅ uv已安装"
fi

# 创建虚拟环境和安装依赖
echo "📦 创建虚拟环境并安装依赖..."
uv sync

# 检查CUDA可用性
echo "🔍 检查CUDA环境..."
uv run python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# 验证安装
echo "🧪 验证安装..."
uv run python -c "
import torch
import transformers
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
import numpy as np
print('✅ 所有依赖包安装成功')
print(f'PyTorch版本: {torch.__version__}')
print(f'Transformers版本: {transformers.__version__}')
"

echo "🎉 环境设置完成!"
echo ""
echo "📋 接下来的步骤:"
echo "1. 检查并修改 config.py 中的模型路径"
echo "2. 准备您的JSONL数据文件，或使用示例数据"
echo "3. 运行分析:"
echo "   uv run python run_analysis.py --run"
echo "   或者"
echo "   uv run jupyter lab logits_analysis_demo.ipynb"
echo ""
echo "📖 查看 README.md 获取详细使用说明"
