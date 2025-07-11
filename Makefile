# Logits Study Makefile
# 简化常用操作

.PHONY: setup install run clean test jupyter help

# 默认目标
help:
	@echo "Logits Study - 可用命令:"
	@echo "  setup     - 设置环境和安装依赖"
	@echo "  install   - 仅安装依赖"
	@echo "  run       - 运行完整分析"
	@echo "  jupyter   - 启动Jupyter notebook"
	@echo "  test      - 运行测试"
	@echo "  clean     - 清理缓存文件"
	@echo "  demo      - 运行演示分析"
	@echo ""
	@echo "使用方法: make <command>"

# 设置环境
setup:
	@echo "🚀 设置Logits Study环境..."
	./setup.sh

# 安装依赖
install:
	@echo "📦 安装依赖..."
	uv sync

# 运行完整分析
run:
	@echo "🔬 运行logits分析..."
	uv run python run_analysis.py --run

# 运行演示分析（使用示例数据）
demo:
	@echo "🎯 运行演示分析..."
	uv run python run_analysis.py --all

# 启动Jupyter
jupyter:
	@echo "📓 启动Jupyter Lab..."
	uv run jupyter lab logits_analysis_demo.ipynb

# 运行测试
test:
	@echo "🧪 运行环境测试..."
	uv run python test_environment.py

# 清理缓存
clean:
	@echo "🧹 清理缓存文件..."
	rm -rf cache/*
	rm -rf results/*
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# 检查环境
check:
	@echo "🔍 检查环境..."
	uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
	uv run python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 创建示例数据
sample-data:
	@echo "📝 创建示例数据..."
	uv run python -c "from data_processor import create_sample_data; create_sample_data('data/queries.jsonl', 50)"

# 显示项目信息
info:
	@echo "📊 Logits Study项目信息:"
	@echo "  项目目录: $(PWD)"
	@echo "  Python版本: $(shell uv run python --version)"
	@echo "  数据文件: $(shell ls -la data/ 2>/dev/null | wc -l) 个文件"
	@echo "  结果文件: $(shell ls -la results/ 2>/dev/null | wc -l) 个文件"
	@echo "  缓存文件: $(shell ls -la cache/ 2>/dev/null | wc -l) 个文件"
