"""
简单的测试文件，验证环境配置是否正确
"""

def test_imports():
    """测试所有必要的包是否能正确导入"""
    try:
        import torch
        import transformers
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.graph_objects as go
        import jsonlines
        print("✅ 所有包导入成功")
        return True
    except ImportError as e:
        print(f"❌ 包导入失败: {e}")
        return False

def test_data_processing():
    """测试数据处理功能"""
    try:
        from data_processor import create_sample_data, DataProcessor
        
        # 创建示例数据
        test_file = "test_data.jsonl"
        create_sample_data(test_file, num_samples=5)
        
        # 测试数据加载
        processor = DataProcessor(test_file)
        data = processor.load_data()
        
        assert len(data) == 5
        assert 'query' in data[0]
        assert 'answer' in data[0]
        assert 'level' in data[0]
        
        # 清理测试文件
        import os
        os.remove(test_file)
        
        print("✅ 数据处理功能正常")
        return True
    except Exception as e:
        print(f"❌ 数据处理测试失败: {e}")
        return False

def test_config():
    """测试配置文件"""
    try:
        from config import MODELS, TARGET_TOKENS, ANALYSIS_CONFIG
        
        assert len(MODELS) == 3
        assert len(TARGET_TOKENS) > 0
        assert 'max_length' in ANALYSIS_CONFIG
        
        print("✅ 配置文件正常")
        return True
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        return False

def test_visualization():
    """测试可视化功能"""
    try:
        from visualizer import LogitsVisualizer
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            viz = LogitsVisualizer(temp_dir)
            assert viz.output_dir == temp_dir
        
        print("✅ 可视化功能正常")
        return True
    except Exception as e:
        print(f"❌ 可视化测试失败: {e}")
        return False

def test_model_loader():
    """测试模型加载器（不实际加载模型）"""
    try:
        from model_loader import LogitsExtractor
        
        # 只测试类定义，不实际加载模型
        # 因为在测试环境中可能没有真实模型
        
        print("✅ 模型加载器定义正常")
        return True
    except Exception as e:
        print(f"❌ 模型加载器测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("🧪 开始运行测试...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_data_processing,
        test_visualization,
        test_model_loader
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！环境配置正确。")
        return True
    else:
        print("❌ 部分测试失败，请检查环境配置。")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
