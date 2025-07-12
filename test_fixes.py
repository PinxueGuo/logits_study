#!/usr/bin/env python3
"""
Test script to verify the fixes
"""

import os
import sys
from pathlib import Path

def test_imports():
    """测试主要的导入是否正常"""
    try:
        from config import MODELS, TARGET_TOKENS, DATA_CONFIG, ANALYSIS_CONFIG
        print("✓ Config imports successful")
        
        from data_processor import DataProcessor
        print("✓ DataProcessor import successful")
        
        from visualizer import LogitsVisualizer
        print("✓ LogitsVisualizer import successful")
        
        from analyzer import LogitsAnalyzer
        print("✓ LogitsAnalyzer import successful")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_processor():
    """测试DataProcessor的create_full_text方法"""
    try:
        from data_processor import DataProcessor
        
        # 创建测试数据
        test_item = {
            'query': 'What is 2+2?',
            'answer': 'Let me think step by step. 2+2=4. \\boxed{4}',
            'level': 1
        }
        
        # 创建临时的DataProcessor实例
        processor = DataProcessor("dummy_file.jsonl")
        
        # 测试带system prompt的文本生成
        full_text_with_prompt = processor.create_full_text(test_item, include_system_prompt=True)
        full_text_without_prompt = processor.create_full_text(test_item, include_system_prompt=False)
        
        assert "Please reason step by step" in full_text_with_prompt
        assert "Please reason step by step" not in full_text_without_prompt
        
        print("✓ DataProcessor create_full_text method works correctly")
        print(f"  With prompt: {full_text_with_prompt[:100]}...")
        print(f"  Without prompt: {full_text_without_prompt[:100]}...")
        
        return True
    except Exception as e:
        print(f"✗ DataProcessor test error: {e}")
        return False

def test_visualizer():
    """测试新的可视化类"""
    try:
        from visualizer import LogitsVisualizer
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            viz = LogitsVisualizer(temp_dir)
            print("✓ LogitsVisualizer initialization successful")
            
            # 检查是否只有我们需要的方法
            methods = [method for method in dir(viz) if not method.startswith('_')]
            expected_methods = ['plot_logits_heatmap', 'create_all_heatmaps']
            
            for method in expected_methods:
                if hasattr(viz, method):
                    print(f"✓ Method {method} exists")
                else:
                    print(f"✗ Method {method} missing")
                    return False
            
            return True
    except Exception as e:
        print(f"✗ Visualizer test error: {e}")
        return False

def test_config():
    """测试配置是否正确"""
    try:
        from config import ANALYSIS_CONFIG, TARGET_TOKENS
        
        # 检查bfloat16配置
        if ANALYSIS_CONFIG['torch_dtype'] == 'bfloat16':
            print("✓ Config has bfloat16 setting")
        else:
            print(f"? Config torch_dtype is {ANALYSIS_CONFIG['torch_dtype']}")
        
        # 检查目标tokens
        print(f"✓ Target tokens: {TARGET_TOKENS}")
        
        return True
    except Exception as e:
        print(f"✗ Config test error: {e}")
        return False

def main():
    """运行所有测试"""
    print("=== Testing Logits Study Fixes ===\n")
    
    tests = [
        ("Imports", test_imports),
        ("Data Processor", test_data_processor),
        ("Visualizer", test_visualizer),
        ("Config", test_config),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- Testing {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    print("\n=== Test Summary ===")
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n✓ All tests passed! The fixes should work correctly.")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
