"""
Logits Study Configuration
"""

import os

# Model configurations
MODELS = {
    'baseline': {
        'name': 'Qwen2.5-Math-7B',
        'path': '/gpfs/models/huggingface.co/Qwen/Qwen2.5-Math-7B',
        'description': 'Baseline Qwen Model'
    },
    'sft': {
        'name': 'DeepSeek-R1-Distill-Qwen-7B',
        'path': '/gpfs/models/huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/',
        'description': 'SFT Model'
    },
    'rl': {
        'name': 'RL-DeepSeek-R1-Distill-Qwen-7B',
        'path': '/gpfs/shared_data/qiji/mllm/models/DeepSeek-R1-Distill-Qwen-7B-xizhihengRL',
        'description': 'RL Trained Model'
    }
}

# Target tokens to analyze - Chinese thinking words that may appear in math reasoning
TARGET_TOKENS = ['等等', '等一下', '检查', '确认', '我知道了', 'wait', 'aha', 'check']

# Data configuration
DATA_CONFIG = {
    'input_file': 'data/logits_study_data_sample4.jsonl',  # Expected format: {'query': str, 'answer': str, 'level': int}
    'output_dir': 'results',
    'cache_dir': 'results/cache'
}

# Analysis configuration
ANALYSIS_CONFIG = {
    'max_length': 32*1024,
    'max_new_tokens': 30*1024,  # Maximum tokens to generate for answers
    'batch_size': 4,
    'device': 'cuda',
    'torch_dtype': 'bfloat16',
    'context_window': 30*1024,  # Tokens before and after target token to analyze
}

# Visualization configuration
VIS_CONFIG = {
    'figsize': (15, 10),
    'dpi': 300,
    'color_palette': 'viridis',
    'save_format': 'png'
}
