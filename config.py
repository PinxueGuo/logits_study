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
        'path': '/gpfs/users/xizhiheng/qiji_projects/NorthRL/checkpoints/xizhiheng_SkyWork_runs_DeepSeek-R1-Distill-Qwen-7B-Rethink-RL-Loss/xizhiheng___fsdp_valFirst_lr2e-6_kl0-low0.4-high0.2-partial-budget-1-len32k-skywork-grpo-temperature0.6-ppo_epochs2-stale1-testtmp-4-testdynamicclip-targetpos0.5-left---right-1.0-inf-1/global_step_380',
        'description': 'RL Trained Model'
    }
}

# Target tokens to analyze
TARGET_TOKENS = ['wait', 'aha', 'check', 'think', 'hmm', 'let', 'actually', 'however', 'so', 'therefore']

# Data configuration
DATA_CONFIG = {
    'input_file': 'data/logits_study_data.jsonl',  # Expected format: {'query': str, 'answer': str, 'level': int}
    'output_dir': 'results',
    'cache_dir': 'cache'
}

# Analysis configuration
ANALYSIS_CONFIG = {
    'max_length': 32*1024,
    'batch_size': 4,
    'device': 'cuda',
    'torch_dtype': 'bfloat16',
    'context_window': 100,  # Tokens before and after target token to analyze
}

# Visualization configuration
VIS_CONFIG = {
    'figsize': (15, 10),
    'dpi': 300,
    'color_palette': 'viridis',
    'save_format': 'png'
}
