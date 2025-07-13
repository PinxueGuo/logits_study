"""
Visualization utilities for logits analysis - focused on heatmaps only
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import os

class LogitsVisualizer:
    def __init__(self, output_dir: str, figsize: Tuple[int, int] = (15, 10), dpi: int = 300):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save plots
            figsize: Figure size for matplotlib plots
            dpi: DPI for saved plots
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_logits_heatmap(self, logits: np.ndarray, tokens: List[str], 
                           target_positions: Dict[str, List[int]], 
                           answer_length: int, total_length: int,
                           model_name: str, level: int, is_correct: bool,
                           target_tokens: List[str]):
        """
        Create heatmap showing logits vs normalized answer position
        
        Args:
            logits: Logits array (seq_len, vocab_size)
            tokens: List of token strings
            target_positions: Dictionary of target token positions
            answer_length: Length of the answer portion
            total_length: Total length of the sequence
            model_name: Name of the model
            level: Difficulty level
            is_correct: Whether the answer is correct
            target_tokens: List of target tokens to analyze
        """
        # Calculate normalized positions (0-1 for answer portion)
        answer_start = total_length - answer_length
        normalized_positions = []
        logits_values = []
        token_labels = []
        
        # Extract logits for target tokens in answer portion
        for i in range(answer_start, total_length):
            if i < len(logits):
                normalized_pos = (i - answer_start) / max(1, answer_length - 1)
                
                # Check if this position contains a target token
                current_token = tokens[i] if i < len(tokens) else ""
                cleaned_token = current_token.replace('Ġ', '').replace('▁', '').lower().strip()
                
                for target_token in target_tokens:
                    if (cleaned_token == target_token.lower() or 
                        current_token.lower() == target_token.lower()):
                        
                        # Get logit values for this target token
                        logit_values = logits[i]  # All vocab logits at this position
                        
                        normalized_positions.append(normalized_pos)
                        logits_values.append(logit_values)
                        token_labels.append(target_token)
        
        if not normalized_positions:
            print(f"No target tokens found in answer for {model_name}, level {level}, correct: {is_correct}")
            return
        
        # Create heatmap data
        # Use top-k most variable tokens across all positions
        all_logits = np.array(logits_values)  # Shape: (n_positions, vocab_size)
        
        if len(all_logits) == 0:
            return
            
        # Select top tokens by variance
        top_k = min(100, all_logits.shape[1])
        token_variances = np.var(all_logits, axis=0)
        top_indices = np.argsort(token_variances)[-top_k:]
        
        heatmap_data = all_logits[:, top_indices].T  # Shape: (top_k, n_positions)
        
        # Create the plot
        plt.figure(figsize=self.figsize)
        
        # Create heatmap
        sns.heatmap(heatmap_data, 
                   xticklabels=[f"{pos:.2f}" for pos in normalized_positions],
                   yticklabels=[f"Vocab_{idx}" for idx in top_indices],
                   cmap='viridis',
                   cbar_kws={'label': 'Logit Value'})
        
        # Mark target token positions
        for i, (pos, token) in enumerate(zip(normalized_positions, token_labels)):
            plt.axvline(x=i, color='red', linestyle='--', alpha=0.8, linewidth=2)
            plt.text(i, len(top_indices)//2, token, rotation=90, 
                    verticalalignment='center', color='white', 
                    fontweight='bold', fontsize=10)
        
        # Set labels and title
        plt.xlabel('Normalized Answer Position (0=start, 1=end)')
        plt.ylabel('Vocabulary Tokens (by variance)')
        
        correctness = "Correct" if is_correct else "Incorrect"
        plt.title(f'Logits Heatmap: {model_name} | Level {level} | {correctness}')
        
        plt.tight_layout()
        
        # Save with descriptive filename
        filename = f"heatmap_{model_name}_level{level}_{correctness.lower()}.png"
        plt.savefig(f"{self.output_dir}/{filename}", dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved heatmap: {filename}")
    
    def create_all_heatmaps(self, results_data: Dict[str, Any], target_tokens: List[str]):
        """
        Create heatmaps for all model/level/correctness combinations
        
        Args:
            results_data: Analysis results containing logits and metadata
            target_tokens: List of target tokens to analyze
        """
        print("Creating heatmaps for all combinations...")
        
        for model_name, model_results in results_data.items():
            for logits, tokens, metadata in model_results:
                # Calculate answer length and positions
                answer_text = metadata.get('answer', '')
                total_text = f"Query: {metadata.get('query', '')}\nAnswer: {answer_text}"
                
                # Rough estimation of answer portion
                answer_start_approx = len(total_text) - len(answer_text)
                answer_length = len(answer_text)
                total_length = len(tokens)
                
                # Find target token positions
                target_positions = {}
                for target_token in target_tokens:
                    positions = []
                    for i, token in enumerate(tokens):
                        cleaned_token = token.replace('Ġ', '').replace('▁', '').lower().strip()
                        if (cleaned_token == target_token.lower() or 
                            token.lower() == target_token.lower()):
                            positions.append(i)
                    target_positions[target_token] = positions
                
                # Get metadata
                level = metadata.get('level', 1)
                is_correct = metadata.get('is_correct', False)
                
                # Create heatmap
                self.plot_logits_heatmap(
                    logits=logits,
                    tokens=tokens,
                    target_positions=target_positions,
                    answer_length=answer_length,
                    total_length=total_length,
                    model_name=model_name,
                    level=level,
                    is_correct=is_correct,
                    target_tokens=target_tokens
                )
    
    def create_entropy_comparison_heatmap(self, query: str, models_data: Dict[str, Any], 
                                        query_idx: int, target_tokens: List[str]):
        """
        为单个query创建模型间entropy对比热力图
        
        Args:
            query: 查询文本
            models_data: 各模型的数据 {model_name: {entropy, tokens, target_positions, metadata}}
            query_idx: 查询索引
            target_tokens: 目标tokens列表
        """
        if not models_data:
            return
        
        # 准备数据
        model_names = list(models_data.keys())
        max_length = max(len(data['entropy']) for data in models_data.values())
        
        # 创建归一化位置数组 (0-1)
        normalized_positions = np.linspace(0, 1, max_length)
        
        # 准备热力图数据：每行是一个模型的entropy
        heatmap_data = []
        model_labels = []
        target_markers = []  # 存储标志词位置
        
        for model_name in model_names:
            data = models_data[model_name]
            entropy = data['entropy']
            
            # 插值到统一长度
            if len(entropy) != max_length:
                indices = np.linspace(0, len(entropy)-1, max_length)
                entropy_interp = np.interp(indices, np.arange(len(entropy)), entropy)
            else:
                entropy_interp = entropy
            
            heatmap_data.append(entropy_interp)
            model_labels.append(model_name)
            
            # 收集标志词位置
            model_markers = []
            for token, positions in data['target_positions'].items():
                for pos in positions:
                    if pos < len(entropy):
                        # 转换为归一化位置
                        norm_pos = pos / (len(entropy) - 1) if len(entropy) > 1 else 0
                        model_markers.append((norm_pos, token))
            target_markers.append(model_markers)
        
        # 创建图形
        plt.figure(figsize=(15, 8))
        
        # 创建热力图
        heatmap_data = np.array(heatmap_data)
        
        # 使用自定义colormap强调差异
        im = plt.imshow(heatmap_data, aspect='auto', cmap='viridis', 
                       extent=[0, 1, len(model_names)-0.5, -0.5])
        
        # 添加颜色条
        cbar = plt.colorbar(im, label='Entropy')
        
        # 标记标志词位置
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'gray', 'cyan']
        token_color_map = {token: colors[i % len(colors)] for i, token in enumerate(target_tokens)}
        
        for model_idx, markers in enumerate(target_markers):
            for norm_pos, token in markers:
                x_pos = norm_pos
                y_pos = model_idx
                
                # 画标记点
                plt.scatter(x_pos, y_pos, color=token_color_map[token], 
                          s=100, marker='|', linewidth=3, alpha=0.8)
                
                # 添加文本标签
                plt.text(x_pos, y_pos - 0.15, token, fontsize=8, 
                        color=token_color_map[token], ha='center', va='top',
                        fontweight='bold')
        
        # 设置标签和标题
        plt.yticks(range(len(model_names)), model_names)
        plt.xlabel('Normalized Answer Position (0=start, 1=end)')
        plt.ylabel('Models')
        
        # 截断query文本用于标题
        short_query = query[:60] + "..." if len(query) > 60 else query
        plt.title(f'Entropy Comparison - Query {query_idx + 1}: {short_query}')
        
        # 添加图例
        legend_elements = [plt.Line2D([0], [0], marker='|', color='w', 
                                    markerfacecolor=token_color_map[token], 
                                    markersize=10, label=token, linewidth=0)
                         for token in target_tokens if token in token_color_map]
        
        if legend_elements:
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        
        # 保存文件
        filename = f"entropy_comparison_query_{query_idx + 1}.png"
        plt.savefig(f"{self.output_dir}/{filename}", dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved entropy heatmap: {filename}")
