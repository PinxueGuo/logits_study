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
