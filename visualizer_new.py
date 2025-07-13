"""
Visualization utilities for entropy analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import os
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

class LogitsVisualizer:
    def __init__(self, output_dir: str, figsize: Tuple[int, int] = (15, 8), dpi: int = 300):
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
        sns.set_palette("viridis")
    
    def create_entropy_heatmaps(self, entropy_results: Dict[str, Any]):
        """
        Create entropy heatmaps for each query
        
        Args:
            entropy_results: Results from entropy analysis
        """
        print(f"Creating entropy heatmaps for {len(entropy_results['query_entropy_data'])} queries...")
        
        for query_data in entropy_results['query_entropy_data']:
            self._create_single_query_heatmap(query_data)
        
        # Create summary heatmap
        self._create_summary_heatmap(entropy_results)
    
    def _create_single_query_heatmap(self, query_data: Dict[str, Any]):
        """
        Create heatmap for a single query
        
        Args:
            query_data: Query data containing entropy information for all models
        """
        query_idx = query_data['query_idx']
        query_text = query_data['query_text']
        models = list(query_data['models'].keys())
        
        if not models:
            return
        
        # Find the maximum sequence length for normalization
        max_length = max(query_data['models'][model]['sequence_length'] for model in models)
        
        # Prepare data for heatmap
        heatmap_data = []
        model_labels = []
        
        for model_name in models:
            model_data = query_data['models'][model_name]
            entropy = model_data['entropy']
            seq_length = len(entropy)
            
            # Normalize positions to 0-1 range
            if seq_length > 1:
                normalized_positions = np.linspace(0, 1, seq_length)
            else:
                normalized_positions = np.array([0.5])
            
            # Interpolate to fixed grid for visualization
            grid_size = 100  # Fixed grid size for visualization
            grid_positions = np.linspace(0, 1, grid_size)
            
            if seq_length > 1:
                interpolated_entropy = np.interp(grid_positions, normalized_positions, entropy)
            else:
                interpolated_entropy = np.full(grid_size, entropy[0])
            
            heatmap_data.append(interpolated_entropy)
            model_labels.append(model_name)
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Convert to numpy array for plotting
        heatmap_array = np.array(heatmap_data)
        
        # Create heatmap
        im = ax.imshow(heatmap_array, aspect='auto', cmap='viridis', 
                      extent=[0, 1, len(models)-0.5, -0.5])
        
        # Set labels
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(model_labels)
        ax.set_xlabel('Normalized Token Position (0-1)')
        ax.set_ylabel('Models')
        ax.set_title(f'Token-wise Entropy - Query {query_idx}\\n{query_text[:100]}...', 
                    fontsize=12, pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Entropy', rotation=270, labelpad=15)
        
        # Mark target token positions
        self._mark_target_tokens(ax, query_data, grid_size)
        
        # Save the plot
        filename = f"entropy_heatmap_query_{query_idx}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved entropy heatmap for query {query_idx}: {filename}")
    
    def _mark_target_tokens(self, ax, query_data: Dict[str, Any], grid_size: int):
        """
        Mark target token positions on the heatmap
        
        Args:
            ax: Matplotlib axis
            query_data: Query data
            grid_size: Size of the interpolation grid
        """
        models = list(query_data['models'].keys())
        
        for model_idx, model_name in enumerate(models):
            model_data = query_data['models'][model_name]
            target_positions = model_data['target_positions']
            seq_length = model_data['sequence_length']
            
            for target_token, positions in target_positions.items():
                for pos in positions:
                    # Convert position to normalized coordinate
                    if seq_length > 1:
                        normalized_pos = pos / (seq_length - 1)
                    else:
                        normalized_pos = 0.5
                    
                    # Add marker
                    ax.plot(normalized_pos, model_idx, 'r*', markersize=8, 
                           markeredgecolor='white', markeredgewidth=1,
                           label=target_token if model_idx == 0 and pos == positions[0] else "")
        
        # Add legend for target tokens (only if there are any)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 1.0))
    
    def _create_summary_heatmap(self, entropy_results: Dict[str, Any]):
        """
        Create a summary heatmap showing average entropy statistics
        
        Args:
            entropy_results: Complete entropy analysis results
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
        
        # Summary statistics heatmap
        summary_stats = entropy_results['summary_stats']
        models = list(summary_stats.keys())
        
        if not models:
            return
        
        # Prepare data for summary statistics
        stats_data = []
        stats_labels = ['Mean Entropy', 'Std Entropy', 'Min Entropy', 'Max Entropy']
        
        for model in models:
            stats = summary_stats[model]
            stats_data.append([
                stats['mean_entropy'],
                stats['std_entropy'], 
                stats['min_entropy'],
                stats['max_entropy']
            ])
        
        # Plot summary statistics
        ax1 = axes[0, 0]
        stats_array = np.array(stats_data)
        im1 = ax1.imshow(stats_array, aspect='auto', cmap='viridis')
        ax1.set_xticks(range(len(stats_labels)))
        ax1.set_xticklabels(stats_labels, rotation=45, ha='right')
        ax1.set_yticks(range(len(models)))
        ax1.set_yticklabels(models)
        ax1.set_title('Summary Statistics Across All Queries')
        plt.colorbar(im1, ax=ax1)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(stats_labels)):
                text = ax1.text(j, i, f'{stats_array[i, j]:.3f}',
                               ha="center", va="center", color="white", fontweight='bold')
        
        # Query-level average entropy comparison
        ax2 = axes[0, 1]
        query_entropies = {model: [] for model in models}
        
        for query_data in entropy_results['query_entropy_data']:
            for model_name, model_data in query_data['models'].items():
                avg_entropy = np.mean(model_data['entropy'])
                query_entropies[model_name].append(avg_entropy)
        
        # Box plot of query-level entropies
        entropy_values = [query_entropies[model] for model in models]
        box_plot = ax2.boxplot(entropy_values, labels=models, patch_artist=True)
        ax2.set_title('Distribution of Average Entropy per Query')
        ax2.set_ylabel('Average Entropy')
        ax2.tick_params(axis='x', rotation=45)
        
        # Color the boxes
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        # Level-wise analysis if available
        ax3 = axes[1, 0]
        level_entropies = {model: {} for model in models}
        
        for query_data in entropy_results['query_entropy_data']:
            level = query_data['metadata'].get('level', 'Unknown')
            for model_name, model_data in query_data['models'].items():
                if level not in level_entropies[model_name]:
                    level_entropies[model_name][level] = []
                avg_entropy = np.mean(model_data['entropy'])
                level_entropies[model_name][level].append(avg_entropy)
        
        # Prepare level comparison data
        levels = sorted(set(query_data['metadata'].get('level', 'Unknown') 
                          for query_data in entropy_results['query_entropy_data']))
        
        if len(levels) > 1 and 'Unknown' not in levels:
            level_means = []
            for model in models:
                model_level_means = []
                for level in levels:
                    if level in level_entropies[model] and level_entropies[model][level]:
                        model_level_means.append(np.mean(level_entropies[model][level]))
                    else:
                        model_level_means.append(0)
                level_means.append(model_level_means)
            
            level_array = np.array(level_means)
            im3 = ax3.imshow(level_array, aspect='auto', cmap='viridis')
            ax3.set_xticks(range(len(levels)))
            ax3.set_xticklabels([f'Level {l}' for l in levels])
            ax3.set_yticks(range(len(models)))
            ax3.set_yticklabels(models)
            ax3.set_title('Average Entropy by Difficulty Level')
            plt.colorbar(im3, ax=ax3)
            
            # Add text annotations
            for i in range(len(models)):
                for j in range(len(levels)):
                    text = ax3.text(j, i, f'{level_array[i, j]:.3f}',
                                   ha="center", va="center", color="white", fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Level analysis not available\\n(insufficient level data)', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Level Analysis')
        
        # Number of target tokens found per model
        ax4 = axes[1, 1]
        target_token_counts = {model: 0 for model in models}
        
        for query_data in entropy_results['query_entropy_data']:
            for model_name, model_data in query_data['models'].items():
                count = sum(len(positions) for positions in model_data['target_positions'].values())
                target_token_counts[model_name] += count
        
        bars = ax4.bar(models, [target_token_counts[model] for model in models], 
                      color=plt.cm.viridis(np.linspace(0, 1, len(models))))
        ax4.set_title('Total Target Tokens Found')
        ax4.set_ylabel('Count')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save summary plot
        filename = "entropy_analysis_summary.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved entropy analysis summary: {filename}")
