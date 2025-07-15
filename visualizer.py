"""
Visualization utilities for entropy analysis
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import os

class LogitsVisualizer:
    def __init__(self, output_dir: str, figsize: Tuple[int, int] = (1200, 600), dpi: int = 300):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save plots
            figsize: Figure size for plotly plots (width, height in pixels)
            dpi: DPI for saved PNG plots
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Define Chinese font family for all plots
        self.font_family = "Microsoft YaHei, SimHei, Arial, sans-serif"
    
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
        Create heatmap for a single query using plotly
        
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
            print(model_name)
            model_data = query_data['models'][model_name]
            entropy = model_data['entropy']
            seq_length = len(entropy)
            
            # Normalize positions to 0-1 range
            if seq_length > 1:
                normalized_positions = np.linspace(0, 1, seq_length)
            else:
                normalized_positions = np.array([0.5])
            
            # Interpolate to fixed grid for visualization
            grid_size = 1000  # Fixed grid size for visualization
            grid_positions = np.linspace(0, 1, grid_size)
            
            if seq_length > 1:
                interpolated_entropy = np.interp(grid_positions, normalized_positions, entropy)
            else:
                interpolated_entropy = np.full(grid_size, entropy[0])
            
            heatmap_data.append(interpolated_entropy)
            model_labels.append(model_name)
        
        # Create the heatmap using plotly
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=np.linspace(0, 1, 1000),
            y=model_labels,
            colorscale='viridis',
            colorbar=dict(
                title=dict(
                    text="Entropy",
                    font=dict(family=self.font_family)
                ),
                # x=1.02  # Move colorbar to the right to avoid overlap
            )
        ))
        
        # Add target token markers
        self._add_target_token_markers_plotly(fig, query_data, 1000)
        
        # Update layout with Chinese font support
        fig.update_layout(
            title=dict(
                text=f'查询 {query_idx}: {query_text[:30]}...',
                font=dict(family=self.font_family)
            ),
            xaxis=dict(
                title=dict(
                    text="归一化标记位置 (0-1)",
                    font=dict(family=self.font_family)
                ),
                tickfont=dict(family=self.font_family)
            ),
            yaxis=dict(
                title=dict(
                    text="模型",
                    font=dict(family=self.font_family)
                ),
                tickfont=dict(family=self.font_family)
            ),
            width=self.figsize[0],
            height=self.figsize[1],
            font=dict(family=self.font_family),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5,
                font=dict(family=self.font_family)
            )
        )
        
        # Save both PNG and HTML
        filename_base = f"entropy_heatmap_query_{query_idx}"
        png_path = os.path.join(self.output_dir, f"{filename_base}.png")
        html_path = os.path.join(self.output_dir, f"{filename_base}.html")
        
        fig.write_image(png_path, width=self.figsize[0], height=self.figsize[1])
        fig.write_html(html_path)
        
        print(f"Saved entropy heatmap for query {query_idx}: {filename_base}.png and {filename_base}.html")
    
    def _add_target_token_markers_plotly(self, fig, query_data: Dict[str, Any], grid_size: int):
        """
        Add target token markers to plotly heatmap
        
        Args:
            fig: Plotly figure
            query_data: Query data
            grid_size: Size of the interpolation grid
        """
        models = list(query_data['models'].keys())
        
        # Define different markers and colors for different target tokens
        marker_symbols = ['star', 'circle', 'square', 'triangle-up', 'triangle-down', 
                         'triangle-left', 'triangle-right', 'diamond', 'pentagon', 'cross', 
                         'x', 'hexagon', 'hourglass', 'star-diamond', 'star-square']
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 
                 'olive', 'cyan', 'magenta', 'yellow', 'navy', 'lime', 'maroon']
        
        # Collect all unique target tokens that have positions
        all_target_tokens = set()
        for model_name in models:
            model_data = query_data['models'][model_name]
            target_positions = model_data['target_positions']
            for target_token, positions in target_positions.items():
                if positions:  # Only include tokens that have positions
                    all_target_tokens.add(target_token)
        
        all_target_tokens = sorted(list(all_target_tokens))
        
        # Create a mapping from target tokens to markers and colors
        token_style_map = {}
        for i, token in enumerate(all_target_tokens):
            token_style_map[token] = {
                'symbol': marker_symbols[i % len(marker_symbols)],
                'color': colors[i % len(colors)]
            }
        
        # Add markers for each model and target token
        for model_idx, model_name in enumerate(models):
            model_data = query_data['models'][model_name]
            target_positions = model_data['target_positions']
            seq_length = model_data['sequence_length']
            
            for target_token, positions in target_positions.items():
                if not positions:  # Skip if no positions found
                    continue
                    
                style = token_style_map[target_token]
                
                for pos in positions:
                    # Convert position to normalized coordinate
                    if seq_length > 1:
                        normalized_pos = pos / (seq_length - 1)
                    else:
                        normalized_pos = 0.5
                    
                    # Add marker
                    fig.add_trace(go.Scatter(
                        x=[normalized_pos],
                        y=[model_name],
                        mode='markers',
                        marker=dict(
                            symbol=style['symbol'],
                            size=12,
                            color=style['color'],
                            line=dict(width=2, color='white')
                        ),
                        name=f'{target_token}',
                        showlegend=target_token not in [trace.name for trace in fig.data if hasattr(trace, 'name') and trace.name],
                        legendgroup=target_token
                    ))
    
    def _create_summary_heatmap(self, entropy_results: Dict[str, Any]):
        """
        Create a summary dashboard showing various entropy statistics using plotly
        
        Args:
            entropy_results: Complete entropy analysis results
        """
        summary_stats = entropy_results['summary_stats']
        models = list(summary_stats.keys())
        
        if not models:
            return
        
        # Create subplots with Chinese titles
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '所有查询的汇总统计', 
                '每查询平均熵值分布',
                '难度等级平均熵值', 
                '发现的目标标记总数'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Summary statistics heatmap
        stats_labels = ['平均熵值', '标记词平均熵值', '非标记词平均熵值', '回答正确查询平均熵值', '回答错误查询平均熵值']
        stats_data = []
        
        # Calculate target, non-target, correct and incorrect query entropies for each model
        for model in models:
            stats = summary_stats[model]
            
            # Calculate target token and non-target token entropies
            target_entropies = []
            non_target_entropies = []
            correct_query_entropies = []
            incorrect_query_entropies = []
            
            for query_data in entropy_results['query_entropy_data']:
                if model in query_data['models']:
                    model_data = query_data['models'][model]
                    entropy_values = model_data['entropy']
                    target_positions = model_data['target_positions']
                    is_correct = model_data.get('is_correct', False)
                    
                    # Get all target token positions
                    all_target_positions = set()
                    for positions in target_positions.values():
                        all_target_positions.update(positions)
                    
                    # Separate target and non-target entropies
                    for i, entropy in enumerate(entropy_values):
                        if i in all_target_positions:
                            target_entropies.append(entropy)
                        else:
                            non_target_entropies.append(entropy)
                    
                    # Separate correct and incorrect query entropies
                    query_avg_entropy = np.mean(entropy_values)
                    if is_correct:
                        correct_query_entropies.append(query_avg_entropy)
                    else:
                        incorrect_query_entropies.append(query_avg_entropy)
            
            # Calculate averages
            target_avg = np.mean(target_entropies) if target_entropies else 0
            non_target_avg = np.mean(non_target_entropies) if non_target_entropies else 0
            correct_avg = np.mean(correct_query_entropies) if correct_query_entropies else 0
            incorrect_avg = np.mean(incorrect_query_entropies) if incorrect_query_entropies else 0
            
            stats_data.append([
                stats['mean_entropy'],
                target_avg,
                non_target_avg,
                correct_avg,
                incorrect_avg
            ])
        
        fig.add_trace(
            go.Heatmap(
                z=stats_data,
                x=stats_labels,
                y=models,
                colorscale='viridis',
                text=[[f'{val:.3f}' for val in row] for row in stats_data],
                texttemplate="%{text}",
                textfont={"size": 10, "family": self.font_family},
                showscale=False  # Remove color bar
            ),
            row=1, col=1
        )
        
        # 2. Query-level average entropy box plot
        query_entropies = {model: [] for model in models}
        
        for query_data in entropy_results['query_entropy_data']:
            for model_name, model_data in query_data['models'].items():
                avg_entropy = np.mean(model_data['entropy'])
                query_entropies[model_name].append(avg_entropy)
        
        for i, model in enumerate(models):
            fig.add_trace(
                go.Box(
                    y=query_entropies[model],
                    name=model,
                    boxpoints='outliers',
                    marker_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
                ),
                row=1, col=2
            )
        
        # 3. Level-wise analysis
        level_entropies = {model: {} for model in models}
        
        for query_data in entropy_results['query_entropy_data']:
            level = query_data['metadata'].get('level', 'Unknown')
            for model_name, model_data in query_data['models'].items():
                if level not in level_entropies[model_name]:
                    level_entropies[model_name][level] = []
                avg_entropy = np.mean(model_data['entropy'])
                level_entropies[model_name][level].append(avg_entropy)
        
        levels = sorted(set(query_data['metadata'].get('level', 'Unknown') 
                          for query_data in entropy_results['query_entropy_data']))
        
        # Ensure all 5 levels are included (1-5)
        all_levels = [1, 2, 3, 4, 5]
        
        if len(levels) > 1 and 'Unknown' not in levels:
            level_means = []
            for model in models:
                model_level_means = []
                for level in all_levels:  # Use all_levels instead of levels
                    if level in level_entropies[model] and level_entropies[model][level]:
                        model_level_means.append(np.mean(level_entropies[model][level]))
                    else:
                        model_level_means.append(0)
                level_means.append(model_level_means)
            
            fig.add_trace(
                go.Heatmap(
                    z=level_means,
                    x=[f'Level {l}' for l in all_levels],  # Use all_levels instead of levels
                    y=models,
                    colorscale='viridis',
                    text=[[f'{val:.3f}' for val in row] for row in level_means],
                    texttemplate="%{text}",
                    textfont={"size": 10, "family": self.font_family},
                    showscale=False  # Remove color bar for this heatmap
                ),
                row=2, col=1
            )
        else:
            # Add placeholder text if level analysis not available
            fig.add_annotation(
                text="等级分析不可用<br>(等级数据不足)",
                x=0.5, y=0.5,
                xref="x3", yref="y3",
                showarrow=False,
                font=dict(size=16, family=self.font_family)
            )
        
        # 4. Number of target tokens found per model
        target_token_counts = {model: 0 for model in models}
        
        for query_data in entropy_results['query_entropy_data']:
            for model_name, model_data in query_data['models'].items():
                count = sum(len(positions) for positions in model_data['target_positions'].values())
                target_token_counts[model_name] += count
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=[target_token_counts[model] for model in models],
                text=[str(target_token_counts[model]) for model in models],
                textposition='auto',
                marker_color=px.colors.qualitative.Set1[:len(models)],
                textfont=dict(family=self.font_family)
            ),
            row=2, col=2
        )
        
        # Update layout with Chinese font support
        fig.update_layout(
            title=dict(
                text='熵值分析汇总报告',
                font=dict(family=self.font_family, size=20)
            ),
            showlegend=False,
            height=800,
            width=1400,
            font=dict(family=self.font_family)
        )
        
        # Update all subplot axes with Chinese fonts
        fig.update_xaxes(tickfont=dict(family=self.font_family))
        fig.update_yaxes(tickfont=dict(family=self.font_family))
        
        # Update specific axis titles
        fig.update_xaxes(title_text="统计指标", row=1, col=1, title_font=dict(family=self.font_family))
        fig.update_yaxes(title_text="模型", row=1, col=1, title_font=dict(family=self.font_family))
        
        fig.update_yaxes(title_text="平均熵值", row=1, col=2, title_font=dict(family=self.font_family))
        
        fig.update_xaxes(title_text="难度等级", row=2, col=1, title_font=dict(family=self.font_family))
        fig.update_yaxes(title_text="模型", row=2, col=1, title_font=dict(family=self.font_family))
        
        fig.update_xaxes(title_text="模型", row=2, col=2, title_font=dict(family=self.font_family))
        fig.update_yaxes(title_text="目标标记数量", row=2, col=2, title_font=dict(family=self.font_family))
        
        # Save both PNG and HTML
        filename_base = "entropy_analysis_summary"
        png_path = os.path.join(self.output_dir, f"{filename_base}.png")
        html_path = os.path.join(self.output_dir, f"{filename_base}.html")
        
        fig.write_image(png_path, width=1400, height=800)
        fig.write_html(html_path)
        
        print(f"Saved entropy analysis summary: {filename_base}.png and {filename_base}.html")
