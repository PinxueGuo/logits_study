"""
Visualization utilities for logits analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
    
    def plot_token_logits_heatmap(self, logits_data: Dict[str, np.ndarray], 
                                 tokens: List[str], target_positions: Dict[str, List[int]],
                                 title: str = "Token Logits Heatmap"):
        """
        Create heatmap of logits around target tokens
        
        Args:
            logits_data: Dictionary mapping model names to logits arrays
            tokens: List of tokens
            target_positions: Dictionary of target token positions
            title: Plot title
        """
        fig, axes = plt.subplots(len(logits_data), 1, figsize=(self.figsize[0], self.figsize[1] * len(logits_data)))
        if len(logits_data) == 1:
            axes = [axes]
        
        for i, (model_name, logits) in enumerate(logits_data.items()):
            # Create a subset of logits for visualization (top-k most probable tokens)
            top_k = 50  # Show top 50 tokens
            avg_logits = np.mean(logits, axis=0)
            top_indices = np.argsort(avg_logits)[-top_k:]
            
            # Create heatmap data
            heatmap_data = logits[:, top_indices].T
            
            sns.heatmap(heatmap_data, ax=axes[i], cmap='viridis', 
                       xticklabels=range(len(tokens)), 
                       yticklabels=[f"Token_{idx}" for idx in top_indices])
            
            axes[i].set_title(f"{title} - {model_name}")
            axes[i].set_xlabel("Token Position")
            axes[i].set_ylabel("Vocabulary Tokens")
            
            # Mark target token positions
            for token, positions in target_positions.items():
                for pos in positions:
                    if pos < len(tokens):
                        axes[i].axvline(x=pos, color='red', linestyle='--', alpha=0.7, linewidth=2)
                        axes[i].text(pos, top_k//2, token, rotation=90, 
                                   verticalalignment='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/logits_heatmap.png", dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_target_token_probabilities(self, prob_data: Dict[str, Dict[str, np.ndarray]], 
                                      title: str = "Target Token Probabilities"):
        """
        Plot probabilities of target tokens across models
        
        Args:
            prob_data: Nested dict {model: {token: probabilities}}
            title: Plot title
        """
        fig = make_subplots(
            rows=len(prob_data), cols=1,
            subplot_titles=list(prob_data.keys()),
            vertical_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, (model_name, model_probs) in enumerate(prob_data.items()):
            for j, (token, probs) in enumerate(model_probs.items()):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(probs))),
                        y=probs,
                        mode='lines+markers',
                        name=f"{token}",
                        line=dict(color=colors[j % len(colors)]),
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title=title,
            height=300 * len(prob_data),
            xaxis_title="Token Position",
            yaxis_title="Probability"
        )
        
        fig.write_html(f"{self.output_dir}/target_token_probabilities.html")
        return fig
    
    def plot_model_comparison(self, comparison_data: Dict[str, Dict[str, float]], 
                            metric: str = "Average Probability"):
        """
        Create comparison plots between models
        
        Args:
            comparison_data: {model: {token: metric_value}}
            metric: Name of the metric being compared
        """
        # Convert to DataFrame for easier plotting
        df_data = []
        for model, token_data in comparison_data.items():
            for token, value in token_data.items():
                df_data.append({'Model': model, 'Token': token, 'Value': value})
        
        df = pd.DataFrame(df_data)
        
        # Create grouped bar plot
        plt.figure(figsize=self.figsize)
        sns.barplot(data=df, x='Token', y='Value', hue='Model')
        plt.title(f"Model Comparison: {metric}")
        plt.xticks(rotation=45)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/model_comparison_{metric.lower().replace(' ', '_')}.png", 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_level_analysis(self, level_data: Dict[int, Dict[str, float]], 
                          metric: str = "Average Probability"):
        """
        Plot analysis by difficulty level
        
        Args:
            level_data: {level: {token: metric_value}}
            metric: Name of the metric being compared
        """
        # Convert to DataFrame
        df_data = []
        for level, token_data in level_data.items():
            for token, value in token_data.items():
                df_data.append({'Level': level, 'Token': token, 'Value': value})
        
        df = pd.DataFrame(df_data)
        
        # Create line plot
        plt.figure(figsize=self.figsize)
        for token in df['Token'].unique():
            token_data = df[df['Token'] == token]
            plt.plot(token_data['Level'], token_data['Value'], marker='o', label=token)
        
        plt.title(f"Analysis by Difficulty Level: {metric}")
        plt.xlabel("Difficulty Level")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/level_analysis_{metric.lower().replace(' ', '_')}.png", 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_correctness_comparison(self, correctness_data: Dict[bool, Dict[str, float]], 
                                  metric: str = "Average Probability"):
        """
        Plot comparison between correct and incorrect answers
        
        Args:
            correctness_data: {is_correct: {token: metric_value}}
            metric: Name of the metric being compared
        """
        # Convert to DataFrame
        df_data = []
        for is_correct, token_data in correctness_data.items():
            correctness_label = "Correct" if is_correct else "Incorrect"
            for token, value in token_data.items():
                df_data.append({'Correctness': correctness_label, 'Token': token, 'Value': value})
        
        df = pd.DataFrame(df_data)
        
        # Create grouped bar plot
        plt.figure(figsize=self.figsize)
        sns.barplot(data=df, x='Token', y='Value', hue='Correctness')
        plt.title(f"Correctness Comparison: {metric}")
        plt.xticks(rotation=45)
        plt.legend(title='Answer Correctness')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/correctness_comparison_{metric.lower().replace(' ', '_')}.png", 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def create_interactive_dashboard(self, all_data: Dict[str, Any]):
        """
        Create an interactive dashboard with all visualizations
        
        Args:
            all_data: Dictionary containing all analysis results
        """
        # This would create a comprehensive dashboard
        # For now, we'll create a simple HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Logits Analysis Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin: 30px 0; }}
                .plot {{ margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>LLM Logits Analysis Dashboard</h1>
            
            <div class="section">
                <h2>Model Comparison</h2>
                <div class="plot">
                    <img src="model_comparison_average_probability.png" alt="Model Comparison">
                </div>
            </div>
            
            <div class="section">
                <h2>Level Analysis</h2>
                <div class="plot">
                    <img src="level_analysis_average_probability.png" alt="Level Analysis">
                </div>
            </div>
            
            <div class="section">
                <h2>Correctness Analysis</h2>
                <div class="plot">
                    <img src="correctness_comparison_average_probability.png" alt="Correctness Comparison">
                </div>
            </div>
            
            <div class="section">
                <h2>Logits Heatmap</h2>
                <div class="plot">
                    <img src="logits_heatmap.png" alt="Logits Heatmap">
                </div>
            </div>
            
            <div class="section">
                <h2>Interactive Probability Plot</h2>
                <div class="plot">
                    <iframe src="target_token_probabilities.html" width="100%" height="600"></iframe>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(f"{self.output_dir}/dashboard.html", 'w') as f:
            f.write(html_content)
        
        print(f"Dashboard created at {self.output_dir}/dashboard.html")
