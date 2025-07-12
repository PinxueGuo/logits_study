"""
Main analysis pipeline for logits study
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import pickle

from config import MODELS, TARGET_TOKENS, DATA_CONFIG, ANALYSIS_CONFIG
from model_loader import LogitsExtractor
from data_processor import DataProcessor, create_sample_data
from visualizer import LogitsVisualizer

class LogitsAnalyzer:
    def __init__(self):
        """Initialize the logits analyzer"""
        self.models = {}
        self.data_processor = None
        self.visualizer = LogitsVisualizer(DATA_CONFIG['output_dir'])
        
        # Create necessary directories
        os.makedirs(DATA_CONFIG['output_dir'], exist_ok=True)
        os.makedirs(DATA_CONFIG['cache_dir'], exist_ok=True)
    
    def load_models(self, model_names: List[str] = None):
        """
        Load specified models
        
        Args:
            model_names: List of model names to load (default: all models)
        """
        if model_names is None:
            model_names = list(MODELS.keys())
        
        for model_name in model_names:
            if model_name not in MODELS:
                print(f"Warning: Model {model_name} not found in configuration")
                continue
            
            model_config = MODELS[model_name]
            print(f"Loading {model_name}: {model_config['description']}")
            
            try:
                self.models[model_name] = LogitsExtractor(
                    model_config['path'],
                    device=ANALYSIS_CONFIG['device'],
                    torch_dtype=ANALYSIS_CONFIG['torch_dtype']
                )
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                continue
    
    def load_data(self, data_file: str = None):
        """
        Load and validate data
        
        Args:
            data_file: Path to data file (default: from config)
        """
        if data_file is None:
            data_file = DATA_CONFIG['input_file']
        
        # Create sample data if file doesn't exist
        if not os.path.exists(data_file):
            print(f"Data file {data_file} not found. Creating sample data...")
            create_sample_data(data_file, num_samples=50)
        
        self.data_processor = DataProcessor(data_file)
        self.data_processor.load_data()
        
        if not self.data_processor.validate_data():
            raise ValueError("Data validation failed")
        
        # Print summary statistics
        stats = self.data_processor.get_summary_stats()
        print("Data Summary:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    def extract_all_logits(self, save_cache: bool = True) -> Dict[str, List[Tuple[np.ndarray, List[str], Dict[str, Any]]]]:
        """
        Extract logits for all models and all data
        
        Args:
            save_cache: Whether to save results to cache
            
        Returns:
            Dictionary mapping model names to list of (logits, tokens, metadata) tuples
        """
        all_results = {}
        
        for model_name, model in self.models.items():
            print(f"\\nExtracting logits for {model_name}...")
            
            cache_file = f"{DATA_CONFIG['cache_dir']}/{model_name}_logits.pkl"
            
            # Try to load from cache first
            if os.path.exists(cache_file):
                print(f"Loading cached results for {model_name}")
                with open(cache_file, 'rb') as f:
                    all_results[model_name] = pickle.load(f)
                continue
            
            model_results = []
            
            for item in tqdm(self.data_processor.data, desc=f"Processing {model_name}"):
                full_text = self.data_processor.create_full_text(item)
                
                try:
                    logits, tokens = model.extract_logits(
                        full_text, 
                        max_length=ANALYSIS_CONFIG['max_length']
                    )
                    
                    # Store with metadata
                    model_results.append((logits, tokens, item))
                    
                except Exception as e:
                    print(f"Error processing item: {e}")
                    continue
            
            all_results[model_name] = model_results
            
            # Save to cache
            if save_cache:
                with open(cache_file, 'wb') as f:
                    pickle.dump(model_results, f)
                print(f"Cached results for {model_name}")
        
        return all_results
    
    def analyze_target_tokens(self, all_logits: Dict[str, List[Tuple[np.ndarray, List[str], Dict[str, Any]]]]) -> Dict[str, Any]:
        """
        Analyze target tokens across all models and conditions
        
        Args:
            all_logits: Results from extract_all_logits
            
        Returns:
            Analysis results
        """
        analysis_results = {
            'model_comparison': {},
            'level_analysis': {},
            'correctness_analysis': {},
            'target_token_stats': {}
        }
        
        # Get target token IDs for each model
        target_token_ids = {}
        for model_name, model in self.models.items():
            target_token_ids[model_name] = {}
            for token in TARGET_TOKENS:
                # Try different variations of the token
                variations = [token, token.capitalize(), f" {token}", f"Ġ{token}", f"▁{token}"]
                for var in variations:
                    try:
                        token_id = model.tokenizer.encode(var, add_special_tokens=False)
                        if token_id:
                            target_token_ids[model_name][token] = token_id[0]
                            break
                    except:
                        continue
        
        # Analyze each model
        for model_name, results in all_logits.items():
            print(f"\\nAnalyzing {model_name}...")
            
            model_token_probs = {token: [] for token in TARGET_TOKENS}
            level_token_probs = {i: {token: [] for token in TARGET_TOKENS} for i in range(1, 6)}
            correct_token_probs = {True: {token: [] for token in TARGET_TOKENS}, 
                                 False: {token: [] for token in TARGET_TOKENS}}
            
            for logits, tokens, metadata in results:
                # Find target token positions
                target_positions = self.models[model_name].find_target_token_positions(
                    tokens, TARGET_TOKENS
                )
                
                # Get probabilities for target tokens
                if model_name in target_token_ids:
                    token_ids = [target_token_ids[model_name].get(token, 0) for token in TARGET_TOKENS]
                    probs = self.models[model_name].get_token_probabilities(logits, token_ids)
                    
                    # Store probabilities
                    for i, token in enumerate(TARGET_TOKENS):
                        if token in target_positions and target_positions[token]:
                            # Average probability around target positions
                            pos_probs = []
                            for pos in target_positions[token]:
                                if pos < len(probs):
                                    pos_probs.append(probs[pos, i])
                            
                            if pos_probs:
                                avg_prob = np.mean(pos_probs)
                                model_token_probs[token].append(avg_prob)
                                level_token_probs[metadata['level']][token].append(avg_prob)
                                
                                if 'is_correct' in metadata:
                                    correct_token_probs[metadata['is_correct']][token].append(avg_prob)
            
            # Store results
            analysis_results['model_comparison'][model_name] = {
                token: np.mean(probs) if probs else 0.0 
                for token, probs in model_token_probs.items()
            }
            
            # Level analysis for this model
            for level in range(1, 6):
                if level not in analysis_results['level_analysis']:
                    analysis_results['level_analysis'][level] = {}
                
                analysis_results['level_analysis'][level][model_name] = {
                    token: np.mean(probs) if probs else 0.0 
                    for token, probs in level_token_probs[level].items()
                }
            
            # Correctness analysis for this model
            for is_correct in [True, False]:
                key = 'correct' if is_correct else 'incorrect'
                if key not in analysis_results['correctness_analysis']:
                    analysis_results['correctness_analysis'][key] = {}
                
                analysis_results['correctness_analysis'][key][model_name] = {
                    token: np.mean(probs) if probs else 0.0 
                    for token, probs in correct_token_probs[is_correct].items()
                }
        
        return analysis_results
    
    def create_visualizations(self, analysis_results: Dict[str, Any], all_logits: Dict[str, List[Tuple[np.ndarray, List[str], Dict[str, Any]]]]):
        """
        Create all visualizations
        
        Args:
            analysis_results: Results from analyze_target_tokens
            all_logits: Raw logits data
        """
        print("\\nCreating visualizations...")
        
        # Model comparison
        if analysis_results['model_comparison']:
            self.visualizer.plot_model_comparison(
                analysis_results['model_comparison'],
                "Average Target Token Probability"
            )
        
        # Level analysis - reorganize data for visualization
        level_viz_data = {}
        for level, level_data in analysis_results['level_analysis'].items():
            for model, token_data in level_data.items():
                for token, prob in token_data.items():
                    if token not in level_viz_data:
                        level_viz_data[token] = {}
                    if level not in level_viz_data[token]:
                        level_viz_data[token][level] = {}
                    level_viz_data[token][level][model] = prob
        
        # Create level analysis plot for each token
        for token in TARGET_TOKENS:
            if token in level_viz_data:
                token_level_data = {}
                for level in range(1, 6):
                    if level in level_viz_data[token]:
                        token_level_data[level] = level_viz_data[token][level]
                
                if token_level_data:
                    # Average across models for this token
                    avg_level_data = {}
                    for level, model_data in token_level_data.items():
                        avg_level_data[level] = np.mean(list(model_data.values()))
                    
                    # Add to level analysis data
                    if not hasattr(self, '_level_plot_data'):
                        self._level_plot_data = {}
                    self._level_plot_data[token] = avg_level_data
        
        if hasattr(self, '_level_plot_data'):
            # Reorganize for plotting
            plot_level_data = {}
            for level in range(1, 6):
                plot_level_data[level] = {}
                for token, level_data in self._level_plot_data.items():
                    if level in level_data:
                        plot_level_data[level][token] = level_data[level]
            
            self.visualizer.plot_level_analysis(plot_level_data, "Average Target Token Probability")
        
        # Correctness analysis
        if 'correct' in analysis_results['correctness_analysis'] and 'incorrect' in analysis_results['correctness_analysis']:
            # Reorganize data
            correctness_viz_data = {True: {}, False: {}}
            
            for token in TARGET_TOKENS:
                correct_probs = []
                incorrect_probs = []
                
                for model in self.models.keys():
                    if model in analysis_results['correctness_analysis']['correct']:
                        if token in analysis_results['correctness_analysis']['correct'][model]:
                            correct_probs.append(analysis_results['correctness_analysis']['correct'][model][token])
                    
                    if model in analysis_results['correctness_analysis']['incorrect']:
                        if token in analysis_results['correctness_analysis']['incorrect'][model]:
                            incorrect_probs.append(analysis_results['correctness_analysis']['incorrect'][model][token])
                
                if correct_probs:
                    correctness_viz_data[True][token] = np.mean(correct_probs)
                if incorrect_probs:
                    correctness_viz_data[False][token] = np.mean(incorrect_probs)
            
            self.visualizer.plot_correctness_comparison(correctness_viz_data, "Average Target Token Probability")
        
        # Create dashboard
        self.visualizer.create_interactive_dashboard(analysis_results)
        
        print(f"All visualizations saved to {DATA_CONFIG['output_dir']}")
    
    def run_full_analysis(self, model_names: List[str] = None, data_file: str = None, 
                         generate_predictions: bool = True):
        """
        Run the complete analysis pipeline
        
        Args:
            model_names: List of model names to analyze
            data_file: Path to data file
            generate_predictions: Whether to generate model predictions for correctness evaluation
        """
        print("=== Starting Logits Analysis ===")
        
        # Load data
        self.load_data(data_file)
        
        # Load models
        self.load_models(model_names)
        
        if not self.models:
            print("No models loaded successfully. Exiting.")
            return
        
        # Generate predictions and evaluate correctness if requested
        if generate_predictions and self.models:
            print("\\n=== Generating Predictions for Correctness Evaluation ===")
            
            # Use the first available model for prediction generation
            # (You can modify this to generate predictions for all models)
            prediction_model = list(self.models.keys())[0]
            print(f"Using {prediction_model} for prediction generation...")
            
            try:
                predictions = self.generate_model_predictions(prediction_model, self.data_processor.data)
                
                # Evaluate correctness and update data
                self.data_processor.add_correctness_from_predictions(predictions)
                
                # Save predictions and evaluations
                eval_file = f"{DATA_CONFIG['output_dir']}/predictions_evaluation.json"
                os.makedirs(DATA_CONFIG['output_dir'], exist_ok=True)
                
                eval_data = {
                    'model_used': prediction_model,
                    'total_samples': len(predictions),
                    'predictions': [
                        {
                            'query': item['query'],
                            'ground_truth': item['answer'],
                            'prediction': item['prediction'],
                            'extracted_answer': item['extracted_answer'],
                            'is_correct': item['is_correct'],
                            'explanation': item['comparison_explanation']
                        }
                        for item in self.data_processor.data
                    ]
                }
                
                with open(eval_file, 'w', encoding='utf-8') as f:
                    json.dump(eval_data, f, ensure_ascii=False, indent=2)
                
                print(f"Predictions and evaluations saved to {eval_file}")
                
            except Exception as e:
                print(f"Warning: Failed to generate predictions: {e}")
                print("Continuing with logits analysis...")
        
        # Extract logits
        all_logits = self.extract_all_logits()
        
        # Analyze target tokens
        analysis_results = self.analyze_target_tokens(all_logits)
        
        # Create visualizations
        self.create_visualizations(analysis_results, all_logits)
        
        # Save analysis results
        results_file = f"{DATA_CONFIG['output_dir']}/analysis_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(analysis_results)
            json.dump(serializable_results, f, indent=2)
        
        print(f"\\n=== Analysis Complete ===")
        print(f"Results saved to {DATA_CONFIG['output_dir']}")
        print(f"Open {DATA_CONFIG['output_dir']}/dashboard.html to view the dashboard")
        if generate_predictions:
            print(f"Prediction evaluations available in {DATA_CONFIG['output_dir']}/predictions_evaluation.json")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible types"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def generate_model_predictions(self, model_name: str, data: List[Dict[str, Any]]) -> List[str]:
        """
        使用指定模型生成预测答案
        
        Args:
            model_name: 模型名称
            data: 输入数据列表
            
        Returns:
            预测答案列表
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        predictions = []
        
        print(f"Generating predictions with {model_name}...")
        
        for item in tqdm(data, desc=f"Predicting with {model_name}"):
            query = item['query']
            
            # 构造输入文本
            input_text = f"Question: {query}\nAnswer:"
            
            try:
                # 生成回答
                inputs = model.tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=ANALYSIS_CONFIG['max_length'],
                    truncation=True
                ).to(model.device)
                
                with torch.no_grad():
                    outputs = model.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=model.tokenizer.eos_token_id
                    )
                
                # 解码生成的文本
                generated_text = model.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                
                predictions.append(generated_text)
                
            except Exception as e:
                print(f"Error generating prediction for item: {e}")
                predictions.append("")  # 添加空预测以保持索引一致
        
        return predictions

if __name__ == "__main__":
    analyzer = LogitsAnalyzer()
    
    # Run analysis with sample models (you can modify this)
    # For testing, we'll just use the first model if others are not available
    try:
        analyzer.run_full_analysis(['baseline'])  # Start with just one model for testing
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("This might be because the model paths are not accessible in this environment.")
        print("Please modify the model paths in config.py to match your actual model locations.")
