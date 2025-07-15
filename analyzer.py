"""
Main analysis pipeline for logits study - Token-level entropy analysis
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
from data_processor import DataProcessor
from visualizer import LogitsVisualizer
from answer_comparator import extract_and_compare_answer

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
            print(f"\nExtracting logits for {model_name}...")
            
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
                    # 使用新的方法：一次前向传播获取生成文本和对应的logits
                    result = model.analyze_generation_uncertainty(
                        full_text, 
                        max_new_tokens=ANALYSIS_CONFIG['max_new_tokens']
                    )
                    
                    # 从结果中提取数据
                    generated_text = result['generated_text']
                    tokens = result['tokens']  # 这些是生成的token
                    logits = result['logits']  # 对应每个生成token的logits
                    entropies = result['entropies']  # 每个token的entropy
                    
                    # Store with metadata and generated answer
                    enhanced_metadata = item.copy()
                    enhanced_metadata['generated_answer'] = generated_text
                    enhanced_metadata['token_entropies'] = entropies
                    model_results.append((logits, tokens, enhanced_metadata))
                    
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
    
    def calculate_token_entropy(self, logits: np.ndarray) -> np.ndarray:
        """
        Calculate entropy for each token position (fallback method)
        Note: This is now redundant as entropy is calculated in model_loader.py
        
        Args:
            logits: Logits array of shape (seq_len, vocab_size)
            
        Returns:
            Array of entropy values for each position
        """
        if logits.size == 0:
            return np.array([])
            
        # Convert to probabilities using softmax
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        
        # Calculate entropy: H = -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        probs = np.clip(probs, epsilon, 1.0)
        entropy = -np.sum(probs * np.log(probs), axis=-1)
        
        return entropy
    
    def find_target_token_positions_in_sequence(self, tokens: List[str], target_tokens: List[str]) -> Dict[str, List[int]]:
        """
        Find positions of target tokens in the token sequence
        
        Args:
            tokens: List of tokens
            target_tokens: List of target tokens to find
            
        Returns:
            Dictionary mapping target tokens to their positions
        """
        positions = {token: [] for token in target_tokens}
        
        # Convert tokens to text for multi-token word detection
        full_text = ''.join([token.replace('▁', ' ').replace('Ġ', ' ') for token in tokens]).lower()
        
        for i, token in enumerate(tokens):
            # Clean token (remove special characters and decode properly)
            clean_token = token.strip()
            
            # Handle different tokenizer prefixes
            if clean_token.startswith('▁'):
                clean_token = clean_token[1:]
            if clean_token.startswith('Ġ'):
                clean_token = clean_token[1:]
            
            # Convert to lowercase for comparison
            clean_token_lower = clean_token.lower()
            
            # Direct token matching
            for target in target_tokens:
                target_lower = target.lower()
                # Exact match or contains match
                if clean_token_lower == target_lower or target_lower in clean_token_lower:
                    if i not in positions[target]:  # Avoid duplicates
                        positions[target].append(i)
        
        # Multi-token word detection
        for target in target_tokens:
            target_lower = target.lower()
            if len(target_lower) > 1 and target_lower in full_text:
                # Search for the target word in context around each token
                for i, token in enumerate(tokens):
                    # Check context around this token (previous 2 and next 2 tokens)
                    context_start = max(0, i - 2)
                    context_end = min(len(tokens), i + 3)
                    context_tokens = tokens[context_start:context_end]
                    context_text = ''.join([t.replace('▁', '').replace('Ġ', '') for t in context_tokens]).lower()
                    
                    if target_lower in context_text and i not in positions[target]:
                        positions[target].append(i)
        
        return positions

    def analyze_entropy_per_query(self, all_logits: Dict[str, List[Tuple[np.ndarray, List[str], Dict[str, Any]]]]) -> Dict[str, Any]:
        """
        Analyze entropy for each token position for each query across all models
        
        Args:
            all_logits: Results from extract_all_logits
            
        Returns:
            Analysis results with entropy data for visualization
        """
        analysis_results = {
            'query_entropy_data': [],  # List of entropy data for each query
            'summary_stats': {}
        }
        
        # Get all model names
        model_names = list(all_logits.keys())
        num_queries = len(all_logits[model_names[0]]) if model_names else 0
        
        print(f"Analyzing entropy for {num_queries} queries across {len(model_names)} models...")
        
        for query_idx in range(num_queries):
            query_data = {
                'query_idx': query_idx,
                'query_text': '',
                'metadata': {},
                'models': {}
            }
            
            # Process each model for this query
            for model_name in model_names:
                if query_idx >= len(all_logits[model_name]):
                    continue
                    
                logits, tokens, metadata = all_logits[model_name][query_idx]
                
                # Store query info (from first model)
                if not query_data['query_text']:
                    query_data['query_text'] = metadata.get('query', f'Query {query_idx}')
                    query_data['metadata'] = metadata
                
                # 使用预计算的entropy或者从logits计算
                if 'token_entropies' in metadata and len(metadata['token_entropies']) > 0:
                    entropy_values = metadata['token_entropies']
                else:
                    # 如果没有预计算的entropy，则从logits计算
                    entropy_values = self.calculate_token_entropy(logits)
                
                # Find target token positions
                target_positions = self.find_target_token_positions_in_sequence(tokens, TARGET_TOKENS)
                
                # Store model data with decoded tokens
                # Try to decode tokens properly for better readability
                decoded_tokens = []
                for token in tokens:
                    # For better readability, try to decode back to text when possible
                    try:
                        if hasattr(self.models[model_name], 'tokenizer'):
                            # For Qwen tokenizer, handle special tokens
                            if token.startswith('<') and token.endswith('>'):
                                decoded_tokens.append(token)  # Keep special tokens as-is
                            else:
                                # Try to get the readable token
                                decoded_tokens.append(token)
                        else:
                            decoded_tokens.append(token)
                    except:
                        decoded_tokens.append(token)
                
                # Check answer correctness if ground truth is available
                generated_answer = metadata.get('generated_answer', '')
                ground_truth = metadata.get('answer', '')
                is_correct = False
                if generated_answer and ground_truth:
                    is_correct, extracted_answer, comparison_note = extract_and_compare_answer(generated_answer, ground_truth)
                
                query_data['models'][model_name] = {
                    'entropy': entropy_values,
                    'tokens': decoded_tokens,
                    'target_positions': target_positions,
                    'sequence_length': len(tokens),
                    'generated_answer': generated_answer,
                    'is_correct': is_correct  # Add correctness information
                }
                
                print(f"Query {query_idx}, Model {model_name}: {len(entropy_values)} positions, "
                      f"avg entropy: {np.mean(entropy_values):.3f}, "
                      f"target tokens found: {sum(len(pos) for pos in target_positions.values())}")
            
            analysis_results['query_entropy_data'].append(query_data)
        
        # Calculate summary statistics
        all_entropies = {model: [] for model in model_names}
        for query_data in analysis_results['query_entropy_data']:
            for model_name, model_data in query_data['models'].items():
                all_entropies[model_name].extend(model_data['entropy'])
        
        analysis_results['summary_stats'] = {
            model: {
                'mean_entropy': np.mean(entropies) if entropies else 0,
                'std_entropy': np.std(entropies) if entropies else 0,
                'min_entropy': np.min(entropies) if entropies else 0,
                'max_entropy': np.max(entropies) if entropies else 0
            }
            for model, entropies in all_entropies.items()
        }
        
        return analysis_results
    
    def run_full_analysis(self, model_names: List[str] = None, data_file: str = None):
        """
        Run the complete analysis pipeline
        
        Args:
            model_names: List of model names to analyze
            data_file: Path to data file
        """
        print("Starting logits study analysis...")
        
        # Load models and data
        self.load_models(model_names)
        self.load_data(data_file)
        
        # Extract logits for all models
        print("\nExtracting logits...")
        all_logits = self.extract_all_logits(save_cache=True)
        
        # Analyze entropy per query
        print("\nAnalyzing entropy per token position...")
        entropy_results = self.analyze_entropy_per_query(all_logits)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self.visualizer.create_entropy_heatmaps(entropy_results)
        
        # Save results as JSONL format
        results_file = f"{DATA_CONFIG['output_dir']}/entropy_analysis_results.jsonl"
        with open(results_file, 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(entropy_results)
            
            # Write each query as a separate line in JSONL format
            for query_data in serializable_results['query_entropy_data']:
                json.dump(query_data, f, ensure_ascii=False)
                f.write('\n')
            
            # Write summary stats as the last line
            summary_line = {'summary_stats': serializable_results['summary_stats']}
            json.dump(summary_line, f, ensure_ascii=False)
            f.write('\n')
        
        print(f"\nAnalysis complete! Results saved to {results_file}")
        print("\nSummary statistics:")
        for model, stats in entropy_results['summary_stats'].items():
            print(f"  {model}: mean={stats['mean_entropy']:.3f}, std={stats['std_entropy']:.3f}")
        
        return entropy_results
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
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
