"""
Data processing utilities for logits analysis
"""

import json
import jsonlines
import pandas as pd
from typing import List, Dict, Any
import os
from pathlib import Path

class DataProcessor:
    def __init__(self, input_file: str):
        """
        Initialize data processor
        
        Args:
            input_file: Path to the input JSONL file
        """
        self.input_file = input_file
        self.data = []
    
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load data from JSONL file
        
        Returns:
            List of data dictionaries
        """
        if not os.path.exists(self.input_file):
            # 尝试使用默认的数据文件
            default_file = "data/logits_study_data.jsonl"
            if os.path.exists(default_file):
                print(f"Using default data file: {default_file}")
                self.input_file = default_file
            else:
                raise FileNotFoundError(f"Input file {self.input_file} not found and no default data available")
        
        with jsonlines.open(self.input_file) as reader:
            self.data = list(reader)
        
        print(f"Loaded {len(self.data)} samples from {self.input_file}")
        return self.data
    
    def validate_data(self) -> bool:
        """
        Validate that data has required fields
        
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['query', 'answer', 'level']
        
        for i, item in enumerate(self.data):
            for field in required_fields:
                if field not in item:
                    print(f"Missing field '{field}' in item {i}")
                    return False
        
        print("Data validation passed")
        return True
    
    def get_data_by_level(self, level: int) -> List[Dict[str, Any]]:
        """
        Filter data by level
        
        Args:
            level: Level to filter by (1-5)
            
        Returns:
            Filtered data
        """
        return [item for item in self.data if item['level'] == level]
    
    def get_data_by_correctness(self, is_correct: bool) -> List[Dict[str, Any]]:
        """
        Filter data by correctness (if correctness field exists)
        
        Args:
            is_correct: Whether to get correct or incorrect answers
            
        Returns:
            Filtered data
        """
        if 'is_correct' not in self.data[0]:
            print("No 'is_correct' field found in data")
            return self.data
        
        return [item for item in self.data if item['is_correct'] == is_correct]
    
    def create_full_text(self, item: Dict[str, Any], include_system_prompt: bool = True) -> str:
        """
        Create full text from query and answer
        
        Args:
            item: Data item
            include_system_prompt: Whether to include system prompt
            
        Returns:
            Full text string
        """
        if include_system_prompt:
            system_prompt = "Please reason step by step, and put your final answer within \\boxed{}"
            return f"System: {system_prompt}\nQuery: {item['query']}\nAnswer: {item['answer']}"
        else:
            return f"Query: {item['query']}\nAnswer: {item['answer']}"
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of the data
        
        Returns:
            Summary statistics
        """
        if not self.data:
            return {}
        
        df = pd.DataFrame(self.data)
        
        stats = {
            'total_samples': len(self.data),
            'level_distribution': df['level'].value_counts().to_dict(),
            'avg_query_length': df['query'].str.len().mean(),
            'avg_answer_length': df['answer'].str.len().mean(),
        }
        
        if 'is_correct' in df.columns:
            stats['correctness_distribution'] = df['is_correct'].value_counts().to_dict()
        
        return stats
    
    def save_processed_data(self, data: List[Dict[str, Any]], output_file: str):
        """
        Save processed data to file
        
        Args:
            data: Data to save
            output_file: Output file path
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with jsonlines.open(output_file, 'w') as writer:
            writer.write_all(data)
        
        print(f"Saved {len(data)} items to {output_file}")

    def evaluate_predictions(self, predictions: List[str]) -> List[Dict[str, Any]]:
        """
        评估预测答案的正确性
        
        Args:
            predictions: 模型预测的回答列表
            
        Returns:
            包含正确性评估的数据列表
        """
        from answer_comparator import extract_and_compare_answer
        
        if len(predictions) != len(self.data):
            raise ValueError(f"预测数量({len(predictions)})与数据数量({len(self.data)})不匹配")
        
        evaluated_data = []
        correct_count = 0
        
        for i, (item, prediction) in enumerate(zip(self.data, predictions)):
            # 提取并比较答案
            is_correct, extracted_answer, explanation = extract_and_compare_answer(
                prediction, item['answer']
            )
            
            # 创建新的数据项
            evaluated_item = item.copy()
            evaluated_item.update({
                'prediction': prediction,
                'extracted_answer': extracted_answer,
                'is_correct': is_correct,
                'comparison_explanation': explanation
            })
            
            evaluated_data.append(evaluated_item)
            
            if is_correct:
                correct_count += 1
        
        accuracy = correct_count / len(predictions) if predictions else 0
        print(f"评估完成: {correct_count}/{len(predictions)} 正确 (准确率: {accuracy:.2%})")
        
        return evaluated_data
    
    def add_correctness_from_predictions(self, predictions: List[str]) -> None:
        """
        根据预测结果添加正确性标记到现有数据
        
        Args:
            predictions: 模型预测的回答列表
        """
        evaluated_data = self.evaluate_predictions(predictions)
        self.data = evaluated_data
        print("数据已更新，包含正确性评估结果")

def create_sample_data(output_file: str, num_samples: int = 20):
    """
    Create sample data for testing
    
    Args:
        output_file: Output file path
        num_samples: Number of samples to create
    """
    import random
    
    sample_queries = [
        "What is the derivative of x^2 + 3x + 1?",
        "Solve the equation 2x + 5 = 15",
        "Find the area of a circle with radius 5",
        "What is the integral of sin(x)?",
        "Solve the quadratic equation x^2 - 4x + 3 = 0",
        "Calculate the limit of (x^2 - 1)/(x - 1) as x approaches 1",
        "Find the slope of the line passing through (1,2) and (3,6)",
        "What is the sum of the first 10 natural numbers?",
        "Solve the system: x + y = 5, x - y = 1",
        "Find the volume of a sphere with radius 3"
    ]
    
    sample_answers = [
        "The derivative is 2x + 3",
        "x = 5",
        "The area is 25π square units",
        "The integral is -cos(x) + C",
        "x = 1 or x = 3",
        "The limit is 2",
        "The slope is 2",
        "The sum is 55",
        "x = 3, y = 2",
        "The volume is 36π cubic units"
    ]
    
    reasoning_words = ['wait', 'aha', 'check', 'think', 'hmm', 'let', 'actually', 'however', 'so', 'therefore']
    
    data = []
    for i in range(num_samples):
        query_idx = i % len(sample_queries)
        
        # Add some reasoning words to answers randomly
        answer = sample_answers[query_idx]
        if random.random() < 0.7:  # 70% chance to add reasoning
            reasoning = random.choice(reasoning_words)
            answer = f"{reasoning}, {answer}"
        
        data.append({
            'query': sample_queries[query_idx],
            'answer': answer,
            'level': random.randint(1, 5),
            'is_correct': random.choice([True, False])
        })
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(data)
    
    print(f"Created {num_samples} sample data points in {output_file}")
