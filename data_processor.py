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
            raise FileNotFoundError(f"Input file {self.input_file} not found")
        
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
    
    def create_full_text(self, item: Dict[str, Any]) -> str:
        """
        Create full text from query and answer
        
        Args:
            item: Data item
            
        Returns:
            Full text string
        """
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
