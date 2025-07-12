"""
Model loader and logits extractor for Qwen models
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import json
import os

class LogitsExtractor:
    def __init__(self, model_path: str, device: str = 'cuda', torch_dtype: str = 'float16'):
        """
        Initialize the logits extractor with a model
        
        Args:
            model_path: Path to the model
            device: Device to load the model on
            torch_dtype: Torch data type for the model
        """
        self.model_path = model_path
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded successfully!")
    
    def extract_logits(self, text: str, max_length: int = 2048) -> Tuple[np.ndarray, List[str]]:
        """
        Extract logits for a given text
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (logits, tokens)
        """
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=max_length, 
            truncation=True,
            padding=False
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get tokens for reference
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0].float().cpu().numpy()  # Convert to float32 before numpy
        
        return logits, tokens
    
    def find_target_token_positions(self, tokens: List[str], target_tokens: List[str]) -> Dict[str, List[int]]:
        """
        Find positions of target tokens in the token sequence
        
        Args:
            tokens: List of tokens
            target_tokens: List of target tokens to find
            
        Returns:
            Dictionary mapping token to list of positions
        """
        positions = {token: [] for token in target_tokens}
        
        for i, token in enumerate(tokens):
            # Handle different tokenization formats
            cleaned_token = token.replace('Ġ', '').replace('▁', '').lower().strip()
            for target in target_tokens:
                if cleaned_token == target.lower() or token.lower() == target.lower():
                    positions[target].append(i)
        
        return positions
    
    def extract_context_logits(self, logits: np.ndarray, positions: List[int], 
                             context_window: int = 10) -> np.ndarray:
        """
        Extract logits around target token positions
        
        Args:
            logits: Full logits array
            positions: List of target token positions
            context_window: Number of tokens before and after to include
            
        Returns:
            Context logits array
        """
        context_logits = []
        seq_len = logits.shape[0]
        
        for pos in positions:
            start = max(0, pos - context_window)
            end = min(seq_len, pos + context_window + 1)
            context_logits.append(logits[start:end])
        
        return context_logits
    
    def get_token_probabilities(self, logits: np.ndarray, token_ids: List[int]) -> np.ndarray:
        """
        Get probabilities for specific tokens
        
        Args:
            logits: Logits array (seq_len, vocab_size)
            token_ids: List of token IDs to get probabilities for
            
        Returns:
            Probabilities array (seq_len, len(token_ids))
        """
        # Apply softmax to get probabilities
        probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        
        # Extract probabilities for target tokens
        target_probs = probabilities[:, token_ids]
        
        return target_probs
    
    def batch_extract_logits(self, texts: List[str], max_length: int = 2048, 
                           batch_size: int = 4) -> List[Tuple[np.ndarray, List[str]]]:
        """
        Extract logits for multiple texts in batches
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            
        Returns:
            List of (logits, tokens) tuples
        """
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting logits"):
            batch_texts = texts[i:i+batch_size]
            
            for text in batch_texts:
                logits, tokens = self.extract_logits(text, max_length)
                results.append((logits, tokens))
        
        return results
