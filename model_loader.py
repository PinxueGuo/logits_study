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
    
    def generate_with_logits(self, input_text: str, max_new_tokens: int = 512) -> Tuple[str, np.ndarray, List[str]]:
        """
        Generate text and extract logits for each generated token
        
        Args:
            input_text: Input text (prompt/question)
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Tuple of (generated_text, logits_array, generated_tokens)
            - generated_text: The generated response text
            - logits_array: numpy array of shape (num_generated_tokens, vocab_size)
            - generated_tokens: List of generated token strings
        """
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            # Generate with logits
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                output_scores=True,           # 返回每个生成token的logits
                return_dict_in_generate=True, # 返回字典格式
                do_sample=False,              # 贪婪解码确保确定性
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 获取生成的token序列 (去掉输入部分)
        generated_token_ids = outputs.sequences[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
        # 获取每个生成token的logits
        scores = outputs.scores  # tuple of tensors, 每个tensor shape: [batch_size, vocab_size]
        
        if len(scores) == 0:
            return generated_text, np.array([]), []
        
        # 将logits转换为numpy数组
        logits_list = []
        generated_tokens = []
        
        for i, score in enumerate(scores):
            if i < len(generated_token_ids):  # 确保索引不越界
                # 获取当前token的logits
                token_logits = score[0].float().cpu().numpy()  # shape: (vocab_size,)
                logits_list.append(token_logits)
                
                # 获取对应的token文本
                token_id = generated_token_ids[i].item()
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                generated_tokens.append(token_text)
        
        # 转换为numpy数组
        logits_array = np.array(logits_list) if logits_list else np.array([])
        
        return generated_text, logits_array, generated_tokens
    
    def calculate_token_entropy(self, logits_array: np.ndarray) -> np.ndarray:
        """
        Calculate entropy for each generated token from its logits
        
        Args:
            logits_array: numpy array of shape (num_tokens, vocab_size)
            
        Returns:
            Entropy array of shape (num_tokens,) - entropy for each generated token
        """
        if logits_array.size == 0:
            return np.array([])
        
        # Convert to tensor for softmax calculation
        logits_tensor = torch.tensor(logits_array, dtype=torch.float32)
        
        # Apply softmax to get probability distribution for each token
        probabilities = torch.softmax(logits_tensor, dim=-1)  # shape: (num_tokens, vocab_size)
        
        # Calculate entropy: H = -sum(p * log(p)) for each token
        epsilon = 1e-12
        probabilities = torch.clamp(probabilities, min=epsilon, max=1.0)
        
        # Calculate entropy for each token position
        log_probs = torch.log(probabilities)
        entropy = -torch.sum(probabilities * log_probs, dim=-1)  # shape: (num_tokens,)
        
        return entropy.numpy()
    
    def analyze_generation_uncertainty(self, input_text: str, max_new_tokens: int = 512) -> Dict:
        """
        Generate text and analyze the uncertainty (entropy) for each generated token
        
        Args:
            input_text: Input prompt/question
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary containing:
            - generated_text: The generated response
            - tokens: List of generated token strings  
            - entropies: Entropy for each generated token
            - logits: Raw logits for each generated token
        """
        # Generate text and get logits
        generated_text, logits_array, generated_tokens = self.generate_with_logits(
            input_text, max_new_tokens
        )
        
        # Calculate entropy for each token
        entropies = self.calculate_token_entropy(logits_array)
        
        return {
            'generated_text': generated_text,
            'tokens': generated_tokens,
            'entropies': entropies,
            'logits': logits_array,
            'input_text': input_text
        }
    
    def generate_answer(self, text: str, max_new_tokens: int = 512) -> str:
        """
        Simple text generation method (deprecated - use analyze_generation_uncertainty instead)
        
        Args:
            text: Input text (question)
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated answer text
        """
        generated_text, _, _ = self.generate_with_logits(text, max_new_tokens)
        return generated_text
