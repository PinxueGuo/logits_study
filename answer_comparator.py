"""
Answer extraction and comparison utilities for mathematical problems
"""

import re
import numpy as np
from fractions import Fraction
from typing import Optional, Union, Tuple
import sympy as sp
from sympy import sympify, simplify, N, latex
import warnings
warnings.filterwarnings('ignore')

class MathAnswerComparator:
    """数学答案提取和比较工具"""
    
    def __init__(self):
        pass
    
    def extract_boxed_answer(self, text: str) -> Optional[str]:
        """
        从文本中提取最后一个\\boxed{}中的内容
        
        Args:
            text: 包含答案的文本
            
        Returns:
            提取的答案字符串，如果没找到返回None
        """
        # 匹配 \\boxed{...} 模式
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        
        if matches:
            return matches[-1].strip()  # 返回最后一个匹配
        
        # 如果没有找到\\boxed，尝试其他常见模式
        # 匹配 $$...$$
        pattern = r'\$\$([^$]+)\$\$'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        
        # 匹配 $...$
        pattern = r'\$([^$]+)\$'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        
        return None
    
    def normalize_answer(self, answer: str) -> str:
        """
        标准化答案格式
        
        Args:
            answer: 原始答案字符串
            
        Returns:
            标准化后的答案
        """
        if not answer:
            return ""
        
        # 移除常见的修饰词
        answer = answer.strip()
        
        # 移除单位词（如"种"、"个"等）
        answer = re.sub(r'[种个项次元度]$', '', answer)
        
        # 移除空格
        answer = re.sub(r'\s+', '', answer)
        
        # 处理中文数字
        chinese_nums = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5', 
                       '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'}
        for ch, num in chinese_nums.items():
            answer = answer.replace(ch, num)
        
        return answer
    
    def parse_mathematical_expression(self, expr: str) -> Optional[Union[float, int, str]]:
        """
        解析数学表达式
        
        Args:
            expr: 数学表达式字符串
            
        Returns:
            解析后的数值或符号表达式
        """
        expr = self.normalize_answer(expr)
        
        if not expr:
            return None
        
        try:
            # 尝试作为分数解析
            if '/' in expr:
                parts = expr.split('/')
                if len(parts) == 2:
                    try:
                        frac = Fraction(int(parts[0]), int(parts[1]))
                        return float(frac)
                    except:
                        pass
            
            # 尝试作为sympy表达式解析
            try:
                # 预处理一些常见的数学符号
                expr_processed = expr.replace('π', 'pi').replace('π', 'pi')
                expr_processed = expr_processed.replace('∞', 'oo')
                expr_processed = expr_processed.replace('×', '*')
                expr_processed = expr_processed.replace('÷', '/')
                
                sympy_expr = sympify(expr_processed)
                
                # 如果是数值，返回浮点数
                if sympy_expr.is_number:
                    return float(N(sympy_expr))
                else:
                    # 返回简化后的符号表达式
                    return str(simplify(sympy_expr))
            except:
                pass
            
            # 尝试直接作为数字解析
            try:
                return float(expr)
            except:
                pass
            
            # 尝试作为整数解析
            try:
                return int(expr)
            except:
                pass
            
            # 如果都失败了，返回原字符串
            return expr
            
        except Exception as e:
            return expr
    
    def compare_answers(self, predicted: str, ground_truth: str, tolerance: float = 1e-6) -> Tuple[bool, str]:
        """
        比较两个答案是否等价
        
        Args:
            predicted: 预测答案
            ground_truth: 标准答案
            tolerance: 数值比较的容差
            
        Returns:
            (是否相等, 比较说明)
        """
        # 提取预测答案中的boxed内容
        extracted_pred = self.extract_boxed_answer(predicted)
        if extracted_pred is None:
            # 如果没有boxed，尝试提取最后一个数字或表达式
            extracted_pred = self._extract_final_answer(predicted)
        
        if extracted_pred is None:
            return False, "无法从预测答案中提取有效答案"
        
        # 解析两个答案
        pred_parsed = self.parse_mathematical_expression(extracted_pred)
        gt_parsed = self.parse_mathematical_expression(ground_truth)
        
        if pred_parsed is None or gt_parsed is None:
            return False, f"解析失败: pred={pred_parsed}, gt={gt_parsed}"
        
        # 如果都是数字，进行数值比较
        if isinstance(pred_parsed, (int, float)) and isinstance(gt_parsed, (int, float)):
            is_equal = abs(pred_parsed - gt_parsed) < tolerance
            return is_equal, f"数值比较: {pred_parsed} vs {gt_parsed}"
        
        # 如果都是字符串，进行字符串比较
        if isinstance(pred_parsed, str) and isinstance(gt_parsed, str):
            # 标准化字符串
            pred_norm = self.normalize_answer(pred_parsed.lower())
            gt_norm = self.normalize_answer(gt_parsed.lower())
            
            is_equal = pred_norm == gt_norm
            
            # 如果字符串不相等，尝试符号比较
            if not is_equal:
                try:
                    pred_sympy = sympify(pred_parsed)
                    gt_sympy = sympify(gt_parsed)
                    is_equal = simplify(pred_sympy - gt_sympy) == 0
                except:
                    pass
            
            return is_equal, f"符号比较: {pred_parsed} vs {gt_parsed}"
        
        # 混合类型比较
        try:
            # 尝试将字符串转换为数值进行比较
            if isinstance(pred_parsed, str):
                pred_num = float(sympify(pred_parsed))
            else:
                pred_num = float(pred_parsed)
            
            if isinstance(gt_parsed, str):
                gt_num = float(sympify(gt_parsed))
            else:
                gt_num = float(gt_parsed)
            
            is_equal = abs(pred_num - gt_num) < tolerance
            return is_equal, f"混合比较: {pred_num} vs {gt_num}"
            
        except:
            return False, f"类型不匹配: {type(pred_parsed)} vs {type(gt_parsed)}"
    
    def _extract_final_answer(self, text: str) -> Optional[str]:
        """
        从文本中提取最终答案（当没有\\boxed时的备选方法）
        """
        # 寻找常见的答案指示词后的内容
        answer_indicators = [
            r'答案是[:：]?\s*([^\s\n。，,]+)',
            r'答案为[:：]?\s*([^\s\n。，,]+)',
            r'结果是[:：]?\s*([^\s\n。，,]+)', 
            r'结果为[:：]?\s*([^\s\n。，,]+)',
            r'所以[:：]?\s*([^\s\n。，,]+)',
            r'因此[:：]?\s*([^\s\n。，,]+)',
            r'故[:：]?\s*([^\s\n。，,]+)',
        ]
        
        for pattern in answer_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1].strip()
        
        # 寻找数字模式
        number_patterns = [
            r'(\d+(?:\.\d+)?(?:/\d+)?)',  # 数字、小数、分数
            r'([+-]?\d*\.?\d*[a-zA-Z]*)',  # 带字母的数字表达式
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # 返回最后一个非空匹配
                valid_matches = [m for m in matches if m.strip()]
                if valid_matches:
                    return valid_matches[-1].strip()
        
        return None

# 创建全局实例
math_comparator = MathAnswerComparator()

def extract_and_compare_answer(predicted_text: str, ground_truth: str) -> Tuple[bool, str, str]:
    """
    便捷函数：提取并比较答案
    
    Args:
        predicted_text: 包含预测答案的文本
        ground_truth: 标准答案
        
    Returns:
        (是否正确, 提取的答案, 比较说明)
    """
    extracted = math_comparator.extract_boxed_answer(predicted_text)
    if extracted is None:
        extracted = math_comparator._extract_final_answer(predicted_text)
    
    if extracted is None:
        return False, "", "无法提取答案"
    
    is_correct, explanation = math_comparator.compare_answers(predicted_text, ground_truth)
    
    return is_correct, extracted, explanation

# 测试函数
def test_answer_extraction():
    """测试答案提取和比较功能"""
    test_cases = [
        ("因此答案是 \\boxed{165种}", "165", True),
        ("所以结果为 \\boxed{-4}", "-4", True),
        ("计算得到 \\boxed{5/2}", "5/2", True),
        ("最终答案是 \\boxed{10}", "10", True),
        ("体积的最大值为 \\boxed{4/3}", "4/3", True),
        ("答案是12", "12", True),
        ("结果为 $\\frac{3}{4}$", "3/4", True),
    ]
    
    print("=== 答案提取和比较测试 ===")
    for i, (pred_text, gt, expected) in enumerate(test_cases):
        is_correct, extracted, explanation = extract_and_compare_answer(pred_text, gt)
        status = "✅" if is_correct == expected else "❌"
        print(f"{status} 测试 {i+1}: 提取='{extracted}', 正确={is_correct}, 说明={explanation}")

if __name__ == "__main__":
    test_answer_extraction()
