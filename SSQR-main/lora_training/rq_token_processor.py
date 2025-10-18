import torch
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer
import re


class RQTokenProcessor:
    """
    RQ-VAE Token处理器，将RQ代码转换为适合LLaMA2的token格式
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-hf",
                 special_tokens: Optional[Dict[str, str]] = None):
        """
        初始化RQ Token处理器
        
        Args:
            model_name: LLaMA2模型名称
            special_tokens: 特殊token字典
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 定义特殊token
        self.special_tokens = special_tokens or {
            'entity_start': '<ENT>',
            'entity_end': '</ENT>',
            'code_start': '<CODE>',
            'code_end': '</CODE>',
            'quantizer_sep': '|',
            'level_sep': ':'
        }
        
        # 添加特殊token到tokenizer
        self._add_special_tokens()
        
        # 创建token映射
        self.token_mappings = self._create_token_mappings()
    
    def _add_special_tokens(self):
        """添加特殊token到tokenizer"""
        special_tokens_list = list(self.special_tokens.values())
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens_list
        })
    
    def _create_token_mappings(self) -> Dict[str, int]:
        """创建特殊token的映射"""
        mappings = {}
        for token_name, token_str in self.special_tokens.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token_str)
            mappings[token_name] = token_id
        return mappings
    
    def rq_codes_to_text(self, rq_codes: np.ndarray, entity_name: str = "") -> str:
        """
        将RQ代码转换为文本格式
        
        Args:
            rq_codes: RQ代码数组 [seq_len, num_quantizers]
            entity_name: 实体名称（可选）
            
        Returns:
            格式化的文本字符串
        """
        if len(rq_codes.shape) != 2:
            raise ValueError(f"Expected 2D array, got shape {rq_codes.shape}")
        
        seq_len, num_quantizers = rq_codes.shape
        
        # 构建代码字符串
        code_parts = []
        for i in range(seq_len):
            level_codes = []
            for j in range(num_quantizers):
                level_codes.append(f"Q{j}_{rq_codes[i, j]}")
            code_parts.append(self.special_tokens['level_sep'].join(level_codes))
        
        code_str = self.special_tokens['quantizer_sep'].join(code_parts)
        
        # 构建完整文本
        if entity_name:
            text = f"{self.special_tokens['entity_start']}{entity_name}{self.special_tokens['entity_end']} {self.special_tokens['code_start']}{code_str}{self.special_tokens['code_end']}"
        else:
            text = f"{self.special_tokens['code_start']}{code_str}{self.special_tokens['code_end']}"
        
        return text
    
    def create_training_prompt(self, 
                             entity_name: str, 
                             rq_codes: np.ndarray,
                             task_type: str = "generation") -> str:
        """
        创建训练提示
        
        Args:
            entity_name: 实体名称
            rq_codes: RQ代码
            task_type: 任务类型 ("generation", "understanding", "reasoning")
            
        Returns:
            训练提示文本
        """
        rq_text = self.rq_codes_to_text(rq_codes, entity_name)
        
        if task_type == "generation":
            prompt = f"Given the entity '{entity_name}', generate its quantized representation:\n{rq_text}"
        elif task_type == "understanding":
            prompt = f"Explain the quantized representation of entity '{entity_name}':\n{rq_text}"
        elif task_type == "reasoning":
            prompt = f"Based on the quantized representation, what can you infer about entity '{entity_name}'?\n{rq_text}"
        else:
            prompt = f"Entity: {entity_name}\nQuantized representation: {rq_text}"
        
        return prompt
    
    def create_knowledge_triplet_prompt(self,
                                      head_entity: str,
                                      relation: str,
                                      tail_entity: str,
                                      head_codes: np.ndarray,
                                      tail_codes: np.ndarray,
                                      task: str = "prediction") -> str:
        """
        创建知识三元组的训练提示
        
        Args:
            head_entity: 头实体
            relation: 关系
            tail_entity: 尾实体
            head_codes: 头实体RQ代码
            tail_codes: 尾实体RQ代码
            task: 任务类型 ("prediction", "validation", "completion")
            
        Returns:
            知识三元组提示文本
        """
        head_text = self.rq_codes_to_text(head_codes, head_entity)
        tail_text = self.rq_codes_to_text(tail_codes, tail_entity)
        
        if task == "prediction":
            prompt = f"Given the head entity and relation, predict the tail entity:\n{head_text}\nRelation: {relation}\nPredicted tail entity: {tail_entity}"
        elif task == "validation":
            prompt = f"Validate the knowledge triplet:\n{head_text}\nRelation: {relation}\n{tail_text}\nIs this triplet valid? Answer: Yes/No"
        elif task == "completion":
            prompt = f"Complete the knowledge triplet:\n{head_text}\nRelation: {relation}\nTail entity: {tail_entity}"
        else:
            prompt = f"Knowledge triplet:\n{head_text}\nRelation: {relation}\n{tail_text}"
        
        return prompt
    
    def tokenize_text(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        对文本进行tokenization
        
        Args:
            text: 输入文本
            max_length: 最大长度
            
        Returns:
            tokenized结果
        """
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
    
    def batch_tokenize(self, texts: List[str], max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        批量tokenization
        
        Args:
            texts: 文本列表
            max_length: 最大长度
            
        Returns:
            批量tokenized结果
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
    
    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """
        解码token ID为文本
        
        Args:
            token_ids: token ID张量
            
        Returns:
            解码后的文本
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
    
    def extract_rq_codes_from_text(self, text: str) -> Optional[np.ndarray]:
        """
        从文本中提取RQ代码
        
        Args:
            text: 包含RQ代码的文本
            
        Returns:
            提取的RQ代码数组，如果未找到则返回None
        """
        # 匹配代码模式: <CODE>Q0_123:Q1_456:Q2_789|Q0_234:Q1_567:Q2_890</CODE>
        code_pattern = r'<CODE>(.*?)</CODE>'
        match = re.search(code_pattern, text)
        
        if not match:
            return None
        
        code_str = match.group(1)
        
        # 解析代码
        try:
            # 按级别分隔符分割
            levels = code_str.split(self.special_tokens['quantizer_sep'])
            codes = []
            
            for level in levels:
                # 按量化器分隔符分割
                quantizers = level.split(self.special_tokens['level_sep'])
                level_codes = []
                
                for quantizer in quantizers:
                    # 提取数字部分
                    if quantizer.startswith('Q') and '_' in quantizer:
                        code_value = int(quantizer.split('_')[1])
                        level_codes.append(code_value)
                    else:
                        return None
                
                codes.append(level_codes)
            
            return np.array(codes)
        
        except (ValueError, IndexError):
            return None
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return len(self.tokenizer)
    
    def get_special_token_ids(self) -> Dict[str, int]:
        """获取特殊token的ID"""
        return self.token_mappings.copy()


# 使用示例
if __name__ == "__main__":
    # 初始化处理器
    processor = RQTokenProcessor()
    
    # 示例RQ代码 [seq_len=4, num_quantizers=3]
    sample_codes = np.array([
        [123, 456, 789],
        [234, 567, 890],
        [345, 678, 901],
        [456, 789, 012]
    ])
    
    # 转换为文本
    text = processor.rq_codes_to_text(sample_codes, "Barack Obama")
    print("Generated text:", text)
    
    # Tokenization
    tokens = processor.tokenize_text(text)
    print("Token IDs:", tokens['input_ids'])
    
    # 解码
    decoded = processor.decode_tokens(tokens['input_ids'])
    print("Decoded text:", decoded)
    
    # 提取代码
    extracted_codes = processor.extract_rq_codes_from_text(decoded)
    print("Extracted codes:", extracted_codes)
