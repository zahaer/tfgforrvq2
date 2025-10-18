import torch
import json
import argparse
from typing import List, Dict, Optional
from pathlib import Path
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rq_token_processor import RQTokenProcessor
import numpy as np


logger = logging.getLogger(__name__)


class RQLoraInference:
    """
    RQ-VAE LoRA推理类
    """
    
    def __init__(self, 
                 base_model_name: str = "meta-llama/Llama-2-7b-hf",
                 lora_adapter_path: str = None,
                 device: str = "auto"):
        """
        初始化推理器
        
        Args:
            base_model_name: 基础模型名称
            lora_adapter_path: LoRA适配器路径
            device: 设备
        """
        self.base_model_name = base_model_name
        self.lora_adapter_path = lora_adapter_path
        self.device = device
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.rq_processor = None
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        logger.info(f"Loading base model: {self.base_model_name}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )
        
        # 加载LoRA适配器
        if self.lora_adapter_path:
            logger.info(f"Loading LoRA adapter: {self.lora_adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path)
        
        # 初始化RQ处理器
        self.rq_processor = RQTokenProcessor(self.base_model_name)
        
        logger.info("Model loading completed")
    
    def generate_response(self, 
                         prompt: str,
                         max_length: int = 256,
                         temperature: float = 0.7,
                         top_p: float = 0.9,
                         top_k: int = 50,
                         repetition_penalty: float = 1.1) -> str:
        """
        生成回答
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top-p参数
            top_k: top-k参数
            repetition_penalty: 重复惩罚
            
        Returns:
            生成的回答
        """
        # Tokenize输入
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 移除输入部分
        response = response[len(prompt):].strip()
        
        return response
    
    def test_entity_understanding(self, 
                                entity_name: str,
                                rq_codes: np.ndarray,
                                task_type: str = "understanding") -> Dict:
        """
        测试实体理解能力
        
        Args:
            entity_name: 实体名称
            rq_codes: RQ代码
            task_type: 任务类型
            
        Returns:
            测试结果
        """
        # 生成提示
        prompt = self.rq_processor.create_training_prompt(
            entity_name, rq_codes, task_type
        )
        
        # 生成回答
        response = self.generate_response(prompt)
        
        return {
            "entity_name": entity_name,
            "task_type": task_type,
            "prompt": prompt,
            "response": response,
            "rq_codes": rq_codes.tolist()
        }
    
    def test_knowledge_reasoning(self,
                               head_entity: str,
                               relation: str,
                               tail_entity: str,
                               head_codes: np.ndarray,
                               tail_codes: np.ndarray,
                               task: str = "prediction") -> Dict:
        """
        测试知识推理能力
        
        Args:
            head_entity: 头实体
            relation: 关系
            tail_entity: 尾实体
            head_codes: 头实体RQ代码
            tail_codes: 尾实体RQ代码
            task: 任务类型
            
        Returns:
            测试结果
        """
        # 生成提示
        prompt = self.rq_processor.create_knowledge_triplet_prompt(
            head_entity, relation, tail_entity, head_codes, tail_codes, task
        )
        
        # 生成回答
        response = self.generate_response(prompt)
        
        return {
            "head_entity": head_entity,
            "relation": relation,
            "tail_entity": tail_entity,
            "task": task,
            "prompt": prompt,
            "response": response,
            "head_codes": head_codes.tolist(),
            "tail_codes": tail_codes.tolist()
        }
    
    def batch_test(self, test_data: List[Dict]) -> List[Dict]:
        """
        批量测试
        
        Args:
            test_data: 测试数据列表
            
        Returns:
            测试结果列表
        """
        results = []
        
        for i, test_item in enumerate(test_data):
            logger.info(f"Testing item {i+1}/{len(test_data)}")
            
            try:
                if test_item["type"] == "entity_understanding":
                    result = self.test_entity_understanding(
                        entity_name=test_item["entity_name"],
                        rq_codes=np.array(test_item["rq_codes"]),
                        task_type=test_item["task_type"]
                    )
                elif test_item["type"] == "knowledge_reasoning":
                    result = self.test_knowledge_reasoning(
                        head_entity=test_item["head_entity"],
                        relation=test_item["relation"],
                        tail_entity=test_item["tail_entity"],
                        head_codes=np.array(test_item["head_codes"]),
                        tail_codes=np.array(test_item["tail_codes"]),
                        task=test_item["task"]
                    )
                else:
                    logger.warning(f"Unknown test type: {test_item['type']}")
                    continue
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error testing item {i+1}: {e}")
                continue
        
        return results
    
    def evaluate_code_extraction(self, test_data: List[Dict]) -> Dict:
        """
        评估代码提取能力
        
        Args:
            test_data: 测试数据
            
        Returns:
            评估结果
        """
        total_tests = 0
        successful_extractions = 0
        extraction_errors = []
        
        for test_item in test_data:
            try:
                # 生成回答
                if test_item["type"] == "entity_understanding":
                    prompt = self.rq_processor.create_training_prompt(
                        test_item["entity_name"],
                        np.array(test_item["rq_codes"]),
                        test_item["task_type"]
                    )
                else:
                    continue
                
                response = self.generate_response(prompt)
                
                # 尝试提取RQ代码
                extracted_codes = self.rq_processor.extract_rq_codes_from_text(response)
                
                total_tests += 1
                
                if extracted_codes is not None:
                    successful_extractions += 1
                else:
                    extraction_errors.append({
                        "entity": test_item["entity_name"],
                        "response": response
                    })
                    
            except Exception as e:
                logger.error(f"Error in code extraction test: {e}")
                continue
        
        accuracy = successful_extractions / total_tests if total_tests > 0 else 0
        
        return {
            "total_tests": total_tests,
            "successful_extractions": successful_extractions,
            "accuracy": accuracy,
            "extraction_errors": extraction_errors
        }


def create_test_data(rq_codes_path: str, 
                    entity_text_path: str,
                    num_samples: int = 100) -> List[Dict]:
    """
    创建测试数据
    
    Args:
        rq_codes_path: RQ代码文件路径
        entity_text_path: 实体文本文件路径
        num_samples: 样本数量
        
    Returns:
        测试数据列表
    """
    # 加载RQ代码
    rq_codes = torch.load(rq_codes_path).numpy()
    
    # 加载实体文本
    entity_texts = []
    with open(entity_text_path, 'r', encoding='utf-8') as f:
        for line in f:
            entity_texts.append(line.strip())
    
    # 创建测试数据
    test_data = []
    num_entities = len(rq_codes)
    
    for i in range(min(num_samples, num_entities)):
        entity_id = i
        entity_name = entity_texts[entity_id]
        rq_codes_entity = rq_codes[entity_id]
        
        # 实体理解任务
        test_data.append({
            "type": "entity_understanding",
            "entity_name": entity_name,
            "rq_codes": rq_codes_entity.tolist(),
            "task_type": "understanding"
        })
        
        # 代码生成任务
        test_data.append({
            "type": "entity_understanding",
            "entity_name": entity_name,
            "rq_codes": rq_codes_entity.tolist(),
            "task_type": "generation"
        })
    
    return test_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RQ-VAE LoRA Inference")
    
    # 模型参数
    parser.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf", help="Base model name")
    parser.add_argument("--lora_adapter", help="LoRA adapter path")
    parser.add_argument("--device", default="auto", help="Device")
    
    # 测试参数
    parser.add_argument("--test_data", help="Test data path")
    parser.add_argument("--rq_codes", help="RQ codes path")
    parser.add_argument("--entity_texts", help="Entity texts path")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of test samples")
    
    # 生成参数
    parser.add_argument("--max_length", type=int, default=256, help="Max generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p")
    
    # 输出参数
    parser.add_argument("--output", default="inference_results.json", help="Output file")
    
    args = parser.parse_args()
    
    # 创建推理器
    inference = RQLoraInference(
        base_model_name=args.base_model,
        lora_adapter_path=args.lora_adapter,
        device=args.device
    )
    
    # 创建测试数据
    if args.test_data:
        with open(args.test_data, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    else:
        test_data = create_test_data(
            rq_codes_path=args.rq_codes,
            entity_text_path=args.entity_texts,
            num_samples=args.num_samples
        )
    
    # 批量测试
    results = inference.batch_test(test_data)
    
    # 评估代码提取能力
    extraction_eval = inference.evaluate_code_extraction(test_data)
    
    # 保存结果
    output_data = {
        "test_results": results,
        "extraction_evaluation": extraction_eval,
        "test_config": {
            "base_model": args.base_model,
            "lora_adapter": args.lora_adapter,
            "num_samples": len(test_data),
            "generation_params": {
                "max_length": args.max_length,
                "temperature": args.temperature,
                "top_p": args.top_p
            }
        }
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {args.output}")
    logger.info(f"Code extraction accuracy: {extraction_eval['accuracy']:.2%}")


if __name__ == "__main__":
    main()
