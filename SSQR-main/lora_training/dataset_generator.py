import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import random
from collections import defaultdict
from rq_token_processor import RQTokenProcessor


class LoraDatasetGenerator:
    """
    LoRA微调数据集生成器
    将RQ-VAE生成的token语料转换为适合LLaMA2 LoRA微调的数据格式
    """
    
    def __init__(self, 
                 rq_codes_path: str,
                 entity_text_path: str,
                 relation_text_path: str,
                 kg_data_path: str,
                 output_dir: str = "lora_data"):
        """
        初始化数据集生成器
        
        Args:
            rq_codes_path: RQ代码文件路径
            entity_text_path: 实体文本文件路径
            relation_text_path: 关系文本文件路径
            kg_data_path: 知识图谱数据路径
            output_dir: 输出目录
        """
        self.rq_codes_path = rq_codes_path
        self.entity_text_path = entity_text_path
        self.relation_text_path = relation_text_path
        self.kg_data_path = kg_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化token处理器
        self.token_processor = RQTokenProcessor()
        
        # 加载数据
        self.rq_codes = self._load_rq_codes()
        self.entity_texts = self._load_entity_texts()
        self.relation_texts = self._load_relation_texts()
        self.kg_triplets = self._load_kg_triplets()
        
        print(f"Loaded {len(self.rq_codes)} entity codes")
        print(f"Loaded {len(self.entity_texts)} entity texts")
        print(f"Loaded {len(self.relation_texts)} relation texts")
        print(f"Loaded {len(self.kg_triplets)} knowledge triplets")
    
    def _load_rq_codes(self) -> np.ndarray:
        """加载RQ代码"""
        codes = torch.load(self.rq_codes_path).numpy()
        print(f"RQ codes shape: {codes.shape}")
        return codes
    
    def _load_entity_texts(self) -> List[str]:
        """加载实体文本"""
        entity_texts = []
        with open(self.entity_text_path, 'r', encoding='utf-8') as f:
            for line in f:
                entity_texts.append(line.strip())
        return entity_texts
    
    def _load_relation_texts(self) -> List[str]:
        """加载关系文本"""
        relation_texts = []
        with open(self.relation_text_path, 'r', encoding='utf-8') as f:
            for line in f:
                relation_texts.append(line.strip())
        return relation_texts
    
    def _load_kg_triplets(self) -> List[Tuple[int, int, int]]:
        """加载知识图谱三元组"""
        triplets = []
        with open(self.kg_data_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    h, r, t = map(int, parts)
                    triplets.append((h, r, t))
        return triplets
    
    def generate_entity_representation_tasks(self, 
                                           num_samples: int = 10000,
                                           task_types: List[str] = None) -> List[Dict]:
        """
        生成实体表示学习任务
        
        Args:
            num_samples: 样本数量
            task_types: 任务类型列表
            
        Returns:
            任务数据列表
        """
        if task_types is None:
            task_types = ["generation", "understanding", "reasoning"]
        
        tasks = []
        num_entities = len(self.rq_codes)
        
        for _ in range(num_samples):
            entity_id = random.randint(0, num_entities - 1)
            entity_name = self.entity_texts[entity_id]
            rq_codes = self.rq_codes[entity_id]
            task_type = random.choice(task_types)
            
            # 生成提示
            prompt = self.token_processor.create_training_prompt(
                entity_name, rq_codes, task_type
            )
            
            # 生成回答
            if task_type == "generation":
                answer = self.token_processor.rq_codes_to_text(rq_codes, entity_name)
            elif task_type == "understanding":
                answer = f"The quantized representation of '{entity_name}' consists of {rq_codes.shape[0]} sequence positions, each with {rq_codes.shape[1]} quantizer levels. This representation captures the entity's semantic properties in a compressed format."
            elif task_type == "reasoning":
                answer = f"Based on the quantized representation, '{entity_name}' appears to be a complex entity with multi-level semantic features. The representation suggests rich contextual information encoded across multiple quantization levels."
            else:
                answer = self.token_processor.rq_codes_to_text(rq_codes, entity_name)
            
            tasks.append({
                "instruction": prompt,
                "input": "",
                "output": answer,
                "task_type": task_type,
                "entity_id": entity_id
            })
        
        return tasks
    
    def generate_knowledge_reasoning_tasks(self, 
                                         num_samples: int = 5000,
                                         task_types: List[str] = None) -> List[Dict]:
        """
        生成知识推理任务
        
        Args:
            num_samples: 样本数量
            task_types: 任务类型列表
            
        Returns:
            任务数据列表
        """
        if task_types is None:
            task_types = ["prediction", "validation", "completion"]
        
        tasks = []
        
        for _ in range(num_samples):
            # 随机选择一个三元组
            triplet = random.choice(self.kg_triplets)
            h_id, r_id, t_id = triplet
            
            # 获取实体和关系文本
            head_entity = self.entity_texts[h_id]
            relation = self.relation_texts[r_id]
            tail_entity = self.entity_texts[t_id]
            
            # 获取RQ代码
            head_codes = self.rq_codes[h_id]
            tail_codes = self.rq_codes[t_id]
            
            task_type = random.choice(task_types)
            
            # 生成提示和回答
            if task_type == "prediction":
                prompt = f"Given the head entity and relation, predict the tail entity:\n{self.token_processor.rq_codes_to_text(head_codes, head_entity)}\nRelation: {relation}"
                answer = f"Based on the head entity '{head_entity}' and relation '{relation}', the predicted tail entity is '{tail_entity}'."
            elif task_type == "validation":
                prompt = f"Validate the knowledge triplet:\n{self.token_processor.rq_codes_to_text(head_codes, head_entity)}\nRelation: {relation}\n{self.token_processor.rq_codes_to_text(tail_codes, tail_entity)}\nIs this triplet valid?"
                answer = "Yes, this knowledge triplet is valid based on the quantized representations and the given relation."
            elif task_type == "completion":
                prompt = f"Complete the knowledge triplet:\n{self.token_processor.rq_codes_to_text(head_codes, head_entity)}\nRelation: {relation}"
                answer = f"The completed triplet is: {head_entity} --[{relation}]--> {tail_entity}"
            else:
                prompt = f"Knowledge triplet:\n{self.token_processor.rq_codes_to_text(head_codes, head_entity)}\nRelation: {relation}\n{self.token_processor.rq_codes_to_text(tail_codes, tail_entity)}"
                answer = f"This triplet represents the relationship between '{head_entity}' and '{tail_entity}' through the relation '{relation}'."
            
            tasks.append({
                "instruction": prompt,
                "input": "",
                "output": answer,
                "task_type": task_type,
                "triplet": triplet
            })
        
        return tasks
    
    def generate_code_understanding_tasks(self, 
                                        num_samples: int = 3000) -> List[Dict]:
        """
        生成代码理解任务
        
        Args:
            num_samples: 样本数量
            
        Returns:
            任务数据列表
        """
        tasks = []
        num_entities = len(self.rq_codes)
        
        for _ in range(num_samples):
            entity_id = random.randint(0, num_entities - 1)
            entity_name = self.entity_texts[entity_id]
            rq_codes = self.rq_codes[entity_id]
            
            # 生成不同类型的代码理解任务
            task_variants = [
                {
                    "prompt": f"Explain the structure of this quantized representation for '{entity_name}':\n{self.token_processor.rq_codes_to_text(rq_codes)}",
                    "answer": f"The quantized representation for '{entity_name}' has {rq_codes.shape[0]} sequence positions and {rq_codes.shape[1]} quantization levels. Each position contains multiple quantizer codes that together encode the entity's semantic information."
                },
                {
                    "prompt": f"What does each quantizer level represent in this code for '{entity_name}':\n{self.token_processor.rq_codes_to_text(rq_codes)}",
                    "answer": f"Each quantizer level in the representation of '{entity_name}' captures different aspects of the entity's semantic information. Level 0 typically represents basic features, while higher levels capture more complex and abstract properties."
                },
                {
                    "prompt": f"How would you interpret the sequence structure in this representation of '{entity_name}':\n{self.token_processor.rq_codes_to_text(rq_codes)}",
                    "answer": f"The sequence structure in '{entity_name}'s representation follows a hierarchical pattern where each position contributes to the overall semantic understanding of the entity, with quantizer levels providing multi-resolution encoding."
                }
            ]
            
            variant = random.choice(task_variants)
            
            tasks.append({
                "instruction": variant["prompt"],
                "input": "",
                "output": variant["answer"],
                "task_type": "code_understanding",
                "entity_id": entity_id
            })
        
        return tasks
    
    def generate_all_tasks(self, 
                          entity_samples: int = 10000,
                          knowledge_samples: int = 5000,
                          code_samples: int = 3000) -> List[Dict]:
        """
        生成所有类型的任务
        
        Args:
            entity_samples: 实体表示任务样本数
            knowledge_samples: 知识推理任务样本数
            code_samples: 代码理解任务样本数
            
        Returns:
            所有任务数据
        """
        print("Generating entity representation tasks...")
        entity_tasks = self.generate_entity_representation_tasks(entity_samples)
        
        print("Generating knowledge reasoning tasks...")
        knowledge_tasks = self.generate_knowledge_reasoning_tasks(knowledge_samples)
        
        print("Generating code understanding tasks...")
        code_tasks = self.generate_code_understanding_tasks(code_samples)
        
        all_tasks = entity_tasks + knowledge_tasks + code_tasks
        random.shuffle(all_tasks)
        
        print(f"Generated {len(all_tasks)} total tasks")
        return all_tasks
    
    def save_tasks(self, tasks: List[Dict], split_ratio: float = 0.8):
        """
        保存任务数据
        
        Args:
            tasks: 任务数据列表
            split_ratio: 训练集比例
        """
        # 分割数据
        split_idx = int(len(tasks) * split_ratio)
        train_tasks = tasks[:split_idx]
        val_tasks = tasks[split_idx:]
        
        # 保存训练集
        train_file = self.output_dir / "train.json"
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_tasks, f, ensure_ascii=False, indent=2)
        
        # 保存验证集
        val_file = self.output_dir / "validation.json"
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_tasks, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(train_tasks)} training tasks to {train_file}")
        print(f"Saved {len(val_tasks)} validation tasks to {val_file}")
        
        # 保存统计信息
        stats = {
            "total_tasks": len(tasks),
            "train_tasks": len(train_tasks),
            "validation_tasks": len(val_tasks),
            "task_types": {}
        }
        
        # 统计任务类型
        for task in tasks:
            task_type = task.get("task_type", "unknown")
            stats["task_types"][task_type] = stats["task_types"].get(task_type, 0) + 1
        
        stats_file = self.output_dir / "dataset_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"Saved dataset statistics to {stats_file}")
    
    def create_alpaca_format(self, tasks: List[Dict]) -> List[Dict]:
        """
        转换为Alpaca格式
        
        Args:
            tasks: 任务数据列表
            
        Returns:
            Alpaca格式数据
        """
        alpaca_data = []
        
        for task in tasks:
            alpaca_item = {
                "instruction": task["instruction"],
                "input": task.get("input", ""),
                "output": task["output"]
            }
            alpaca_data.append(alpaca_item)
        
        return alpaca_data


# 使用示例
if __name__ == "__main__":
    # 配置路径
    rq_codes_path = "codes_new/FB15K-237N_16_1024_4_rq.pt"
    entity_text_path = "data/FB15K-237N/entity2text.txt"
    relation_text_path = "data/FB15K-237N/relation2id.txt"
    kg_data_path = "data/FB15K-237N/train2id.txt"
    
    # 创建数据集生成器
    generator = LoraDatasetGenerator(
        rq_codes_path=rq_codes_path,
        entity_text_path=entity_text_path,
        relation_text_path=relation_text_path,
        kg_data_path=kg_data_path,
        output_dir="lora_data"
    )
    
    # 生成所有任务
    all_tasks = generator.generate_all_tasks(
        entity_samples=5000,
        knowledge_samples=3000,
        code_samples=2000
    )
    
    # 保存数据
    generator.save_tasks(all_tasks)
    
    # 创建Alpaca格式
    alpaca_data = generator.create_alpaca_format(all_tasks)
    alpaca_file = generator.output_dir / "alpaca_format.json"
    with open(alpaca_file, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved Alpaca format data to {alpaca_file}")
