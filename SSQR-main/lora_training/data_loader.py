import torch
import json
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Union
from transformers import AutoTokenizer
import random
from rq_token_processor import RQTokenProcessor


class LoraDataset(Dataset):
    """
    LoRA微调数据集类
    """
    
    def __init__(self, 
                 data: List[Dict],
                 tokenizer: AutoTokenizer,
                 max_length: int = 512,
                 instruction_template: str = None,
                 response_template: str = None):
        """
        初始化数据集
        
        Args:
            data: 数据列表
            tokenizer: 分词器
            max_length: 最大序列长度
            instruction_template: 指令模板
            response_template: 回答模板
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 默认模板
        self.instruction_template = instruction_template or "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        self.response_template = response_template or "{output}"
        
        # 设置特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建输入文本
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        # 格式化输入
        formatted_input = self.instruction_template.format(
            instruction=instruction,
            input=input_text
        )
        
        # 格式化输出
        formatted_output = self.response_template.format(output=output)
        
        # 完整文本
        full_text = formatted_input + formatted_output
        
        # Tokenization
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        )
        
        # 创建标签（用于计算损失）
        labels = encoding['input_ids'].clone()
        
        # 找到回答部分的开始位置
        response_start = len(self.tokenizer.encode(formatted_input, add_special_tokens=False))
        
        # 将指令部分的标签设为-100（不计算损失）
        labels[0, :response_start] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }


class LoraDataLoader:
    """
    LoRA数据加载器
    """
    
    def __init__(self, 
                 tokenizer: AutoTokenizer,
                 max_length: int = 512,
                 batch_size: int = 4,
                 num_workers: int = 4):
        """
        初始化数据加载器
        
        Args:
            tokenizer: 分词器
            max_length: 最大序列长度
            batch_size: 批次大小
            num_workers: 工作进程数
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def load_data(self, data_path: str) -> List[Dict]:
        """
        加载数据文件
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            数据列表
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def create_dataset(self, data: List[Dict]) -> LoraDataset:
        """
        创建数据集
        
        Args:
            data: 数据列表
            
        Returns:
            数据集对象
        """
        return LoraDataset(
            data=data,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
    
    def create_dataloader(self, 
                         dataset: LoraDataset,
                         shuffle: bool = True) -> DataLoader:
        """
        创建数据加载器
        
        Args:
            dataset: 数据集
            shuffle: 是否打乱数据
            
        Returns:
            数据加载器
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
    
    def collate_fn(self, batch):
        """
        批处理函数
        
        Args:
            batch: 批次数据
            
        Returns:
            批处理后的数据
        """
        # 获取最大长度
        max_len = max([item['input_ids'].size(0) for item in batch])
        
        # 初始化批次张量
        batch_size = len(batch)
        input_ids = torch.full((batch_size, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        # 填充数据
        for i, item in enumerate(batch):
            seq_len = item['input_ids'].size(0)
            input_ids[i, :seq_len] = item['input_ids']
            attention_mask[i, :seq_len] = item['attention_mask']
            labels[i, :seq_len] = item['labels']
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class RQLoraDataLoader(LoraDataLoader):
    """
    专门用于RQ-VAE token的LoRA数据加载器
    """
    
    def __init__(self, 
                 tokenizer: AutoTokenizer,
                 max_length: int = 512,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 rq_processor: Optional[RQTokenProcessor] = None):
        """
        初始化RQ LoRA数据加载器
        
        Args:
            tokenizer: 分词器
            max_length: 最大序列长度
            batch_size: 批次大小
            num_workers: 工作进程数
            rq_processor: RQ token处理器
        """
        super().__init__(tokenizer, max_length, batch_size, num_workers)
        self.rq_processor = rq_processor or RQTokenProcessor()
    
    def create_rq_dataset(self, data: List[Dict]) -> LoraDataset:
        """
        创建RQ数据集
        
        Args:
            data: 数据列表
            
        Returns:
            RQ数据集对象
        """
        return LoraDataset(
            data=data,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            instruction_template="### RQ-VAE Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            response_template="{output}"
        )
    
    def validate_rq_tokens(self, data: List[Dict]) -> List[Dict]:
        """
        验证RQ token的有效性
        
        Args:
            data: 数据列表
            
        Returns:
            验证后的数据列表
        """
        valid_data = []
        
        for item in data:
            try:
                # 检查是否包含RQ代码
                instruction = item.get("instruction", "")
                output = item.get("output", "")
                
                # 尝试提取RQ代码
                rq_codes = self.rq_processor.extract_rq_codes_from_text(instruction + " " + output)
                
                if rq_codes is not None:
                    valid_data.append(item)
                else:
                    # 如果没有找到RQ代码，检查是否包含特殊token
                    if any(token in instruction + output for token in self.rq_processor.special_tokens.values()):
                        valid_data.append(item)
            
            except Exception as e:
                print(f"Error validating item: {e}")
                continue
        
        print(f"Validated {len(valid_data)}/{len(data)} items")
        return valid_data
    
    def create_balanced_dataset(self, 
                              data: List[Dict],
                              task_type_weights: Dict[str, float] = None) -> List[Dict]:
        """
        创建平衡的数据集
        
        Args:
            data: 数据列表
            task_type_weights: 任务类型权重
            
        Returns:
            平衡后的数据列表
        """
        if task_type_weights is None:
            task_type_weights = {
                "generation": 0.4,
                "understanding": 0.3,
                "reasoning": 0.2,
                "code_understanding": 0.1
            }
        
        # 按任务类型分组
        task_groups = {}
        for item in data:
            task_type = item.get("task_type", "unknown")
            if task_type not in task_groups:
                task_groups[task_type] = []
            task_groups[task_type].append(item)
        
        # 计算目标样本数
        total_samples = len(data)
        balanced_data = []
        
        for task_type, weight in task_type_weights.items():
            if task_type in task_groups:
                target_samples = int(total_samples * weight)
                available_samples = task_groups[task_type]
                
                if len(available_samples) >= target_samples:
                    # 随机采样
                    selected = random.sample(available_samples, target_samples)
                else:
                    # 重复采样
                    selected = random.choices(available_samples, k=target_samples)
                
                balanced_data.extend(selected)
        
        # 打乱数据
        random.shuffle(balanced_data)
        
        print(f"Created balanced dataset with {len(balanced_data)} samples")
        return balanced_data


# 使用示例
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据加载器
    data_loader = RQLoraDataLoader(
        tokenizer=tokenizer,
        max_length=512,
        batch_size=2,
        num_workers=2
    )
    
    # 加载数据
    train_data = data_loader.load_data("lora_data/train.json")
    val_data = data_loader.load_data("lora_data/validation.json")
    
    # 验证数据
    train_data = data_loader.validate_rq_tokens(train_data)
    val_data = data_loader.validate_rq_tokens(val_data)
    
    # 创建平衡数据集
    train_data = data_loader.create_balanced_dataset(train_data)
    
    # 创建数据集
    train_dataset = data_loader.create_rq_dataset(train_data)
    val_dataset = data_loader.create_rq_dataset(val_data)
    
    # 创建数据加载器
    train_loader = data_loader.create_dataloader(train_dataset, shuffle=True)
    val_loader = data_loader.create_dataloader(val_dataset, shuffle=False)
    
    # 测试数据加载
    for batch in train_loader:
        print("Batch shape:")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        print(f"  labels: {batch['labels'].shape}")
        break
