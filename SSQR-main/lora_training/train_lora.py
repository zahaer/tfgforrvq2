import os
import torch
import json
import argparse
from typing import Dict, List, Optional
from pathlib import Path
import logging
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import wandb
from data_loader import RQLoraDataLoader
from rq_token_processor import RQTokenProcessor


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RQLoraTrainer:
    """
    RQ-VAE LoRA训练器
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-hf",
                 output_dir: str = "lora_outputs",
                 lora_config: Optional[Dict] = None,
                 training_args: Optional[Dict] = None):
        """
        初始化训练器
        
        Args:
            model_name: 基础模型名称
            output_dir: 输出目录
            lora_config: LoRA配置
            training_args: 训练参数
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 默认LoRA配置
        self.lora_config = lora_config or {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM
        }
        
        # 默认训练参数
        self.training_args = training_args or {
            "output_dir": str(self.output_dir),
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 100,
            "learning_rate": 2e-4,
            "fp16": True,
            "logging_steps": 10,
            "evaluation_strategy": "steps",
            "eval_steps": 500,
            "save_steps": 500,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "report_to": "wandb",
            "run_name": f"rq_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.trainer = None
        
        # 初始化RQ token处理器
        self.rq_processor = RQTokenProcessor(model_name)
    
    def setup_model_and_tokenizer(self):
        """设置模型和分词器"""
        logger.info(f"Loading model and tokenizer from {self.model_name}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 添加特殊token
        special_tokens = list(self.rq_processor.special_tokens.values())
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens
        })
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 调整模型嵌入层大小
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"Model loaded with {len(self.tokenizer)} tokens")
    
    def setup_lora(self):
        """设置LoRA"""
        logger.info("Setting up LoRA configuration")
        
        # 创建LoRA配置
        lora_config = LoraConfig(**self.lora_config)
        
        # 应用LoRA到模型
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # 打印可训练参数
        self.peft_model.print_trainable_parameters()
        
        logger.info("LoRA setup completed")
    
    def setup_data(self, 
                   train_data_path: str,
                   val_data_path: str,
                   batch_size: int = 4,
                   max_length: int = 512) -> tuple:
        """
        设置数据加载器
        
        Args:
            train_data_path: 训练数据路径
            val_data_path: 验证数据路径
            batch_size: 批次大小
            max_length: 最大长度
            
        Returns:
            训练和验证数据加载器
        """
        logger.info("Setting up data loaders")
        
        # 创建数据加载器
        data_loader = RQLoraDataLoader(
            tokenizer=self.tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            rq_processor=self.rq_processor
        )
        
        # 加载数据
        train_data = data_loader.load_data(train_data_path)
        val_data = data_loader.load_data(val_data_path)
        
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
        
        logger.info(f"Data setup completed: {len(train_data)} train, {len(val_data)} val samples")
        
        return train_loader, val_loader
    
    def setup_trainer(self, train_loader, val_loader):
        """设置训练器"""
        logger.info("Setting up trainer")
        
        # 更新训练参数
        training_args = TrainingArguments(**self.training_args)
        
        # 创建训练器
        self.trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        logger.info("Trainer setup completed")
    
    def train(self):
        """开始训练"""
        logger.info("Starting training")
        
        # 开始训练
        self.trainer.train()
        
        # 保存最终模型
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Training completed. Model saved to {self.output_dir}")
    
    def evaluate(self):
        """评估模型"""
        logger.info("Evaluating model")
        
        eval_results = self.trainer.evaluate()
        
        logger.info("Evaluation results:")
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value}")
        
        return eval_results
    
    def save_lora_adapters(self, save_path: str):
        """保存LoRA适配器"""
        logger.info(f"Saving LoRA adapters to {save_path}")
        
        self.peft_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info("LoRA adapters saved")
    
    def load_lora_adapters(self, adapter_path: str):
        """加载LoRA适配器"""
        logger.info(f"Loading LoRA adapters from {adapter_path}")
        
        self.peft_model = PeftModel.from_pretrained(self.model, adapter_path)
        
        logger.info("LoRA adapters loaded")
    
    def generate_response(self, 
                         prompt: str, 
                         max_length: int = 256,
                         temperature: float = 0.7,
                         top_p: float = 0.9) -> str:
        """
        生成回答
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top-p参数
            
        Returns:
            生成的回答
        """
        # Tokenize输入
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 生成回答
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除输入部分
        response = response[len(prompt):].strip()
        
        return response


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train LLaMA2 with RQ-VAE tokens using LoRA")
    
    # 模型参数
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf", help="Base model name")
    parser.add_argument("--output_dir", default="lora_outputs", help="Output directory")
    
    # 数据参数
    parser.add_argument("--train_data", default="lora_data/train.json", help="Training data path")
    parser.add_argument("--val_data", default="lora_data/validation.json", help="Validation data path")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    
    # LoRA参数
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    
    # 其他参数
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--wandb_project", default="rq-lora", help="Wandb project name")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # 初始化wandb
    wandb.init(project=args.wandb_project, config=vars(args))
    
    # 创建训练器
    trainer = RQLoraTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        lora_config={
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM
        },
        training_args={
            "output_dir": args.output_dir,
            "num_train_epochs": args.epochs,
            "per_device_train_batch_size": args.batch_size,
            "per_device_eval_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "warmup_steps": args.warmup_steps,
            "learning_rate": args.learning_rate,
            "fp16": args.fp16,
            "logging_steps": 10,
            "evaluation_strategy": "steps",
            "eval_steps": 500,
            "save_steps": 500,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "report_to": "wandb",
            "run_name": f"rq_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
    )
    
    # 设置模型和分词器
    trainer.setup_model_and_tokenizer()
    
    # 设置LoRA
    trainer.setup_lora()
    
    # 设置数据
    train_loader, val_loader = trainer.setup_data(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # 设置训练器
    trainer.setup_trainer(train_loader, val_loader)
    
    # 开始训练
    trainer.train()
    
    # 评估模型
    eval_results = trainer.evaluate()
    
    # 保存LoRA适配器
    adapter_path = os.path.join(args.output_dir, "lora_adapters")
    trainer.save_lora_adapters(adapter_path)
    
    # 测试生成
    test_prompt = "### RQ-VAE Instruction:\nExplain the quantized representation of entity 'Barack Obama':\n<ENT>Barack Obama</ENT> <CODE>Q0_123:Q1_456:Q2_789|Q0_234:Q1_567:Q2_890</CODE>\n\n### Response:\n"
    response = trainer.generate_response(test_prompt)
    logger.info(f"Test generation: {response}")
    
    wandb.finish()


if __name__ == "__main__":
    main()
