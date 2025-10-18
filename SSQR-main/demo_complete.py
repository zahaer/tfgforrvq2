#!/usr/bin/env python3
"""
完整的RQ-VAE项目演示脚本
展示从RQ-VAE训练到LoRA微调的完整流程
"""

import os
import sys
import torch
import numpy as np
import json
from pathlib import Path

# 添加model目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

def demo_rq_vae_training():
    """演示RQ-VAE训练过程"""
    print("🎯 步骤1: RQ-VAE训练演示")
    print("=" * 50)
    
    from RQCodebook import RQCodebook
    
    # 设置参数
    num_codes = 1024
    latent_dim = 200
    num_quantizers = 4
    batch_size = 64
    seq_len = 16
    num_epochs = 10
    
    print(f"训练参数:")
    print(f"  - 码本大小: {num_codes}")
    print(f"  - 潜在维度: {latent_dim}")
    print(f"  - 量化器数量: {num_quantizers}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 序列长度: {seq_len}")
    print(f"  - 训练轮数: {num_epochs}")
    
    # 创建RQ码本
    rq_codebook = RQCodebook(
        num_codes=num_codes,
        latent_dim=latent_dim,
        num_quantizers=num_quantizers
    )
    
    # 创建优化器
    optimizer = torch.optim.Adam(rq_codebook.parameters(), lr=0.001)
    
    print(f"\n开始训练...")
    
    # 训练循环
    for epoch in range(num_epochs):
        # 生成随机训练数据（模拟知识图谱实体嵌入）
        train_data = torch.randn(batch_size, seq_len, latent_dim)
        
        # 前向传播
        quantized_output, code_indices, loss = rq_codebook(train_data)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1:2d}/{num_epochs}: Loss = {loss.item():.4f}")
    
    print(f"✅ RQ-VAE训练完成!")
    
    # 生成实体代码
    print(f"\n生成实体代码...")
    num_entities = 1000
    entity_embeddings = torch.randn(num_entities, seq_len, latent_dim)
    
    with torch.no_grad():
        entity_codes = rq_codebook.cal_codes(entity_embeddings)
    
    # 保存代码文件
    codes_dir = Path("codes_new")
    codes_dir.mkdir(exist_ok=True)
    
    codes_file = codes_dir / "FB15k-237_16_1024_4_rq.pt"
    torch.save(entity_codes, codes_file)
    
    print(f"✅ 实体代码已保存到: {codes_file}")
    print(f"   代码形状: {entity_codes.shape}")
    
    return codes_file, rq_codebook

def demo_data_generation(codes_file):
    """演示数据生成过程"""
    print("\n🎯 步骤2: LoRA训练数据生成演示")
    print("=" * 50)
    
    # 加载RQ代码
    entity_codes = torch.load(codes_file)
    num_entities, seq_len, num_quantizers = entity_codes.shape
    
    print(f"加载RQ代码:")
    print(f"  - 实体数量: {num_entities}")
    print(f"  - 序列长度: {seq_len}")
    print(f"  - 量化器数量: {num_quantizers}")
    
    # 模拟实体名称
    entity_names = [f"Entity_{i:04d}" for i in range(num_entities)]
    
    # 生成训练任务
    tasks = []
    
    # 实体表示学习任务
    for i in range(min(100, num_entities)):  # 生成100个任务
        entity_name = entity_names[i]
        entity_code = entity_codes[i]
        
        # 生成代码字符串
        code_parts = []
        for pos in range(seq_len):
            level_codes = [f"Q{j}_{entity_code[pos, j].item()}" for j in range(num_quantizers)]
            code_parts.append(":".join(level_codes))
        code_str = "|".join(code_parts)
        
        # 创建任务
        task = {
            "instruction": f"Explain the quantized representation of entity '{entity_name}':\n<ENT>{entity_name}</ENT> <CODE>{code_str}</CODE>",
            "input": "",
            "output": f"The quantized representation of '{entity_name}' consists of {seq_len} sequence positions, each with {num_quantizers} quantizer levels. This representation captures the entity's semantic properties in a compressed format.",
            "task_type": "understanding",
            "entity_id": i
        }
        tasks.append(task)
    
    # 知识推理任务
    for i in range(min(50, num_entities // 2)):
        head_entity = entity_names[i * 2]
        tail_entity = entity_names[i * 2 + 1]
        relation = f"relation_{i % 10}"
        
        head_code = entity_codes[i * 2]
        tail_code = entity_codes[i * 2 + 1]
        
        # 生成代码字符串
        def code_to_str(code):
            code_parts = []
            for pos in range(seq_len):
                level_codes = [f"Q{j}_{code[pos, j].item()}" for j in range(num_quantizers)]
                code_parts.append(":".join(level_codes))
            return "|".join(code_parts)
        
        head_code_str = code_to_str(head_code)
        tail_code_str = code_to_str(tail_code)
        
        task = {
            "instruction": f"Validate the knowledge triplet:\n<ENT>{head_entity}</ENT> <CODE>{head_code_str}</CODE>\nRelation: {relation}\n<ENT>{tail_entity}</ENT> <CODE>{tail_code_str}</CODE>\nIs this triplet valid?",
            "input": "",
            "output": f"Yes, this knowledge triplet is valid. The relationship between '{head_entity}' and '{tail_entity}' through '{relation}' is supported by their quantized representations.",
            "task_type": "validation",
            "triplet": (i * 2, i % 10, i * 2 + 1)
        }
        tasks.append(task)
    
    print(f"生成了 {len(tasks)} 个训练任务")
    print(f"  - 实体表示任务: 100")
    print(f"  - 知识推理任务: 50")
    
    # 保存训练数据
    lora_data_dir = Path("lora_training/lora_data")
    lora_data_dir.mkdir(exist_ok=True)
    
    # 分割训练和验证数据
    train_tasks = tasks[:120]
    val_tasks = tasks[120:]
    
    train_file = lora_data_dir / "train.json"
    val_file = lora_data_dir / "validation.json"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_tasks, f, ensure_ascii=False, indent=2)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_tasks, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 训练数据已保存:")
    print(f"  - 训练集: {train_file} ({len(train_tasks)} 个任务)")
    print(f"  - 验证集: {val_file} ({len(val_tasks)} 个任务)")
    
    return train_file, val_file

def demo_lora_training(train_file, val_file):
    """演示LoRA训练过程（模拟）"""
    print("\n🎯 步骤3: LoRA微调演示")
    print("=" * 50)
    
    print("LoRA训练配置:")
    print("  - 基础模型: LLaMA2-7B")
    print("  - LoRA秩: 16")
    print("  - LoRA Alpha: 32")
    print("  - 学习率: 2e-4")
    print("  - 批次大小: 4")
    print("  - 训练轮数: 3")
    
    # 模拟训练过程
    print(f"\n模拟训练过程...")
    
    # 加载训练数据
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(val_file, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    print(f"  - 训练样本: {len(train_data)}")
    print(f"  - 验证样本: {len(val_data)}")
    
    # 模拟训练指标
    print(f"\n模拟训练指标:")
    for epoch in range(3):
        train_loss = 2.5 - epoch * 0.3 + np.random.normal(0, 0.1)
        val_loss = 2.8 - epoch * 0.2 + np.random.normal(0, 0.1)
        print(f"  Epoch {epoch+1}: Train Loss = {train_loss:.3f}, Val Loss = {val_loss:.3f}")
    
    print(f"✅ LoRA训练完成!")
    
    # 保存模型
    model_dir = Path("lora_training/lora_outputs")
    model_dir.mkdir(exist_ok=True)
    
    adapter_dir = model_dir / "lora_adapters"
    adapter_dir.mkdir(exist_ok=True)
    
    # 创建模拟的模型配置文件
    config = {
        "base_model_name": "meta-llama/Llama-2-7b-hf",
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.1
        },
        "training_args": {
            "learning_rate": 2e-4,
            "batch_size": 4,
            "num_epochs": 3
        }
    }
    
    config_file = adapter_dir / "adapter_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"✅ LoRA适配器已保存到: {adapter_dir}")
    
    return adapter_dir

def demo_inference(adapter_dir, codes_file):
    """演示推理过程"""
    print("\n🎯 步骤4: 模型推理演示")
    print("=" * 50)
    
    # 加载RQ代码
    entity_codes = torch.load(codes_file)
    
    print("推理测试:")
    print("  - 加载LoRA适配器")
    print("  - 加载RQ代码")
    print("  - 执行推理")
    
    # 模拟推理过程
    test_entities = ["Entity_0001", "Entity_0002", "Entity_0003"]
    
    for i, entity_name in enumerate(test_entities):
        entity_code = entity_codes[i]
        
        # 生成代码字符串
        code_parts = []
        for pos in range(entity_code.shape[0]):
            level_codes = [f"Q{j}_{entity_code[pos, j].item()}" for j in range(entity_code.shape[1])]
            code_parts.append(":".join(level_codes))
        code_str = "|".join(code_parts)
        
        # 模拟模型输出
        prompt = f"Explain the quantized representation of entity '{entity_name}':\n<ENT>{entity_name}</ENT> <CODE>{code_str}</CODE>"
        
        # 模拟生成回答
        response = f"The quantized representation of '{entity_name}' consists of {entity_code.shape[0]} sequence positions, each with {entity_code.shape[1]} quantizer levels. This representation captures the entity's semantic properties in a compressed format, enabling efficient knowledge graph reasoning."
        
        print(f"\n测试 {i+1}: {entity_name}")
        print(f"  输入: {prompt[:100]}...")
        print(f"  输出: {response[:100]}...")
    
    print(f"\n✅ 推理测试完成!")
    
    # 保存推理结果
    results = {
        "test_entities": test_entities,
        "inference_results": [
            {
                "entity": entity,
                "prompt": f"Explain the quantized representation of entity '{entity}'",
                "response": f"The quantized representation of '{entity}' consists of multiple sequence positions with quantizer levels."
            }
            for entity in test_entities
        ]
    }
    
    results_file = Path("lora_training/inference_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 推理结果已保存到: {results_file}")

def main():
    """主函数"""
    print("🚀 RQ-VAE项目完整演示")
    print("=" * 60)
    print("本演示将展示从RQ-VAE训练到LoRA微调的完整流程")
    print("=" * 60)
    
    try:
        # 步骤1: RQ-VAE训练
        codes_file, rq_codebook = demo_rq_vae_training()
        
        # 步骤2: 数据生成
        train_file, val_file = demo_data_generation(codes_file)
        
        # 步骤3: LoRA训练
        adapter_dir = demo_lora_training(train_file, val_file)
        
        # 步骤4: 推理测试
        demo_inference(adapter_dir, codes_file)
        
        print("\n" + "=" * 60)
        print("🎉 完整演示成功完成!")
        print("=" * 60)
        print("生成的文件:")
        print(f"  - RQ代码: {codes_file}")
        print(f"  - 训练数据: {train_file}")
        print(f"  - 验证数据: {val_file}")
        print(f"  - LoRA适配器: {adapter_dir}")
        print(f"  - 推理结果: lora_training/inference_results.json")
        print("\n项目已成功配置并运行!")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
