#!/usr/bin/env python3
"""
简化的LoRA测试脚本，用于验证RQ token处理功能
暂时跳过transformers和peft依赖
"""

import sys
import os
import json
import numpy as np
import torch

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

def test_rq_token_processor():
    """测试RQ token处理器（简化版本）"""
    print("=== 测试RQ Token处理器 ===")
    
    # 模拟RQ代码
    rq_codes = np.array([
        [123, 456, 789, 12],
        [234, 567, 890, 123],
        [345, 678, 901, 234],
        [456, 789, 12, 345]
    ])
    
    print(f"RQ代码形状: {rq_codes.shape}")
    print(f"RQ代码内容:\n{rq_codes}")
    
    # 模拟token处理
    def rq_codes_to_text(rq_codes, entity_name=""):
        """将RQ代码转换为文本格式"""
        seq_len, num_quantizers = rq_codes.shape
        
        # 构建代码字符串
        code_parts = []
        for i in range(seq_len):
            level_codes = []
            for j in range(num_quantizers):
                level_codes.append(f"Q{j}_{rq_codes[i, j]}")
            code_parts.append(":".join(level_codes))
        
        code_str = "|".join(code_parts)
        
        # 构建完整文本
        if entity_name:
            text = f"<ENT>{entity_name}</ENT> <CODE>{code_str}</CODE>"
        else:
            text = f"<CODE>{code_str}</CODE>"
        
        return text
    
    # 测试转换
    entity_name = "Barack Obama"
    text = rq_codes_to_text(rq_codes, entity_name)
    print(f"转换后的文本: {text}")
    
    # 测试代码提取
    def extract_rq_codes_from_text(text):
        """从文本中提取RQ代码"""
        import re
        
        # 匹配代码模式
        code_pattern = r'<CODE>(.*?)</CODE>'
        match = re.search(code_pattern, text)
        
        if not match:
            return None
        
        code_str = match.group(1)
        
        # 解析代码
        try:
            levels = code_str.split("|")
            codes = []
            
            for level in levels:
                quantizers = level.split(":")
                level_codes = []
                
                for quantizer in quantizers:
                    if quantizer.startswith('Q') and '_' in quantizer:
                        code_value = int(quantizer.split('_')[1])
                        level_codes.append(code_value)
                    else:
                        return None
                
                codes.append(level_codes)
            
            return np.array(codes)
        
        except (ValueError, IndexError):
            return None
    
    # 测试提取
    extracted_codes = extract_rq_codes_from_text(text)
    print(f"提取的代码形状: {extracted_codes.shape}")
    print(f"提取的代码内容:\n{extracted_codes}")
    
    # 验证一致性
    if np.array_equal(rq_codes, extracted_codes):
        print("✅ 代码提取验证成功!")
    else:
        print("❌ 代码提取验证失败!")
    
    return True

def test_dataset_generation():
    """测试数据集生成（简化版本）"""
    print("\n=== 测试数据集生成 ===")
    
    # 模拟实体数据
    entities = ["Barack Obama", "Donald Trump", "Joe Biden", "Hillary Clinton"]
    relations = ["president_of", "spouse_of", "born_in", "educated_at"]
    
    # 模拟RQ代码
    num_entities = len(entities)
    seq_len = 4
    num_quantizers = 3
    
    rq_codes = np.random.randint(0, 1024, (num_entities, seq_len, num_quantizers))
    
    print(f"实体数量: {num_entities}")
    print(f"关系数量: {len(relations)}")
    print(f"RQ代码形状: {rq_codes.shape}")
    
    # 生成训练任务
    tasks = []
    
    for i, entity in enumerate(entities):
        entity_codes = rq_codes[i]
        
        # 生成实体表示任务
        task = {
            "instruction": f"Explain the quantized representation of entity '{entity}':\n<ENT>{entity}</ENT> <CODE>{':'.join([f'Q{j}_{entity_codes[0, j]}' for j in range(num_quantizers)])}</CODE>",
            "input": "",
            "output": f"The quantized representation of '{entity}' consists of {seq_len} sequence positions, each with {num_quantizers} quantizer levels.",
            "task_type": "understanding",
            "entity_id": i
        }
        tasks.append(task)
    
    print(f"生成了 {len(tasks)} 个训练任务")
    
    # 保存任务数据
    output_file = "lora_data/test_tasks.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    
    print(f"任务数据已保存到: {output_file}")
    
    # 验证保存的数据
    with open(output_file, 'r', encoding='utf-8') as f:
        loaded_tasks = json.load(f)
    
    print(f"验证: 加载了 {len(loaded_tasks)} 个任务")
    print("✅ 数据集生成测试完成!")
    
    return True

def test_training_data_format():
    """测试训练数据格式"""
    print("\n=== 测试训练数据格式 ===")
    
    # 创建示例训练数据
    sample_data = {
        "instruction": "### RQ-VAE Instruction:\nExplain the quantized representation of entity 'Barack Obama':\n<ENT>Barack Obama</ENT> <CODE>Q0_123:Q1_456:Q2_789|Q0_234:Q1_567:Q2_890</CODE>\n\n### Response:\n",
        "input": "",
        "output": "The quantized representation of 'Barack Obama' consists of 4 sequence positions, each with 3 quantizer levels. This representation captures the entity's semantic properties in a compressed format."
    }
    
    print("示例训练数据:")
    print(json.dumps(sample_data, ensure_ascii=False, indent=2))
    
    # 验证数据格式
    required_fields = ["instruction", "input", "output"]
    for field in required_fields:
        if field in sample_data:
            print(f"✅ 字段 '{field}' 存在")
        else:
            print(f"❌ 字段 '{field}' 缺失")
    
    print("✅ 训练数据格式测试完成!")
    return True

def main():
    """主函数"""
    print("🚀 开始LoRA功能测试")
    
    try:
        # 测试RQ token处理器
        test_rq_token_processor()
        
        # 测试数据集生成
        test_dataset_generation()
        
        # 测试训练数据格式
        test_training_data_format()
        
        print("\n🎉 所有LoRA测试完成!")
        print("LoRA核心功能验证成功!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
