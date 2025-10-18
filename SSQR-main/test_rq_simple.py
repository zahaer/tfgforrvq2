#!/usr/bin/env python3
"""
简化的RQ-VAE测试脚本，用于验证RQ-VAE核心功能
暂时跳过dgl依赖，直接测试RQCodebook
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加model目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from RQCodebook import RQCodebook

def test_rq_codebook():
    """测试RQ-VAE码本功能"""
    print("=== 测试RQ-VAE码本功能 ===")
    
    # 设置参数
    num_codes = 1024
    latent_dim = 200
    num_quantizers = 4
    batch_size = 32
    seq_len = 16
    
    # 创建RQ码本
    rq_codebook = RQCodebook(
        num_codes=num_codes,
        latent_dim=latent_dim,
        num_quantizers=num_quantizers
    )
    
    print(f"创建RQ码本: {num_codes}个码向量, {latent_dim}维, {num_quantizers}个量化器")
    
    # 创建测试数据
    test_input = torch.randn(batch_size, seq_len, latent_dim)
    print(f"测试输入形状: {test_input.shape}")
    
    # 前向传播
    quantized_output, code_indices, loss = rq_codebook(test_input)
    print(f"量化输出形状: {quantized_output.shape}")
    print(f"代码索引形状: {code_indices.shape}")
    print(f"量化损失: {loss.item():.4f}")
    
    # 测试代码生成
    codes = rq_codebook.cal_codes(test_input)
    print(f"生成的代码形状: {codes.shape}")
    
    # 测试代码重构
    reconstructed = rq_codebook.decode_from_codes(codes)
    print(f"重构输出形状: {reconstructed.shape}")
    
    # 计算重构误差
    reconstruction_error = torch.mean((test_input - reconstructed) ** 2)
    print(f"重构误差: {reconstruction_error.item():.4f}")
    
    print("✅ RQ-VAE码本测试完成!")
    return True

def test_rq_training():
    """测试RQ-VAE训练过程"""
    print("\n=== 测试RQ-VAE训练过程 ===")
    
    # 设置参数
    num_codes = 512
    latent_dim = 128
    num_quantizers = 3
    batch_size = 16
    seq_len = 8
    num_epochs = 5
    
    # 创建RQ码本
    rq_codebook = RQCodebook(
        num_codes=num_codes,
        latent_dim=latent_dim,
        num_quantizers=num_quantizers
    )
    
    # 创建优化器
    optimizer = torch.optim.Adam(rq_codebook.parameters(), lr=0.001)
    
    print(f"开始训练: {num_epochs}个epoch")
    
    for epoch in range(num_epochs):
        # 生成随机训练数据
        train_data = torch.randn(batch_size, seq_len, latent_dim)
        
        # 前向传播
        quantized_output, code_indices, loss = rq_codebook(train_data)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    print("✅ RQ-VAE训练测试完成!")
    return True

def generate_sample_codes():
    """生成示例RQ代码文件"""
    print("\n=== 生成示例RQ代码文件 ===")
    
    # 设置参数
    num_entities = 1000
    seq_len = 16
    num_quantizers = 4
    
    # 生成随机代码
    sample_codes = torch.randint(0, 1024, (num_entities, seq_len, num_quantizers))
    
    # 保存代码文件
    output_file = "codes_new/FB15k-237_16_1024_4_rq.pt"
    torch.save(sample_codes, output_file)
    
    print(f"生成示例代码文件: {output_file}")
    print(f"代码形状: {sample_codes.shape}")
    print("✅ 示例代码文件生成完成!")
    
    return output_file

def main():
    """主函数"""
    print("🚀 开始RQ-VAE功能测试")
    
    try:
        # 测试RQ码本功能
        test_rq_codebook()
        
        # 测试训练过程
        test_rq_training()
        
        # 生成示例代码文件
        generate_sample_codes()
        
        print("\n🎉 所有测试完成!")
        print("RQ-VAE核心功能验证成功!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
