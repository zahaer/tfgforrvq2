# RQ-VAE替换VQ-VAE使用指南

本文档说明如何将项目中的VQ-VAE替换为RQ-VAE（Residual Quantized VAE）。

## 主要变化

### 1. 新增文件

- `model/RQCodebook.py`: RQ-VAE码本实现
- `model/RQGCN.py`: 使用RQ-VAE的图卷积网络模型
- `run_rq.py`: RQ-VAE训练脚本
- `Class/gen_fb_data_rq.py`: RQ-VAE数据生成脚本

### 2. 核心差异

#### VQ-VAE vs RQ-VAE

| 特性 | VQ-VAE | RQ-VAE |
|------|--------|--------|
| 码本数量 | 1个 | 多个（可配置） |
| 量化方式 | 单次量化 | 多级残差量化 |
| 代码维度 | [B, L] | [B, L, num_quantizers] |
| 表示精度 | 固定 | 可调节（通过量化器数量） |

#### RQ-VAE优势

1. **更好的表示能力**: 多级残差量化可以捕获更细粒度的信息
2. **可调节精度**: 通过调整量化器数量平衡精度和效率
3. **渐进式量化**: 每级量化器专注于前一级的残差

## 使用方法

### 1. 训练RQ-VAE模型

```bash
python run_rq.py \
    --dataset data/FB15k-237 \
    --seq_len 16 \
    --num_code 1024 \
    --num_quantizers 4 \
    --gcn_layers 2 \
    --tf_layers 1 \
    --epoch 800 \
    --lr 0.0005 \
    --batch 1024
```

### 2. 生成LLM训练数据

```bash
python Class/gen_fb_data_rq.py
```

### 3. 参数说明

#### 新增参数

- `--num_quantizers`: RQ-VAE量化器数量（默认4）
- 其他参数与原始VQ-VAE相同

#### 推荐配置

- **小数据集**: `num_quantizers=2-3`
- **中等数据集**: `num_quantizers=4-6`  
- **大数据集**: `num_quantizers=6-8`

## 代码结构

### RQCodebook类

```python
class RQCodebook(nn.Module):
    def __init__(self, num_codes, latent_dim, num_quantizers=4, beta=0.25):
        # 为每个量化器创建独立码本
        self.codebooks = nn.ModuleList([...])
    
    def forward(self, z):
        # 多级残差量化
        # 返回: z_q, all_indices, total_loss
    
    def cal_codes(self, z):
        # 生成所有量化器的代码索引
        # 返回: [B, L, num_quantizers]
```

### RQGCN类

```python
class RQGCN(nn.Module):
    def __init__(self, ..., num_quantizers=4):
        # 使用RQCodebook替代Codebook
        self.rq_codebook = RQCodebook(...)
    
    def cal_allent_codes(self, g):
        # 生成形状为[M, L, num_quantizers]的代码
```

## 输出格式

### 代码文件

- **VQ-VAE**: `{dataset}_{seq_len}_{num_code}.pt` (形状: [M, L])
- **RQ-VAE**: `{dataset}_{seq_len}_{num_code}_{num_quantizers}_rq.pt` (形状: [M, L, num_quantizers])

### 嵌入文件

- **VQ-VAE**: `{dataset}_{seq_len}_{num_code}_emd.pt` (形状: [M, L, D])
- **RQ-VAE**: `{dataset}_{seq_len}_{num_code}_{num_quantizers}_rq_emd.pt` (形状: [M, L, D])

## 性能对比

### 内存使用

- **VQ-VAE**: O(num_codes × latent_dim)
- **RQ-VAE**: O(num_quantizers × num_codes × latent_dim)

### 计算复杂度

- **VQ-VAE**: O(B × L × num_codes)
- **RQ-VAE**: O(B × L × num_quantizers × num_codes)

## 注意事项

1. **内存需求**: RQ-VAE需要更多内存，建议适当调整batch_size
2. **训练时间**: 多级量化会增加训练时间
3. **超参数调优**: 需要重新调优学习率和权重衰减
4. **代码兼容性**: 生成的代码格式不同，需要相应修改下游任务

## 迁移步骤

1. **备份原始代码**: 保存VQ-VAE相关文件
2. **替换模型**: 使用RQGCN替代VQGCN
3. **更新训练脚本**: 使用run_rq.py
4. **修改数据生成**: 使用gen_fb_data_rq.py
5. **调整超参数**: 根据数据集大小调整num_quantizers
6. **验证结果**: 对比VQ-VAE和RQ-VAE的性能

## 故障排除

### 常见问题

1. **内存不足**: 减少batch_size或num_quantizers
2. **训练不稳定**: 降低学习率或增加梯度裁剪
3. **性能下降**: 检查num_quantizers设置，可能需要调整

### 调试建议

1. 使用较小的num_quantizers进行初步测试
2. 监控每个量化器的损失变化
3. 可视化量化后的表示质量
