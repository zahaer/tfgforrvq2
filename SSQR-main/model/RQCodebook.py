import torch
import torch.nn as nn
import torch.nn.functional as F


class RQCodebook(nn.Module):
    """
    Residual Quantized Codebook for RQ-VAE
    支持多级残差量化，每级使用独立的码本
    """
    def __init__(self, num_codes, latent_dim, num_quantizers=4, beta=0.25):
        super(RQCodebook, self).__init__()
        self.num_codebook_vectors = num_codes
        self.latent_dim = latent_dim
        self.num_quantizers = num_quantizers
        self.beta = beta
        
        # 为每个量化器创建独立的码本
        self.codebooks = nn.ModuleList([
            nn.Embedding(self.num_codebook_vectors, self.latent_dim)
            for _ in range(self.num_quantizers)
        ])
        
        # 初始化码本权重
        for codebook in self.codebooks:
            codebook.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 
                                        1.0 / self.num_codebook_vectors)
    
    def forward(self, z):
        """
        前向传播：执行多级残差量化
        Args:
            z: 输入特征 [B, L, D]
        Returns:
            z_q: 量化后的特征 [B, L, D]
            all_indices: 所有量化器的索引 [B, L, num_quantizers]
            total_loss: 总损失
        """
        batch_size, seq_len, dim = z.shape
        device = z.device
        
        # 存储所有量化器的索引
        all_indices = torch.zeros(batch_size, seq_len, self.num_quantizers, 
                                dtype=torch.long, device=device)
        
        # 存储量化后的特征
        z_q = torch.zeros_like(z)
        
        # 残差
        residual = z.clone()
        
        total_loss = 0.0
        
        for i, codebook in enumerate(self.codebooks):
            # 将残差展平进行量化
            residual_flat = residual.reshape(-1, self.latent_dim)  # [B*L, D]
            
            # 计算到码本的距离
            distances = torch.sum(residual_flat**2, dim=1, keepdim=True) + \
                       torch.sum(codebook.weight**2, dim=1) - \
                       2 * torch.matmul(residual_flat, codebook.weight.t())
            
            # 找到最近的码向量
            min_encoding_indices = torch.argmin(distances, dim=1)
            
            # 获取量化后的特征
            z_q_flat = codebook(min_encoding_indices)  # [B*L, D]
            z_q_flat = z_q_flat.view(batch_size, seq_len, dim)
            
            # 更新残差
            residual = residual - z_q_flat.detach()
            
            # 累积量化特征
            z_q = z_q + z_q_flat
            
            # 计算损失
            commitment_loss = torch.mean((z_q_flat.detach() - residual)**2)
            codebook_loss = torch.mean((z_q_flat - residual.detach())**2)
            quantizer_loss = commitment_loss + self.beta * codebook_loss
            total_loss = total_loss + quantizer_loss
            
            # 存储索引
            all_indices[:, :, i] = min_encoding_indices.view(batch_size, seq_len)
        
        # 使用straight-through estimator
        z_q = z + (z_q - z).detach()
        
        return z_q, all_indices, total_loss
    
    def cal_codes(self, z):
        """
        生成所有量化器的代码索引
        Args:
            z: 输入特征 [B, L, D]
        Returns:
            all_indices: 所有量化器的索引 [B, L, num_quantizers]
        """
        batch_size, seq_len, dim = z.shape
        device = z.device
        
        all_indices = torch.zeros(batch_size, seq_len, self.num_quantizers, 
                                dtype=torch.long, device=device)
        
        residual = z.clone()
        
        for i, codebook in enumerate(self.codebooks):
            residual_flat = residual.reshape(-1, self.latent_dim)
            
            distances = torch.sum(residual_flat**2, dim=1, keepdim=True) + \
                       torch.sum(codebook.weight**2, dim=1) - \
                       2 * torch.matmul(residual_flat, codebook.weight.t())
            
            min_encoding_indices = torch.argmin(distances, dim=1)
            
            # 获取量化后的特征用于更新残差
            z_q_flat = codebook(min_encoding_indices)
            z_q_flat = z_q_flat.view(batch_size, seq_len, dim)
            
            residual = residual - z_q_flat
            
            all_indices[:, :, i] = min_encoding_indices.view(batch_size, seq_len)
        
        return all_indices
    
    def cal_code_emds(self, z):
        """
        生成量化后的嵌入表示
        Args:
            z: 输入特征 [B, L, D]
        Returns:
            z_q: 量化后的特征 [B, L, D]
        """
        batch_size, seq_len, dim = z.shape
        
        z_q = torch.zeros_like(z)
        residual = z.clone()
        
        for codebook in self.codebooks:
            residual_flat = residual.reshape(-1, self.latent_dim)
            
            distances = torch.sum(residual_flat**2, dim=1, keepdim=True) + \
                       torch.sum(codebook.weight**2, dim=1) - \
                       2 * torch.matmul(residual_flat, codebook.weight.t())
            
            min_encoding_indices = torch.argmin(distances, dim=1)
            z_q_flat = codebook(min_encoding_indices)
            z_q_flat = z_q_flat.view(batch_size, seq_len, dim)
            
            residual = residual - z_q_flat
            z_q = z_q + z_q_flat
        
        return z_q
    
    def decode_from_codes(self, codes):
        """
        从代码索引重构特征
        Args:
            codes: 代码索引 [B, L, num_quantizers]
        Returns:
            z_q: 重构的特征 [B, L, D]
        """
        batch_size, seq_len, num_quantizers = codes.shape
        device = codes.device
        
        z_q = torch.zeros(batch_size, seq_len, self.latent_dim, device=device)
        
        for i, codebook in enumerate(self.codebooks):
            codes_flat = codes[:, :, i].reshape(-1)  # [B*L]
            z_q_flat = codebook(codes_flat)  # [B*L, D]
            z_q_flat = z_q_flat.view(batch_size, seq_len, self.latent_dim)
            z_q = z_q + z_q_flat
        
        return z_q
