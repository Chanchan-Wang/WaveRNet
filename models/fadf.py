"""
Multi-Domain Adapter Module
为每个域学习可学习的domain token，提升跨域泛化能力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDomainAdapter(nn.Module):
    """
    多域适配器
    为每个域（RECOVERY、CHASE、DRIVE、STARE）学习一个domain token
    """
    def __init__(self, embed_dim=256, num_domains=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_domains = num_domains
        
        # 为每个域创建可学习的domain token
        self.domain_tokens = nn.Parameter(torch.randn(num_domains, embed_dim))
        
        # Domain-specific adapter layers
        self.domain_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, embed_dim)
            ) for _ in range(num_domains)
        ])
        
        # 初始化
        nn.init.normal_(self.domain_tokens, std=0.02)
        
    def forward(self, x, domain_idx=None, training=True):
        """
        Args:
            x: 特征图 [B, C, H, W]
            domain_idx: 训练时的域索引，测试时为None
            training: 是否训练模式
        
        Returns:
            训练时: 单个域的输出
            测试时: 所有域的输出列表（用于选择最佳IoU）
        """
        B, C, H, W = x.shape
        
        if training:
            # 训练模式：使用指定的domain token
            assert domain_idx is not None, "domain_idx required in training mode"
            
            # 确保domain_idx是整数（如果是tensor，取第一个元素）
            # 必须在索引之前转换！
            if isinstance(domain_idx, torch.Tensor):
                if domain_idx.dim() == 0:  # 标量tensor
                    domain_idx = domain_idx.item()
                elif domain_idx.numel() == 1:  # 单元素tensor
                    domain_idx = domain_idx.item()
                else:  # 多元素tensor，取第一个
                    domain_idx = domain_idx[0].item()
            
            # 转换为int类型（防止numpy等类型）
            domain_idx = int(domain_idx)
            
            domain_token = self.domain_tokens[domain_idx].unsqueeze(0).expand(B, -1)  # [B, C]
            
            # 应用domain-specific adapter
            adapted_token = self.domain_adapters[domain_idx](domain_token)  # [B, C]
            
            # 将domain token融合到特征图
            adapted_token = adapted_token.view(B, C, 1, 1).expand(B, C, H, W)
            out = x + adapted_token
            
            return out
        else:
            # 测试模式：返回所有域的输出
            outputs = []
            for idx in range(self.num_domains):
                domain_token = self.domain_tokens[idx].unsqueeze(0).expand(B, -1)
                adapted_token = self.domain_adapters[idx](domain_token)
                adapted_token = adapted_token.view(B, C, 1, 1).expand(B, C, H, W)
                out = x + adapted_token
                outputs.append(out)
            
            return outputs  # List of [B, C, H, W]
    
    def get_domain_idx(self, image_path):
        """
        根据图像路径自动识别域
        """
        if 'RECOVERY' in image_path.upper():
            return 0
        elif 'CHASE' in image_path.upper():
            return 1
        elif 'DRIVE' in image_path.upper():
            return 2
        elif 'STARE' in image_path.upper():
            return 3
        else:
            return 0  # 默认第一个域

