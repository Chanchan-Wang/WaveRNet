"""
Simple RWKV Layer for SAM
基于RWKV-4架构的简化实现
用于替换Transformer中的自注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RWKVChannelMix(nn.Module):
    """RWKV的Channel Mix (类似FFN)"""
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.key = nn.Linear(dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] - batch, sequence length, channels
        Returns:
            out: [B, N, C]
        """
        k = self.key(x)
        k = torch.square(torch.relu(k))  # RWKV特有的激活
        v = self.value(k)
        r = torch.sigmoid(self.receptance(x))
        
        return r * v


class RWKVTimeMix(nn.Module):
    """RWKV的Time Mix (替代自注意力) - 简化版线性注意力"""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        简化的线性注意力：用全局池化替代标准注意力
        
        Args:
            x: [B, N, C]
        Returns:
            out: [B, N, C]
        """
        B, N, C = x.shape
        
        # 生成k, v, r
        k = self.key(x)      # [B, N, C]
        v = self.value(x)    # [B, N, C]
        r = self.receptance(x)  # [B, N, C]
        
        # 简化的线性注意力：全局加权平均
        # 1. k通过softmax生成权重
        k_weights = F.softmax(k, dim=1)  # [B, N, C] - 对序列维度softmax
        
        # 2. 全局上下文：加权求和
        context = torch.sum(k_weights * v, dim=1, keepdim=True)  # [B, 1, C]
        
        # 3. 广播到所有位置
        context = context.expand(-1, N, -1)  # [B, N, C]
        
        # 4. 使用receptance gate控制信息流
        r_gate = torch.sigmoid(r)  # [B, N, C]
        out = r_gate * context  # [B, N, C]
        
        # 5. 输出投影
        out = self.output(out)
        
        return out


class RWKVLayer(nn.Module):
    """完整的RWKV层 = TimeMix + ChannelMix"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(dim)
        self.time_mix = RWKVTimeMix(dim, num_heads)
        
        self.ln2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.channel_mix = RWKVChannelMix(dim, mlp_hidden_dim)
        
        # DropPath（可选）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C]
        Returns:
            out: [B, N, C]
        """
        # TimeMix with residual
        x = x + self.drop_path(self.time_mix(self.ln1(x)))
        
        # ChannelMix with residual
        x = x + self.drop_path(self.channel_mix(self.ln2(x)))
        
        return x


class RWKVBlock(nn.Module):
    """多层RWKV堆叠"""
    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            RWKVLayer(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] 或 [B, C, H, W]
        Returns:
            out: same shape as input
        """
        # 处理spatial输入 [B, C, H, W]
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
            is_spatial = True
        else:
            is_spatial = False
            B, N, C = x.shape
        
        # 通过RWKV层
        for layer in self.layers:
            x = layer(x)
        
        # 恢复spatial格式
        if is_spatial:
            x = x.transpose(1, 2).view(B, C, H, W)
        
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth)"""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


# 测试代码
if __name__ == '__main__':
    print("=" * 80)
    print("RWKV Module 测试")
    print("=" * 80)
    
    # 测试1: RWKVLayer (sequence input)
    print("\n【测试1】RWKVLayer - 序列输入")
    B, N, C = 2, 1024, 256
    x_seq = torch.randn(B, N, C)
    rwkv_layer = RWKVLayer(dim=256, num_heads=8)
    out = rwkv_layer(x_seq)
    print(f"✓ Input: {x_seq.shape} → Output: {out.shape}")
    assert out.shape == x_seq.shape
    
    # 测试2: RWKVBlock (spatial input)
    print("\n【测试2】RWKVBlock - 空间输入")
    B, C, H, W = 2, 256, 64, 64
    x_spatial = torch.randn(B, C, H, W)
    rwkv_block = RWKVBlock(dim=256, num_layers=2, num_heads=8)
    out = rwkv_block(x_spatial)
    print(f"✓ Input: {x_spatial.shape} → Output: {out.shape}")
    assert out.shape == x_spatial.shape
    
    # 测试3: 参数统计
    print("\n【测试3】参数统计")
    total_params = sum(p.numel() for p in rwkv_block.parameters())
    trainable_params = sum(p.numel() for p in rwkv_block.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    print("\n" + "=" * 80)
    print("✅ 所有测试通过！RWKV模块可用")
    print("=" * 80)
    print("\n用法示例：")
    print("  from rwkv_module import RWKVBlock")
    print("  rwkv = RWKVBlock(dim=256, num_layers=2)")
    print("  out = rwkv(dense_embeddings)  # [B, C, H, W]")
