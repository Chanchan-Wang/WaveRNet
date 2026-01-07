"""
Wavelet Transform Module - 小波变换模块
作为独立的创新点，在Image Encoder之后使用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleWaveletTransform(nn.Module):
    """
    简化版小波变换：用可学习的卷积模拟小波的高频/低频分解
    
    特点：
    - 独立模块（不是Adapter的一部分）
    - 使用残差连接
    - 不使用BatchNorm（避免与Multi-Domain Adapter冲突）
    - 维度保持不变
    """
    def __init__(self, in_channels=256, residual_weight=0.1):
        super().__init__()
        
        self.residual_weight = nn.Parameter(torch.tensor(residual_weight))
        
        # 高频分支：提取边缘、纹理等细节信息
        self.high_freq_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
        )
        
        # 低频分支：提取整体结构信息
        self.low_freq_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
        )
        
        # 融合层：将高频和低频信息融合
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x, return_components=False):
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C, H, W]，来自Image Encoder
            return_components: 是否返回高频和低频分量（用于域泛化）
            
        Returns:
            如果return_components=False: 增强后的特征 [B, C, H, W]
            如果return_components=True: (output, high_freq, low_freq)
        """
        # ⚡ 并行计算两个分支（cudnn会自动优化）
        high_freq = self.high_freq_branch(x)
        low_freq = self.low_freq_branch(x)
        
        # 融合
        fused = self.fusion(torch.cat([high_freq, low_freq], dim=1))
        
        # 残差连接
        output = x + self.residual_weight * fused
        
        if return_components:
            return output, high_freq, low_freq
        return output
    
    def get_stats(self):
        """返回模块统计信息（用于调试）"""
        return {
            'residual_weight': self.residual_weight.item(),
            'num_params': sum(p.numel() for p in self.parameters()),
        }


# 用于测试
if __name__ == '__main__':
    # 测试模块
    wavelet = SimpleWaveletTransform(in_channels=256)
    x = torch.randn(2, 256, 64, 64)
    y = wavelet(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Stats: {wavelet.get_stats()}")
    
    assert x.shape == y.shape, "Shape should not change"
    print("✓ Test passed!")
