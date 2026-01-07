"""
Progressive Decoder V4 - 修复decoder_512输出问题
关键改进：
1. decoder_512内部真正输出512×512（不是外部上采样）
2. 保留RWKV增强
3. 修改decoder_512的output_upscaling层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from models.rwkv import RWKVBlock
from SAM.modeling.common import LayerNorm2d


class MaskDecoder512(nn.Module):
    """
    修改版MaskDecoder，内部上采样到512×512
    基于原始MaskDecoder，添加额外的上采样层
    """
    def __init__(self, base_decoder):
        """
        Args:
            base_decoder: 原始的256 MaskDecoder
        """
        super().__init__()
        
        # 复制所有原始decoder的组件
        self.transformer_dim = base_decoder.transformer_dim
        self.transformer = base_decoder.transformer
        self.num_multimask_outputs = base_decoder.num_multimask_outputs
        self.num_domains = base_decoder.num_domains
        self.iou_token = base_decoder.iou_token
        self.num_mask_tokens = base_decoder.num_mask_tokens
        self.mask_tokens = base_decoder.mask_tokens
        self.output_hypernetworks_mlps = base_decoder.output_hypernetworks_mlps
        self.iou_prediction_head = base_decoder.iou_prediction_head
        
        # ===== 关键修改：扩展output_upscaling到512 =====
        # 原始: 256 → 64 → 32 (输出256×256)
        # 修改: 256 → 64 → 32 → 16 (输出512×512)
        
        transformer_dim = self.transformer_dim
        self.output_upscaling = nn.Sequential(
            # Stage 1: 64×64 → 128×128, 256→64
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            
            # Stage 2: 128×128 → 256×256, 64→32
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
            
            # ✨ Stage 3 (新增): 256×256 → 512×512, 32→16 ✨
            nn.ConvTranspose2d(transformer_dim // 8, transformer_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 16),
            nn.GELU(),
        )
        
        # 更新output_hypernetworks_mlps以匹配新的输出维度
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 16, 3)  # 改为16
                for i in range(self.num_mask_tokens)
            ]
        )
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        domain_id: int = 0,
    ):
        """
        Forward pass，输出512×512的mask
        
        Args:
            image_embeddings: [B, 256, 64, 64]
            image_pe: [1, 256, 64, 64]
            sparse_prompt_embeddings: [B, N, 256]
            dense_prompt_embeddings: [B, 256, 64, 64]
            multimask_output: bool
            domain_id: int (0-3)
        
        Returns:
            masks: [B, 1, 512, 512]  ← 关键：输出512
            iou_pred: [B, 1]
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            domain_id=domain_id,
        )
        
        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        
        return masks, iou_pred
    
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        domain_id: int = 0,
    ):
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        b = image_embeddings.shape[0]
        
        # 使用domain_id选择对应的token
        domain_idx = int(domain_id)
        iou_token_out = self.iou_token.weight[domain_idx:domain_idx+1].unsqueeze(0).expand(b, -1, -1)
        
        mask_tokens_start = domain_idx * self.num_mask_tokens
        mask_tokens_end = mask_tokens_start + self.num_mask_tokens
        mask_tokens_out = self.mask_tokens.weight[mask_tokens_start:mask_tokens_end].unsqueeze(0).expand(b, -1, -1)
        
        output_tokens = torch.cat([iou_token_out, mask_tokens_out], dim=1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        
        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        
        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, self.transformer_dim, 64, 64)
        upscaled_embedding = self.output_upscaling(src)  # [B, 16, 512, 512] ← 关键
        
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        
        masks = (hyper_in @ upscaled_embedding.view(b, self.transformer_dim // 16, -1)).view(
            b, -1, 512, 512  # ← 关键：输出512×512
        )
        
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        
        return masks, iou_pred


class MLP(nn.Module):
    """Simple MLP"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class ProgressiveDecoderV4(nn.Module):
    """
    Progressive Decoder V4 - 真正的512输出 + RWKV增强
    
    关键改进：
    1. decoder_512内部输出512×512（修改output_upscaling）
    2. 保留RWKV增强embeddings
    3. 不需要外部上采样
    """
    
    def __init__(
        self,
        mask_decoder_256,
        prompt_encoder,
        use_rwkv: bool = True,
        rwkv_layers: int = 2,
        rwkv_heads: int = 8,
    ):
        """
        Args:
            mask_decoder_256: 原始SAM的MaskDecoder (输出256)
            prompt_encoder: SAM的PromptEncoder
            use_rwkv: 是否使用RWKV增强
            rwkv_layers: RWKV层数
            rwkv_heads: RWKV注意力头数
        """
        super().__init__()
        
        # 第一个Decoder：256×256（原始SAM）
        self.mask_decoder_256 = mask_decoder_256
        
        # Prompt Encoder
        self.prompt_encoder = prompt_encoder
        
        # ✨ 第二个Decoder：512×512（真正的512输出）✨
        self.mask_decoder_512 = MaskDecoder512(mask_decoder_256)
        
        # RWKV增强模块
        self.use_rwkv = use_rwkv
        if self.use_rwkv:
            self.rwkv_dense = RWKVBlock(
                dim=256,
                num_layers=rwkv_layers,
                num_heads=rwkv_heads,
                mlp_ratio=4.0,
                drop_path=0.1
            )
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        initial_sparse_embeddings: torch.Tensor,
        initial_dense_embeddings: torch.Tensor,
        domain_id: int = 0,
        return_all_stages: bool = False,
    ):
        """
        Forward pass
        
        Returns:
            如果return_all_stages=True:
                dict with keys: 'mask_256', 'mask_512', 'mask_1024', 'iou_pred'
            否则:
                masks: [B, 1, 1024, 1024]
                iou_pred: [B, 1]
        """
        # ===== Stage 1: 第一个Decoder (256×256) =====
        low_res_masks_256, iou_pred_1 = self.mask_decoder_256(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=initial_sparse_embeddings,
            dense_prompt_embeddings=initial_dense_embeddings,
            multimask_output=False,
            domain_id=domain_id,
        )  # [B, 1, 256, 256]
        
        # ===== Stage 2: Mask作为Prompt =====
        sparse_emb_from_mask, dense_emb_from_mask = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=low_res_masks_256,
        )
        
        # ===== Stage 2.5: RWKV增强 =====
        if self.use_rwkv:
            dense_emb_enhanced = self.rwkv_dense(dense_emb_from_mask)
            sparse_emb_enhanced = sparse_emb_from_mask
        else:
            dense_emb_enhanced = dense_emb_from_mask
            sparse_emb_enhanced = sparse_emb_from_mask
        
        # ===== Stage 3: 第二个Decoder (512×512) =====
        # ✨ 关键：直接输出512，不需要外部上采样 ✨
        low_res_masks_512, iou_pred_2 = self.mask_decoder_512(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb_enhanced,
            dense_prompt_embeddings=dense_emb_enhanced,
            multimask_output=False,
            domain_id=domain_id,
        )  # [B, 1, 512, 512] ← 真正的512！
        
        # ===== Stage 4: 上采样到1024 =====
        masks_1024 = F.interpolate(
            low_res_masks_512,
            size=(1024, 1024),
            mode='bilinear',
            align_corners=False
        )  # [B, 1, 1024, 1024]
        
        # 始终返回字典格式，但只在需要时包含中间输出
        result = {
            'mask_1024': masks_1024,
            'iou_pred_2': iou_pred_2,
        }
        
        if return_all_stages:
            result['mask_256'] = low_res_masks_256
            result['mask_512'] = low_res_masks_512
        
        return result


if __name__ == "__main__":
    print("=" * 60)
    print("Progressive Decoder V4 - 真正的512输出")
    print("=" * 60)
    print("\n关键改进:")
    print("  1. ✅ decoder_512内部输出512×512")
    print("  2. ✅ 修改output_upscaling: 256→64→32→16")
    print("  3. ✅ 保留RWKV增强")
    print("  4. ✅ 不需要外部上采样")
    print("\n预期提升: +0.5-1.0% IoU")
    print("=" * 60)
