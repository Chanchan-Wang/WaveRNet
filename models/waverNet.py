import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import cv2
import os
from SAM.modeling.mask_decoder import MaskDecoder
from SAM.modeling.prompt_encoder import PromptEncoder
from SAM.modeling.transformer import TwoWayTransformer
from SAM.modeling.common import LayerNorm2d
from SAM.modeling.image_encoder import ImageEncoderViT
from SAM.modeling.lora_encoder import ImageEncoderViT as LoRA
from SAM.modeling.small_encoder import TinyViT 
from SAM.modeling.adapter_encoder import ImageEncoderViT as Adapter
from functools import partial
from utils.transforms import ResizeLongestSide
from models.sdm import SimpleWaveletTransform
from models.fadf import MultiDomainAdapter
from models.hmpr import ProgressiveDecoderV4


def PointGenerator(mask, visual=False):
    np.random.seed(42)
    point_coord = []
    point_class = []
    box_coord = []

    if visual!=True:
        mask = mask.cpu().detach().numpy()
        # 彻底确保mask是2D的 [H, W]
        while mask.ndim > 2:
            mask = mask.squeeze()  # 去掉所有单维度
        if mask.ndim == 1:
            raise ValueError(f"Mask is 1D after squeeze, original shape issue")
    else:
        mask[mask < 255] = 0

    # 确保是uint8并且2D
    mask = mask.astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
    
    mask_shape = mask.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask * 255)
    for i in range(1, num_labels):
        row, col = int(centroids[i][0]), int(centroids[i][1])
        box_coord.append([stats[i][0], stats[i][1], stats[i][0]+stats[i][2], stats[i][1]+stats[i][3]])
        point_coord.append([row, col])
        point_class.append(1)

    return point_coord, point_class, mask_shape, box_coord


class SAMB(nn.Module):
    def __init__(self, data_path=None, img_size=1024, use_wavelet=False, use_adapter=False, use_progressive=False, use_rwkv=False, use_v4=False):
        super(SAMB, self).__init__()
        self.use_wavelet = use_wavelet
        self.use_adapter = use_adapter
        self.use_progressive = use_progressive
        self.use_rwkv = use_rwkv  # 是否使用RWKV增强
        self.use_v4 = use_v4  # 是否使用V4版本（真正512输出）
        
        # 用于存储Progressive Decoder的中间输出（用于多阶段损失计算）
        self.progressive_outputs = None

        self.image_encoder = Adapter(
            depth=12,
            embed_dim=768,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256
        )

        # self.image_encoder = TinyViT(img_size=1024, in_chans=3, num_classes=1000,
        #         embed_dims=[64, 128, 160, 320],
        #         depths=[2, 2, 6, 2],
        #         num_heads=[2, 4, 5, 10],
        #         window_sizes=[7, 7, 14, 7],
        #         mlp_ratio=4.,
        #         drop_rate=0.,
        #         drop_path_rate=0.0,
        #         use_checkpoint=False,
        #         mbconv_expand_ratio=4.0,
        #         local_conv_size=3,
        #         layer_lr_decay=0.8
        #     )


        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(img_size // 16, img_size // 16), # 1024 // 16
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
            )
        
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_domains=4,  # 支持4个域
        )
        
        # Wavelet Transform（可选）
        if self.use_wavelet:
            self.wavelet_transform = SimpleWaveletTransform(in_channels=256)
        
        # Multi-Domain Adapter（可选）
        if self.use_adapter:
            self.multi_domain_adapter = MultiDomainAdapter(embed_dim=256, num_domains=4)
        
        # Progressive Decoder V4（可选）
        if self.use_progressive:
            # 只使用V4版本
            self.progressive_decoder = ProgressiveDecoderV4(
                mask_decoder_256=self.mask_decoder,
                prompt_encoder=self.prompt_encoder,
                use_rwkv=self.use_rwkv if hasattr(self, 'use_rwkv') else True,  # ✅ 默认开启RWKV（老师要求）
                rwkv_layers=2,
                rwkv_heads=8
            )
        

        self.path = data_path
        self.img_size = img_size
        self.pt = ResizeLongestSide(img_size)

 
    def forward(self, x, mask=None, img_id=None, domain_id=None):

        b = x.shape[0]
        image_embeddings = self.image_encoder(x)
        
        # 应用Wavelet Transform（如果启用）
        if self.use_wavelet:
            image_embeddings = self.wavelet_transform(image_embeddings)
        
        # 应用Multi-Domain Adapter（如果启用）
        if self.use_adapter:
            image_embeddings = self.multi_domain_adapter(image_embeddings, domain_id)

        # 🚀 优化：prompt embeddings只计算一次（不在循环里）
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        image_pe = self.prompt_encoder.get_dense_pe()

        # 不再需要存储中间输出（已优化）
        
        outputs_mask = []
        all_ious = []
        
        for idx in range(b): # for each batch 
            
            # 提取当前batch元素的domain_id
            if domain_id is not None:
                if isinstance(domain_id, (list, tuple, torch.Tensor)):
                    # 如果是list/tuple/tensor，提取idx位置的元素
                    current_domain_id = domain_id[idx]
                else:
                    # 如果是单个值（int或单元素tensor），直接使用
                    current_domain_id = domain_id
                    
                # 确保是Python int（不是tensor）
                if isinstance(current_domain_id, torch.Tensor):
                    current_domain_id = int(current_domain_id.item())
                elif not isinstance(current_domain_id, int):
                    current_domain_id = int(current_domain_id)
            else:
                current_domain_id = 0  # 默认domain
            
            # 使用Progressive Decoder V2或原始Decoder
            if self.use_progressive:
                # Progressive Decoder V2路径（使用预先计算的embeddings）
                # 调用Progressive Decoder V2
                prog_outputs = self.progressive_decoder(
                    image_embeddings=image_embeddings[idx].unsqueeze(0),
                    image_pe=image_pe,  # 使用预先计算的
                    initial_sparse_embeddings=sparse_embeddings,  # 使用预先计算的
                    initial_dense_embeddings=dense_embeddings,  # 使用预先计算的
                    return_all_stages=False,  # 禁用中间输出，节省内存
                    domain_id=current_domain_id
                )
                
                # 使用最终的1024×1024作为主输出
                masks = prog_outputs['mask_1024']  # [1, 1, 1024, 1024]
                score = prog_outputs['iou_pred_2']  # [1, 1] IoU预测（第二个decoder的）
            else:
                # 原始Decoder路径（使用预先计算的embeddings，避免重复计算）
                low_res_masks, score = self.mask_decoder(
                    image_embeddings=image_embeddings[idx].unsqueeze(0),
                    image_pe=image_pe,  # 使用预先计算的image_pe
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    domain_id=current_domain_id
                )

                masks = F.interpolate(low_res_masks, (self.img_size, self.img_size), mode="bilinear", align_corners=False)

            outputs_mask.append(masks.squeeze(0))
            all_ious.append(score.squeeze(0))
            

        return torch.stack(outputs_mask, dim=0), torch.stack(all_ious, dim=0)
    
    def ensemble_predict(self, x, mask=None, img_id=None, trained_domains=None):
        """
        Ensemble预测：用所有训练过的域进行预测，选择IoU最高的结果
        用于域泛化实验中的unseen domain测试
        
        Args:
            x: 输入图像 [B, C, H, W]
            trained_domains: 训练过的域列表，例如[0, 1, 3]表示CHASE, DRIVE, STARE
        
        Returns:
            best_masks: IoU最高的mask预测 [B, 1, H, W]
            best_ious: 对应的IoU预测 [B, 1]
        """
        if trained_domains is None:
            trained_domains = [0, 1, 2, 3]  # 默认所有域
        
        b = x.shape[0]
        
        # 初始化：用第一个域的预测
        with torch.no_grad():
            best_masks, best_ious = self.forward(x, mask, img_id, domain_id=trained_domains[0])
        
        # 逐个域比较，只保留最好的（节省显存）
        for domain_idx in trained_domains[1:]:
            with torch.no_grad():
                pred_mask, pred_iou = self.forward(x, mask, img_id, domain_id=domain_idx)
                
                # 对每个batch样本，如果新预测的IoU更高，则替换
                for i in range(b):
                    if pred_iou[i, 0] > best_ious[i, 0]:
                        best_masks[i] = pred_mask[i]
                        best_ious[i] = pred_iou[i]
                
                # 立即释放当前域的预测，避免显存累积
                del pred_mask, pred_iou
                torch.cuda.empty_cache()  # 清理显存碎片
        
        return best_masks, best_ious
    
    def ensemble_predict_wavelet_soft(self, x, mask=None, img_id=None, trained_domains=None, domain_freq_stats=None):
        """
        基于小波频率相似度的Soft Weighting Ensemble（老师推荐的高级方法）
        
        老师的方案：
        1. 计算测试图的high/low frequency与各训练域的相似度
        2. 用相似度作为权重（softmax归一化）
        3. 用所有domain得到不同的image_embeddings（经过adapter后）
        4. 按权重融合embeddings："根据权重给它相乘"
        5. 融合后的embedding输入decoder
        
        Args:
            x: 输入图像 [B, C, H, W]
            trained_domains: 训练过的域列表，例如[0, 1, 2]
            domain_freq_stats: {domain_id: {'avg_high': tensor, 'avg_low': tensor}}
        
        Returns:
            pred_masks: 加权融合后的预测 [B, 1, H, W]
            pred_ious: IoU预测 [B, 1]
            domain_weights: 每个domain的权重 [B, num_domains]
        """
        if trained_domains is None:
            trained_domains = [0, 1, 2, 3]
        
        if domain_freq_stats is None:
            print("⚠️  domain_freq_stats未提供，使用简单ensemble")
            return self.ensemble_predict(x, mask, img_id, trained_domains)
        
        b = x.shape[0]
        device = x.device
        num_domains = len(trained_domains)
        
        # Step 1: 提取测试图像的频率特征
        with torch.no_grad():
            image_embeddings_base = self.image_encoder(x)  # [B, 256, 64, 64]
            
            if not self.use_wavelet:
                print("⚠️  模型未启用wavelet，使用均匀权重ensemble")
                pred_mask, pred_iou = self.ensemble_predict(x, mask, img_id, trained_domains)
                # 返回均匀权重
                b = x.shape[0]
                num_domains = len(trained_domains)
                uniform_weights = torch.ones(b, num_domains, device=x.device) / num_domains
                return pred_mask, pred_iou, uniform_weights
            
            # 获取高频和低频分量
            _, test_high, test_low = self.wavelet_transform(image_embeddings_base, return_components=True)
            # test_high: [B, 256, 64, 64]
            # test_low: [B, 256, 64, 64]
        
        # Step 2: 计算与各域的相似度权重
        all_weights = []  # [B, num_domains]
        
        for i in range(b):
            similarities = []
            
            for domain_id in trained_domains:
                if domain_id not in domain_freq_stats:
                    similarities.append(0.0)
                    continue
                
                # 获取该域的平均频率
                avg_high = domain_freq_stats[domain_id]['avg_high'].to(device)  # [256, 64, 64]
                avg_low = domain_freq_stats[domain_id]['avg_low'].to(device)    # [256, 64, 64]
                
                # 计算high frequency相似度
                sim_high = F.cosine_similarity(
                    test_high[i].flatten().unsqueeze(0),
                    avg_high.flatten().unsqueeze(0),
                    dim=1
                ).item()
                
                # 计算low frequency相似度
                sim_low = F.cosine_similarity(
                    test_low[i].flatten().unsqueeze(0),
                    avg_low.flatten().unsqueeze(0),
                    dim=1
                ).item()
                
                # 平均相似度
                similarity = (sim_high + sim_low) / 2.0
                similarities.append(similarity)
            
            # Softmax归一化得到权重
            similarities_tensor = torch.tensor(similarities, device=device)
            weights = F.softmax(similarities_tensor, dim=0)  # [num_domains]
            all_weights.append(weights)
        
        all_weights = torch.stack(all_weights)  # [B, num_domains]
        
        # Step 3: 用每个domain获取adapter后的embeddings并加权融合
        fused_embeddings = []
        
        for i in range(b):
            domain_embeddings = []
            
            # 对每个训练域获取adapter后的embedding
            for domain_id in trained_domains:
                # 过adapter（如果有）
                if self.use_adapter:
                    emb = self.multi_domain_adapter(
                        image_embeddings_base[i:i+1], 
                        domain_idx=domain_id,  # 正确的参数名
                        training=True
                    )  # [1, 256, 64, 64]
                else:
                    emb = image_embeddings_base[i:i+1]
                
                domain_embeddings.append(emb.squeeze(0))  # [256, 64, 64]
            
            # 按权重融合："根据权重给它相乘"
            domain_embeddings = torch.stack(domain_embeddings)  # [num_domains, 256, 64, 64]
            weights_expanded = all_weights[i].view(num_domains, 1, 1, 1)  # [num_domains, 1, 1, 1]
            
            fused_emb = torch.sum(domain_embeddings * weights_expanded, dim=0)  # [256, 64, 64]
            fused_embeddings.append(fused_emb)
        
        fused_embeddings = torch.stack(fused_embeddings)  # [B, 256, 64, 64]
        
        # Step 4: 用融合后的embedding通过decoder预测
        # 注意：这里使用domain_id=trained_domains[0]只是为了选择decoder的token
        # 实际的domain信息已经融合在embedding中了
        outputs_mask = []
        all_ious = []
        
        for idx in range(b):
            # 生成prompt embeddings
            points, labels_points, _, _ = PointGenerator(mask[idx:idx+1] if mask is not None else None)
            
            # 转换为tensor
            if len(points) > 0:
                points = torch.tensor(points, dtype=torch.float32, device=device).unsqueeze(0)  # [1, N, 2]
                labels_points = torch.tensor(labels_points, dtype=torch.int64, device=device).unsqueeze(0)  # [1, N]
            else:
                # 如果没有points，使用None
                points = None
                labels_points = None
            
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(points, labels_points) if points is not None else None,
                boxes=None,
                masks=None,
            )
            
            # 获取image_pe
            image_pe = self.prompt_encoder.get_dense_pe()
            
            # 使用融合的embedding + 第一个domain的decoder token
            if self.use_progressive:
                # Progressive decoder路径 (return_all_stages=False返回dict)
                prog_outputs = self.progressive_decoder(
                    image_embeddings=fused_embeddings[idx:idx+1],
                    image_pe=image_pe,
                    initial_sparse_embeddings=sparse_embeddings,
                    initial_dense_embeddings=dense_embeddings,
                    domain_id=trained_domains[0],  # 使用第一个domain的token
                    return_all_stages=False
                )
                # 从dict中提取结果
                masks = prog_outputs['mask_1024']
                score = prog_outputs['iou_pred_2']
            else:
                # 普通decoder路径
                low_res_masks, score = self.mask_decoder(
                    image_embeddings=fused_embeddings[idx:idx+1],
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    domain_id=trained_domains[0]
                )
                masks = F.interpolate(low_res_masks, (self.img_size, self.img_size), 
                                     mode="bilinear", align_corners=False)
            
            outputs_mask.append(masks.squeeze(0))
            all_ious.append(score.squeeze(0))
        
        final_masks = torch.stack(outputs_mask, dim=0)
        final_ious = torch.stack(all_ious, dim=0)
        
        return final_masks, final_ious, all_weights



class SAMB_Eval(nn.Module):
    def __init__(self, data_path=None, img_size=1024, use_wavelet=False, use_adapter=False, use_progressive=False, use_rwkv=False, use_v4=False):
        super(SAMB_Eval, self).__init__()
        self.use_wavelet = use_wavelet
        self.use_adapter = use_adapter
        self.use_progressive = use_progressive
        self.use_rwkv = use_rwkv
        self.use_v4 = use_v4

        self.image_encoder = Adapter(
            depth=12,
            embed_dim=768,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256
        )

        # self.image_encoder = TinyViT(img_size=1024, in_chans=3, num_classes=1000,
        #         embed_dims=[64, 128, 160, 320],
        #         depths=[2, 2, 6, 2],
        #         num_heads=[2, 4, 5, 10],
        #         window_sizes=[7, 7, 14, 7],
        #         mlp_ratio=4.,
        #         drop_rate=0.,
        #         drop_path_rate=0.0,
        #         use_checkpoint=False,
        #         mbconv_expand_ratio=4.0,
        #         local_conv_size=3,
        #         layer_lr_decay=0.8
        #     )


        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(img_size // 16, img_size // 16), # 1024 // 16
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
            )
        
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_domains=4,  # 支持4个域
        )
        
        # Wavelet Transform（可选）
        if self.use_wavelet:
            self.wavelet_transform = SimpleWaveletTransform(in_channels=256)
        
        # Multi-Domain Adapter（可选）
        if self.use_adapter:
            self.multi_domain_adapter = MultiDomainAdapter(embed_dim=256, num_domains=4)
        
        # Progressive Decoder V4（可选）
        if self.use_progressive:
            # 只使用V4版本
            self.progressive_decoder = ProgressiveDecoderV4(
                mask_decoder_256=self.mask_decoder,
                prompt_encoder=self.prompt_encoder,
                use_rwkv=self.use_rwkv if hasattr(self, 'use_rwkv') else True,  # ✅ 默认开启RWKV（老师要求）
                rwkv_layers=2,
                rwkv_heads=8
            )
        

        self.path = data_path
        self.img_size = img_size
        self.pt = ResizeLongestSide(img_size)

 
    def forward(self, x, mask=None, img_id=None, domain_id=None):

        b = x.shape[0]
        image_embeddings = self.image_encoder(x)
        
        # 应用Wavelet Transform（如果启用）
        if self.use_wavelet:
            image_embeddings = self.wavelet_transform(image_embeddings)
        
        # 应用Multi-Domain Adapter（如果启用）
        if self.use_adapter:
            image_embeddings = self.multi_domain_adapter(image_embeddings, domain_id)

        # 🚀 优化：prompt embeddings只计算一次（所有模式都适用）
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        image_pe = self.prompt_encoder.get_dense_pe()

        outputs_mask = []
        all_ious = []
        
        # 如果使用Progressive Decoder V2
        if self.use_progressive:
            for idx in range(b):
                # Progressive Decoder V2返回dict格式
                prog_outputs = self.progressive_decoder(
                    image_embeddings=image_embeddings[idx].unsqueeze(0),
                    image_pe=image_pe,  # 使用预计算的
                    initial_sparse_embeddings=sparse_embeddings,  # 使用预计算的
                    initial_dense_embeddings=dense_embeddings,  # 使用预计算的
                    return_all_stages=False,  # eval时只需要最终输出
                    domain_id=0  # eval模式的默认domain
                )
                
                # 从dict中提取结果
                mask_1024_logits = prog_outputs['mask_1024']
                score = prog_outputs['iou_pred_2']
                
                outputs_mask.append(mask_1024_logits.squeeze(0))
                all_ious.append(score.squeeze(0))
        else:
            # 原始MaskDecoder路径（使用预先计算的embeddings）
            for idx in range(b):
                
                # 提取当前batch元素的domain_id
                if domain_id is not None:
                    if isinstance(domain_id, (list, tuple, torch.Tensor)):
                        current_domain_id = domain_id[idx]
                    else:
                        current_domain_id = domain_id
                        
                    if isinstance(current_domain_id, torch.Tensor):
                        current_domain_id = int(current_domain_id.item())
                    elif not isinstance(current_domain_id, int):
                        current_domain_id = int(current_domain_id)
                else:
                    current_domain_id = 0
        
                low_res_masks, score = self.mask_decoder(
                    image_embeddings=image_embeddings[idx].unsqueeze(0),
                    image_pe=image_pe,  # 使用预先计算的image_pe
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    domain_id=current_domain_id
                )

                best_iou_inds = torch.argmax(score, dim=-1)
                best_idx = best_iou_inds[0].item()  # 转换为Python int
                low_res_masks = low_res_masks[0:1, best_idx:best_idx+1]  # [1, 1, 256, 256]

                masks = F.interpolate(low_res_masks, (self.img_size, self.img_size), mode="bilinear", align_corners=False)

                outputs_mask.append(masks.squeeze(0))
                all_ious.append(score.squeeze(0))
            

        return torch.stack(outputs_mask, dim=0), torch.stack(all_ious, dim=0)

