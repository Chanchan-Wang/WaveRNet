"""
域泛化训练脚本 - 基于小波频率相似度的Soft Weighting
LODO (Leave-One-Domain-Out) 训练策略
训练后统计每个域的平均频率特征用于测试时的domain选择
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import time
from torch.optim import lr_scheduler
import pandas as pd
import argparse
from datasets.retinal_dataset import BinaryLoader
from utils.loss import *
from tqdm import tqdm
import json
from models.waverNet import SAMB
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from monai.metrics import MeanIoU

torch.set_num_threads(8)

# 启用cuDNN优化
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def compute_domain_freq_stats(model, dataloader, trained_domains, device='cuda'):
    """
    计算训练域的平均频率特征
    老师要求：统计每个域的平均 high frequency 和 low frequency
    """
    print(f"\n{'='*80}")
    print("📊 统计训练域的频率特征（用于域泛化测试）")
    print(f"{'='*80}")
    
    # 如果模型没有wavelet，直接返回空统计
    if not (hasattr(model, 'wavelet_transform') and model.use_wavelet):
        print("⚠️  模型未启用Wavelet，跳过频率统计")
        print(f"{'='*80}\n")
        return {}
    
    model.eval()
    
    # 初始化每个域的特征收集器
    domain_features = {domain_id: {'high': [], 'low': []} for domain_id in trained_domains}
    
    with torch.no_grad():
        for _, img, labels, img_id, domain_id in tqdm(dataloader, desc='统计频率'):
            img = img.to(device)
            b = img.shape[0]
            
            # 提取小波频率特征（批量处理）
            image_embeddings = model.image_encoder(img)  # [B, 256, 64, 64]
            
            if hasattr(model, 'wavelet_transform') and model.use_wavelet:
                # 获取高频和低频分量（批量）
                _, high_freq, low_freq = model.wavelet_transform(
                    image_embeddings, return_components=True
                )
                
                # 处理batch中的每个样本
                for i in range(b):
                    domain_id_val = domain_id[i].item()
                    
                    if domain_id_val not in trained_domains:
                        continue
                    
                    # 收集该域的特征
                    domain_features[domain_id_val]['high'].append(high_freq[i:i+1].cpu())
                    domain_features[domain_id_val]['low'].append(low_freq[i:i+1].cpu())
    
    # 计算每个域的平均特征
    domain_freq_stats = {}
    
    for domain_id in trained_domains:
        if len(domain_features[domain_id]['high']) == 0:
            print(f"⚠️  Domain {domain_id} 没有训练样本")
            continue
        
        # 计算平均
        avg_high = torch.mean(torch.cat(domain_features[domain_id]['high'], dim=0), dim=0)
        avg_low = torch.mean(torch.cat(domain_features[domain_id]['low'], dim=0), dim=0)
        
        domain_freq_stats[domain_id] = {
            'avg_high': avg_high,  # [256, 64, 64]
            'avg_low': avg_low     # [256, 64, 64]
        }
        
        print(f"✓ Domain {domain_id}: {len(domain_features[domain_id]['high'])} 样本")
    
    print(f"✅ 频率统计完成")
    print(f"{'='*80}\n")
    
    return domain_freq_stats


def train_model(model, criterion_mask, optimizer, scheduler, dataloaders, 
                num_epochs=100, output_dir='../outputs/dg', 
                trained_domains=None, test_domain_name=''):
    """训练函数（域泛化版本）"""
    since = time.time()
    
    Loss_list = {'train': [], 'valid': []}
    Accuracy_list = {'train': [], 'valid': []}
    
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    best_epoch = 0
    
    mse_loss = torch.nn.MSELoss()
    monai_iou = MeanIoU(include_background=False, reduction='none')
    
    # 创建GradScaler用于混合精度训练
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(start_epoch, num_epochs):
        print(f'\nEpoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss_mask = []
            running_corrects_mask = []
            running_loss_mse = []
        
            pbar = tqdm(dataloaders[phase], 
                       desc=phase,
                       disable=False,
                       leave=True,
                       ncols=80,
                       ascii=False,
                       file=sys.stdout)
            
            for _, img, labels, img_id, domain_id in pbar:
                img = Variable(img.cuda())
                labels = Variable(labels.cuda())

                # 使用混合精度
                with torch.set_grad_enabled(phase == 'train'):
                    # 前向传播使用AMP
                    with torch.amp.autocast('cuda'):
                        pred_mask, pred_iou = model(x=img, mask=labels, img_id=img_id, domain_id=domain_id)
                        pred_mask = torch.sigmoid(pred_mask)
                    
                    # Loss计算在FP32
                    loss1 = criterion_mask(pred_mask.float(), labels.float())
                    score_mask1 = monai_iou(pred_mask.float(), labels.float())
                    mse = mse_loss(pred_iou.float(), score_mask1.float())
                    loss = loss1 + mse

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)  # ⚡ 用set_to_none加速
                    
                # 记录指标
                running_loss_mask.append(loss1.item())
                running_corrects_mask.append(torch.mean(score_mask1).item())
                running_loss_mse.append(mse.item())

            # 计算epoch平均值
            epoch_loss = np.mean(running_loss_mask)
            epoch_acc = np.mean(running_corrects_mask)
            epoch_mse = np.mean(running_loss_mse)
            
            Loss_list[phase].append(epoch_loss)
            Accuracy_list[phase].append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} IoU: {epoch_acc:.4f}')
            
            if phase == 'valid' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                best_epoch = epoch
                save_path = os.path.join(output_dir, f'best_model_epoch{epoch}.pth')
                torch.save(best_model_wts, save_path)
                print(f'✓ Saved best model (loss: {best_loss:.4f})')
                
            if phase == 'valid':
                scheduler.step()
                
                # 保存checkpoint用于续训
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'best_epoch': best_epoch
                }
                torch.save(checkpoint_data, os.path.join(output_dir, 'checkpoint.pth'))

        print(f'Epoch {epoch} completed')
        
    time_elapsed = time.time() - since
    print(f'\n{"="*80}')
    print(f'✅ Training Complete!')
    print(f'{"="*80}')
    print(f'⏱️  Time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'🏆 Best val loss: {best_loss:.4f} (epoch {best_epoch})')
    print(f'🎯 Test domain: {test_domain_name} (unseen)')
    print(f'📁 Saved to: {output_dir}')
    print(f'{"="*80}')
    
    # 加载最佳模型
    model.load_state_dict(best_model_wts)
    
    # 统计频率特征（用于测试时的domain选择）
    domain_freq_stats = compute_domain_freq_stats(
        model, dataloaders['train'], trained_domains
    )
    
    # 保存最终模型和频率统计
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'domain_freq_stats': domain_freq_stats,
        'trained_domains': trained_domains,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'config': {
            'use_wavelet': use_wavelet,
            'use_adapter': use_adapter,
            'use_progressive': use_progressive,
            'use_v4': use_progressive  # V4与Progressive绑定
        }
    }
    torch.save(checkpoint, os.path.join(output_dir, 'final_checkpoint.pth'))
    print(f"✅ 已保存: 模型 + 频率统计 → {output_dir}/final_checkpoint.pth")
    
    # 保存训练数据
    df_train = pd.DataFrame({
        'epoch': range(len(Loss_list['train'])),
        'loss': Loss_list['train'],
        'iou': Accuracy_list['train']
    })
    df_train.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    
    df_valid = pd.DataFrame({
        'epoch': range(len(Loss_list['valid'])),
        'loss': Loss_list['valid'],
        'iou': Accuracy_list['valid']
    })
    df_valid.to_csv(os.path.join(output_dir, 'valid_data.csv'), index=False)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(Loss_list['train'], label='Train Loss')
    plt.plot(Loss_list['valid'], label='Valid Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(Accuracy_list['train'], label='Train IoU')
    plt.plot(Accuracy_list['valid'], label='Valid IoU')
    plt.title('IoU')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    return model, Loss_list, Accuracy_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='域泛化训练 (LODO)')
    parser.add_argument('--data_config', type=str, required=True, 
                       help='LODO数据配置文件 (e.g., config_dg_test_chase.json)')
    parser.add_argument('--sam_pretrain', type=str, default='../sam_vit_b_01ec64.pth')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--use_wavelet', type=str, default='True', help='是否使用Wavelet (True/False)')
    parser.add_argument('--use_adapter', type=str, default='True', help='是否使用Adapter (True/False)')
    parser.add_argument('--use_progressive', type=str, default='True', help='是否使用Progressive (True/False)')
    parser.add_argument('--resume_from', type=str, help='续训checkpoint路径')
    args = parser.parse_args()
    
    # 转换字符串参数为布尔值
    use_wavelet = args.use_wavelet.lower() == 'true'
    use_adapter = args.use_adapter.lower() == 'true'
    use_progressive = args.use_progressive.lower() == 'true'

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("🚀 域泛化训练 (Leave-One-Domain-Out)")
    print("="*80)

    # 加载LODO配置
    with open(args.data_config, 'r') as f:
        data_config = json.load(f)
        train_files = data_config['train']
        val_files = data_config['valid']
        test_files = data_config['test']
        trained_domains = data_config['trained_domains']
        test_domain = data_config['test_domain']
        test_domain_name = data_config['test_domain_name']
    
    print(f"📊 数据配置:")
    print(f"  训练: {len(train_files)} 样本 (域: {trained_domains})")
    print(f"  验证: {len(val_files)} 样本")
    print(f"  测试: {len(test_files)} 样本 ({test_domain_name}, unseen)")
    
    # 域映射
    DOMAIN_MAP = {'CHASE': 0, 'DRIVE': 1, 'RECOVERY': 2, 'STARE': 3}
    train_domain_list = {}
    val_domain_list = {}
    
    for domain_name, domain_id in DOMAIN_MAP.items():
        if domain_id in trained_domains:
            train_domain_list[domain_id] = [f for f in train_files if domain_name in f]
            val_domain_list[domain_id] = [f for f in val_files if domain_name in f]
    
    # 显示各域分布
    print(f"  域分布: ", end='')
    for domain_id in trained_domains:
        domain_name = [k for k, v in DOMAIN_MAP.items() if v == domain_id][0]
        print(f"{domain_name}({len(train_domain_list[domain_id])})", end=' ')
    print()

    # 创建数据加载器
    train_dataset = BinaryLoader("mask_1024", train_files, A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ], 
        additional_targets={'mask2': 'mask'}), domain_list=train_domain_list)
        
    val_dataset = BinaryLoader("mask_1024", val_files, A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ],
        additional_targets={'mask2': 'mask'}), domain_list=val_domain_list)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch, 
        shuffle=True, 
        drop_last=True,
        num_workers=4,  # 增加到4，加速数据加载
        pin_memory=True,
        prefetch_factor=4,  # 增加预加载
        persistent_workers=True  # 保持worker进程
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=2,  # 保持和训练一致
        num_workers=4,  # 数据加载加速（不影响结果）
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    
    dataloaders = {'train': train_loader, 'valid': val_loader}

    # 创建模型（根据参数配置）
    w_icon = '✓' if use_wavelet else '✗'
    a_icon = '✓' if use_adapter else '✗'
    p_icon = '✓' if use_progressive else '✗'
    print(f"🏗️  模型配置: W={w_icon} A={a_icon} P={p_icon}")
    
    model = SAMB(
        data_path=None,
        use_wavelet=use_wavelet,
        use_adapter=use_adapter,
        use_progressive=use_progressive,
        use_v4=True if use_progressive else False
    )

    # 加载SAM预训练
    encoder_dict = torch.load(args.sam_pretrain)
    
    pre_dict = {k: v for k, v in encoder_dict.items() if list(k.split('.'))[0] == 'image_encoder'}
    model.load_state_dict(pre_dict, strict=False)

    pre_dict = {k: v for k, v in encoder_dict.items() if list(k.split('.'))[0] == 'prompt_encoder'}
    model.load_state_dict(pre_dict, strict=False)

    # 多GPU支持
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.cuda()

    # 续训逻辑
    start_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"🔄 从checkpoint续训: {args.resume_from}")
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"   从epoch {start_epoch}继续训练")
    elif args.resume_from:
        print(f"⚠️  checkpoint文件不存在: {args.resume_from}")
        print("   将从头开始训练")

    # 统计参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"📦 参数: {trainable/1e6:.1f}M/{total/1e6:.1f}M ({trainable/total*100:.0f}%)")

    # 训练设置
    criterion_mask = BinaryMaskLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    print("="*80)
    print(f"▶️  训练 {args.epoch} Epochs...")
    model, Loss_list, Accuracy_list = train_model(
        model, criterion_mask, optimizer, exp_lr_scheduler, 
        dataloaders, num_epochs=args.epoch, output_dir=args.output_dir,
        trained_domains=trained_domains, test_domain_name=test_domain_name
    )
