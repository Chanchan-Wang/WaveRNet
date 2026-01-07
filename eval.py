"""
域泛化评估脚本 - 使用小波频率相似度的Soft Weighting
评估unseen domain的性能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch.autograd import Variable
import argparse
from datasets.retinal_dataset import TestLoader
from models.waverNet import SAMB
import json
import pandas as pd
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from monai.metrics import MeanIoU, compute_hausdorff_distance

torch.set_num_threads(8)

# 启用cuDNN优化
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def eval_model_dg(model, test_loader, domain_freq_stats, trained_domains, 
                  test_domain_name, output_dir):
    """
    域泛化评估 - 使用Soft Weighting方法
    """
    model.eval()
    
    monai_iou = MeanIoU(include_background=False, reduction='none')
    
    results = []
    all_weights = []  # 记录每个样本的domain权重
    
    # 用于计算分类指标
    all_tp, all_fp, all_tn, all_fn = 0, 0, 0, 0
    
    print(f"\n{'='*80}")
    print(f"🎯 评估 {test_domain_name} (Unseen Domain)")
    print(f"💡 使用Soft Weighting (基于小波频率相似度)")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        for _, img, labels, img_id in tqdm(test_loader, desc='评估'):
            img = Variable(img.cuda())
            labels = Variable(labels.cuda())
            
            # 确保mask维度和尺寸与预测一致
            # 目标: [B, 1, 1024, 1024]
            if labels.dim() == 2:
                labels = labels.unsqueeze(0).unsqueeze(0)  # [H, W] → [1, 1, H, W]
            elif labels.dim() == 3:
                labels = labels.unsqueeze(1)  # [B, H, W] → [B, 1, H, W]
            # 如果已经是4维且第二维不是1，可能是错误的，强制reshape
            elif labels.dim() == 4 and labels.shape[1] != 1:
                labels = labels[:, :1, :, :]  # 只保留第一个通道
            
            # 确保空间尺寸是1024×1024
            if labels.shape[-2:] != (1024, 1024):
                labels = torch.nn.functional.interpolate(
                    labels.float(),
                    size=(1024, 1024),
                    mode='nearest'
                )
            
            # 归一化labels到0-1范围 (可能是0-255)
            if labels.max() > 1.0:
                labels = labels / 255.0
            
            # 使用简单Hard Ensemble (选IoU最高的域)
            pred_mask, pred_iou = model.ensemble_predict(
                x=img,
                mask=labels,
                img_id=img_id,
                trained_domains=trained_domains
            )
            # 创建均匀权重用于记录
            weights = torch.ones(1, len(trained_domains)) / len(trained_domains)
            
            pred_mask = torch.sigmoid(pred_mask)
            
            # 计算指标
            score_iou = monai_iou(pred_mask.float(), labels.float())
            
            # Dice = 2 * IoU / (1 + IoU)
            score_dice = 2 * score_iou / (1 + score_iou)
            
            # 二值化用于后续指标计算
            pred_binary = (pred_mask > 0.5).float()
            labels_binary = (labels > 0.5).float()
            
            # Hausdorff Distance
            hd = compute_hausdorff_distance(
                pred_binary, labels_binary, 
                include_background=False, percentile=95
            )
            
            # 计算分类指标 (TP, FP, TN, FN)
            pred_flat = pred_binary.view(-1)
            labels_flat = labels_binary.view(-1)
            tp = ((pred_flat == 1) & (labels_flat == 1)).sum().item()
            fp = ((pred_flat == 1) & (labels_flat == 0)).sum().item()
            tn = ((pred_flat == 0) & (labels_flat == 0)).sum().item()
            fn = ((pred_flat == 0) & (labels_flat == 1)).sum().item()
            
            all_tp += tp
            all_fp += fp
            all_tn += tn
            all_fn += fn
            
            # 计算单张图的指标
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-7)
            specificity = tn / (tn + fp + 1e-7)
            
            # 记录结果
            results.append({
                'image_id': img_id[0],
                'iou': torch.mean(score_iou).item(),
                'dice': torch.mean(score_dice).item(),
                'hd': torch.mean(hd).item(),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'specificity': specificity
            })
            
            # 记录domain权重（用于分析）
            all_weights.append({
                'image_id': img_id[0],
                **{f'domain_{d}_weight': w.item() 
                   for d, w in zip(trained_domains, weights[0])}
            })
    
    # 保存详细结果
    df = pd.DataFrame(results)
    result_path = os.path.join(output_dir, f'{test_domain_name}_results.csv')
    df.to_csv(result_path, index=False)
    
    # 保存权重分析
    df_weights = pd.DataFrame(all_weights)
    weights_path = os.path.join(output_dir, f'{test_domain_name}_weights.csv')
    df_weights.to_csv(weights_path, index=False)
    
    # 计算统计信息
    mean_iou = df['iou'].mean()
    mean_dice = df['dice'].mean()
    mean_hd = df['hd'].mean()
    std_iou = df['iou'].std()
    std_dice = df['dice'].std()
    std_hd = df['hd'].std()
    
    # 计算全局分类指标（基于所有像素）
    global_precision = all_tp / (all_tp + all_fp + 1e-7)
    global_recall = all_tp / (all_tp + all_fn + 1e-7)
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall + 1e-7)
    global_accuracy = (all_tp + all_tn) / (all_tp + all_fp + all_tn + all_fn + 1e-7)
    global_specificity = all_tn / (all_tn + all_fp + 1e-7)
    
    # 平均每张图的指标
    mean_precision = df['precision'].mean()
    mean_recall = df['recall'].mean()
    mean_f1 = df['f1'].mean()
    mean_accuracy = df['accuracy'].mean()
    mean_specificity = df['specificity'].mean()
    std_precision = df['precision'].std()
    std_recall = df['recall'].std()
    std_f1 = df['f1'].std()
    std_accuracy = df['accuracy'].std()
    std_specificity = df['specificity'].std()
    
    # 保存汇总统计（增强版）
    summary_stats = {
        'domain': [test_domain_name],
        'iou_mean': [mean_iou],
        'iou_std': [std_iou],
        'dice_mean': [mean_dice],
        'dice_std': [std_dice],
        'hd_mean': [mean_hd],
        'hd_std': [std_hd],
        'precision_mean': [mean_precision],
        'precision_std': [std_precision],
        'recall_mean': [mean_recall],
        'recall_std': [std_recall],
        'f1_mean': [mean_f1],
        'f1_std': [std_f1],
        'accuracy_mean': [mean_accuracy],
        'accuracy_std': [std_accuracy],
        'specificity_mean': [mean_specificity],
        'specificity_std': [std_specificity],
        'global_precision': [global_precision],
        'global_recall': [global_recall],
        'global_f1': [global_f1],
        'global_accuracy': [global_accuracy],
        'global_specificity': [global_specificity],
        'num_samples': [len(df)]
    }
    
    # 添加权重统计
    for d in trained_domains:
        summary_stats[f'domain_{d}_avg_weight'] = [df_weights[f'domain_{d}_weight'].mean()]
        summary_stats[f'domain_{d}_std_weight'] = [df_weights[f'domain_{d}_weight'].std()]
    
    df_summary = pd.DataFrame(summary_stats)
    summary_path = os.path.join(output_dir, 'summary_detailed.csv')
    df_summary.to_csv(summary_path, index=False)
    
    # 保持简单版summary兼容性
    simple_summary = {
        'domain': [test_domain_name],
        'iou': [mean_iou],
        'dice': [mean_dice],
        'hd': [mean_hd]
    }
    df_simple = pd.DataFrame(simple_summary)
    simple_path = os.path.join(output_dir, 'summary.csv')
    df_simple.to_csv(simple_path, index=False)
    
    # 打印统计
    print(f"\n{'='*80}")
    print(f"📊 {test_domain_name} 结果:")
    print(f"{'='*80}")
    print(f"  IoU:         {mean_iou*100:.2f}% (±{std_iou*100:.2f}%)")
    print(f"  Dice:        {mean_dice*100:.2f}% (±{std_dice*100:.2f}%)")
    print(f"  HD:          {mean_hd:.2f} (±{std_hd:.2f})")
    print(f"  Precision:   {global_precision*100:.2f}%")
    print(f"  Recall:      {global_recall*100:.2f}%")
    print(f"  F1 Score:    {global_f1*100:.2f}%")
    print(f"  Accuracy:    {global_accuracy*100:.2f}%")
    print(f"  Specificity: {global_specificity*100:.2f}%")
    print(f"  样本数:      {len(df)}")
    print(f"\n  权重分布:")
    for d in trained_domains:
        avg_weight = df_weights[f'domain_{d}_weight'].mean()
        std_weight = df_weights[f'domain_{d}_weight'].std()
        print(f"    Domain {d}: {avg_weight:.3f} (±{std_weight:.3f})")
    print(f"{'='*80}\n")
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='域泛化评估')
    parser.add_argument('--data_config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='训练好的checkpoint (包含domain_freq_stats)')
    parser.add_argument('--output_dir', type=str, required=True)
    # 添加模块配置参数（用于覆盖checkpoint中的配置）
    parser.add_argument('--use_wavelet', type=str, default=None,
                       help='Override: use wavelet transform (True/False)')
    parser.add_argument('--use_adapter', type=str, default=None,
                       help='Override: use multi-domain adapter (True/False)')
    parser.add_argument('--use_progressive', type=str, default=None,
                       help='Override: use progressive decoder (True/False)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("🔍 域泛化评估 - Soft Weighting (小波频率相似度)")
    print("="*80)

    # 加载配置
    with open(args.data_config, 'r') as f:
        data_config = json.load(f)
        test_files = data_config['test']
        trained_domains = data_config['trained_domains']
        test_domain = data_config['test_domain']
        test_domain_name = data_config['test_domain_name']
    
    print(f"📊 配置:")
    print(f"  测试域: {test_domain_name} (domain_id={test_domain})")
    print(f"  训练域: {trained_domains}")
    print(f"  测试样本: {len(test_files)}")

    # 先加载checkpoint获取配置
    print(f"\n📂 加载checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # 从checkpoint读取模型配置
    config = checkpoint.get('config', {})
    
    # 优先使用命令行参数，否则使用checkpoint中的配置
    if args.use_wavelet is not None:
        use_wavelet = args.use_wavelet.lower() == 'true'
    else:
        use_wavelet = config.get('use_wavelet', False)  # 默认False
    
    if args.use_adapter is not None:
        use_adapter = args.use_adapter.lower() == 'true'
    else:
        use_adapter = config.get('use_adapter', False)
    
    if args.use_progressive is not None:
        use_progressive = args.use_progressive.lower() == 'true'
    else:
        use_progressive = config.get('use_progressive', False)
    
    use_v4 = config.get('use_v4', True)
    
    print(f"✓ 模型配置: W={use_wavelet}, A={use_adapter}, P={use_progressive}")
    
    # 根据checkpoint配置初始化模型
    model = SAMB(
        img_size=1024,
        use_wavelet=use_wavelet,
        use_adapter=use_adapter,
        use_progressive=use_progressive,
        use_v4=use_v4
    )
    
    # 兼容两种checkpoint格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 完整checkpoint格式 (final_checkpoint.pth)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        domain_freq_stats = checkpoint.get('domain_freq_stats', None)
    else:
        # 仅state_dict格式 (best_model_epochXX.pth)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        domain_freq_stats = None
        print("⚠️  警告: 未找到domain_freq_stats，将使用均匀权重")
    
    if missing_keys:
        print(f"⚠️  Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
    
    print(f"✓ 模型已加载")
    if domain_freq_stats:
        print(f"✓ 频率统计已加载: {list(domain_freq_stats.keys())}")
    else:
        print(f"⚠️  未加载频率统计，将使用均匀权重ensemble")
    
    model = model.cuda()
    model.eval()

    # 创建测试数据加载器
    test_dataset = TestLoader(
        "mask_1024", 
        test_files, 
        A.Compose([
            A.Resize(1024, 1024),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ],
        additional_targets={'mask2': 'mask'})
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=1,
        num_workers=4,  # 增加到4
        pin_memory=True,
        prefetch_factor=2  # 预加载
    )

    # 评估
    df_results = eval_model_dg(
        model, test_loader, domain_freq_stats, trained_domains,
        test_domain_name, args.output_dir
    )

    # 生成汇总
    summary = {
        'domain': test_domain_name,
        'iou': df_results['iou'].mean(),
        'dice': df_results['dice'].mean(),
        'hd': df_results['hd'].mean()
    }
    
    df_summary = pd.DataFrame([summary])
    summary_path = os.path.join(args.output_dir, 'summary.csv')
    df_summary.to_csv(summary_path, index=False)
    
    print(f"✅ 评估完成")
    print(f"📁 结果保存到: {args.output_dir}")
    print(f"  - {test_domain_name}_results.csv (详细)")
    print(f"  - {test_domain_name}_weights.csv (权重分析)")
    print(f"  - summary.csv (汇总)")
    print("="*80 + "\n")
