#!/usr/bin/env python
"""
Step 2: Validate CLIP Zero-Shot Classification
==============================================
验证MobileCLIP能否零样本区分油泄漏vs干扰物（黑土、设备等）。

这是Week 2实验1的核心脚本。

Usage:
    python scripts/02_validate_clip.py \
        --data_dir ./data/sampled/val \
        --output_dir ./results/clip_validation \
        --device cuda

Output:
    - 分类准确率、召回率、精确率
    - 最优阈值（满足目标召回率）
    - 混淆矩阵可视化
    - 错误样本分析
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.mobileclip_extractor import (
    MobileCLIPExtractor, 
    CLIPConfig,
    CLIPZeroShotValidator,
    DEFAULT_POSITIVE_PROMPTS,
    DEFAULT_NEGATIVE_PROMPTS
)
from src.data.frame_splitter import split_rgb_ir


def parse_args():
    parser = argparse.ArgumentParser(description="Validate CLIP zero-shot classification")
    
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing images (with leak/normal subdirs)")
    parser.add_argument("--output_dir", type=str, default="./results/clip_validation",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu", "mps"])
    parser.add_argument("--model", type=str, default="ViT-B-32",
                       help="CLIP model name")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--recall_target", type=float, default=0.85,
                       help="Target recall for threshold optimization")
    parser.add_argument("--use_rgb_only", action="store_true",
                       help="Only use RGB images (split from combined)")
    parser.add_argument("--split_mode", type=str, default="horizontal")
    parser.add_argument("--rgb_position", type=str, default="right")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples per class (for quick testing)")
    
    return parser.parse_args()


def load_images(data_dir: Path, args) -> tuple:
    """加载图像和标签"""
    images = []
    labels = []
    paths = []
    
    for label_name, label_value in [('leak', 1), ('normal', 0)]:
        label_dir = data_dir / label_name
        if not label_dir.exists():
            logger.warning(f"Directory not found: {label_dir}")
            continue
        
        image_files = list(label_dir.glob('*.jpg')) + list(label_dir.glob('*.png'))
        
        if args.max_samples:
            image_files = image_files[:args.max_samples]
        
        for img_path in tqdm(image_files, desc=f"Loading {label_name}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if args.use_rgb_only:
                rgb, _ = split_rgb_ir(img, args.split_mode, args.rgb_position)
                img = rgb
            
            images.append(img)
            labels.append(label_value)
            paths.append(str(img_path))
    
    logger.info(f"Loaded {len(images)} images: "
               f"{sum(labels)} leak, {len(labels) - sum(labels)} normal")
    
    return images, labels, paths


def plot_confusion_matrix(y_true, y_pred, output_path):
    """绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['Normal', 'Leak']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return cm


def plot_roc_curve(y_true, y_scores, output_path):
    """绘制ROC曲线"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return roc_auc


def plot_threshold_analysis(y_true, y_scores, output_path, recall_target=0.85):
    """绘制阈值分析图"""
    thresholds = np.arange(0.1, 0.9, 0.02)
    recalls = []
    precisions = []
    accuracies = []
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    for thresh in thresholds:
        preds = (y_scores > thresh).astype(int)
        
        leak_mask = y_true == 1
        recall = preds[leak_mask].mean() if leak_mask.sum() > 0 else 0
        recalls.append(recall)
        
        precision = y_true[preds == 1].mean() if (preds == 1).sum() > 0 else 0
        precisions.append(precision)
        
        accuracy = (preds == y_true).mean()
        accuracies.append(accuracy)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, recalls, 'b-', label='Recall', linewidth=2)
    ax.plot(thresholds, precisions, 'r-', label='Precision', linewidth=2)
    ax.plot(thresholds, accuracies, 'g-', label='Accuracy', linewidth=2)
    
    ax.axhline(y=recall_target, color='b', linestyle='--', alpha=0.5, 
               label=f'Target Recall ({recall_target})')
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Threshold Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading images from {args.data_dir}")
    images, labels, paths = load_images(Path(args.data_dir), args)
    
    if len(images) == 0:
        logger.error("No images loaded!")
        sys.exit(1)
    
    logger.info(f"Initializing CLIP model: {args.model}")
    config = CLIPConfig(model_name=args.model, device=args.device)
    extractor = MobileCLIPExtractor(config)
    
    logger.info("Running zero-shot classification...")
    all_probs = []
    
    for i in tqdm(range(0, len(images), args.batch_size)):
        batch = images[i:i + args.batch_size]
        probs = extractor.zero_shot_classify(batch)
        all_probs.extend(probs.cpu().numpy().tolist())
    
    all_probs = np.array(all_probs)
    
    default_threshold = 0.5
    predictions = (all_probs > default_threshold).astype(int)
    
    labels_np = np.array(labels)
    accuracy = (predictions == labels_np).mean()
    
    leak_mask = labels_np == 1
    normal_mask = labels_np == 0
    
    recall = predictions[leak_mask].mean() if leak_mask.sum() > 0 else 0
    precision = labels_np[predictions == 1].mean() if (predictions == 1).sum() > 0 else 0
    specificity = (1 - predictions[normal_mask]).mean() if normal_mask.sum() > 0 else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    validator = CLIPZeroShotValidator(config)
    optimal_threshold, optimal_metrics = validator.find_optimal_threshold(
        images, labels, recall_target=args.recall_target
    )
    
    optimal_predictions = (all_probs > optimal_threshold).astype(int)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model': args.model,
            'data_dir': args.data_dir,
            'num_images': len(images),
            'num_leak': int(leak_mask.sum()),
            'num_normal': int(normal_mask.sum()),
        },
        'default_threshold_results': {
            'threshold': default_threshold,
            'accuracy': float(accuracy),
            'recall': float(recall),
            'precision': float(precision),
            'f1': float(f1)
        },
        'optimal_threshold_results': {
            'threshold': float(optimal_threshold),
            **optimal_metrics
        }
    }
    
    print("\n" + "="*60)
    print("CLIP Zero-Shot Classification Results")
    print("="*60)
    print(f"\nDataset: {len(images)} images ({leak_mask.sum()} leak, {normal_mask.sum()} normal)")
    print(f"\nDefault Threshold ({default_threshold}):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print("="*60)
    
    logger.info("Generating visualizations...")
    cm = plot_confusion_matrix(labels, optimal_predictions, output_dir / 'confusion_matrix.png')
    roc_auc = plot_roc_curve(labels, all_probs, output_dir / 'roc_curve.png')
    plot_threshold_analysis(labels, all_probs, output_dir / 'threshold_analysis.png', args.recall_target)
    
    results['roc_auc'] = float(roc_auc)
    
    results_path = output_dir / 'results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
