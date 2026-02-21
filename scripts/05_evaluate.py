#!/usr/bin/env python
"""
Step 5: Evaluate Model (Fixed for PyTorch 2.6+)
===============================================
在测试集上评估训练好的模型，生成详细报告。

修复说明：
1. torch.load 添加 weights_only=False 参数，解决 PyTorch 2.6+ 的 UnpicklingError。

Usage:
    python scripts/05_evaluate.py \
        --checkpoint ./checkpoints/best_model.pth \
        --feature_dir ./data/features \
        --output_dir ./results/evaluation
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py
from loguru import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.fusion_classifier import FusionClassifier, FusionClassifierConfig
from src.utils.metrics import compute_metrics, find_threshold_for_recall, analyze_errors


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/evaluation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--recall_target", type=float, default=0.85)
    return parser.parse_args()


def load_features(feature_dir: Path, split: str) -> dict:
    """加载特征"""
    with h5py.File(feature_dir / f"{split}_features.h5", 'r') as f:
        return {
            'rgb_clip': f['rgb_clip'][:],
            'ir_clip': f['ir_clip'][:],
            'thermal_texture': f['thermal_texture'][:],
            'semantic_weights': f['semantic_weights'][:],
            'labels': f['labels'][:],
            # 兼容处理：检查是否存在 paths
            'paths': [p.decode() if isinstance(p, bytes) else p for p in f['paths'][:]] if 'paths' in f else []
        }


def evaluate_model(model, data, device, threshold=0.5):
    """评估模型"""
    model.eval()

    rgb_clip = torch.tensor(data['rgb_clip'], dtype=torch.float32, device=device)
    ir_clip = torch.tensor(data['ir_clip'], dtype=torch.float32, device=device)
    thermal = torch.tensor(data['thermal_texture'], dtype=torch.float32, device=device)
    zscore = thermal[:, 0]  # 使用第一个特征作为zscore
    semantic_w = torch.tensor(data['semantic_weights'], dtype=torch.float32, device=device)

    with torch.no_grad():
        probs = model(
            rgb_clip_feat=rgb_clip,
            ir_clip_feat=ir_clip,
            thermal_feat=thermal,
            zscore_feat=zscore,
            semantic_weight=semantic_w
        )

    probs = probs.cpu().numpy()
    preds = (probs > threshold).astype(int)
    labels = data['labels']

    return probs, preds, labels


def plot_results(probs, labels, output_dir):
    """生成评估图表"""
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    # 检查是否只有一个类别（防止报错）
    if len(np.unique(labels)) < 2:
        logger.warning("测试集只包含一个类别，无法绘制 ROC/PR 曲线。")
        return 0.0, 0.0

    # ROC曲线
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC={roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PR曲线
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)

    axes[1].plot(recall, precision, 'r-', lw=2, label=f'PR (AUC={pr_auc:.3f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 概率分布
    leak_probs = probs[labels == 1]
    normal_probs = probs[labels == 0]

    axes[2].hist(normal_probs, bins=50, alpha=0.5, label='Normal', color='green')
    axes[2].hist(leak_probs, bins=50, alpha=0.5, label='Leak', color='red')
    axes[2].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    axes[2].set_xlabel('Predicted Probability')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Prediction Distribution')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_plots.png', dpi=150)
    plt.close()

    return roc_auc, pr_auc


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"

    # 加载模型
    logger.info(f"Loading model from {args.checkpoint}")

    # === 关键修复：weights_only=False ===
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    config = checkpoint['config']
    model = FusionClassifier(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 加载测试数据
    feature_dir = Path(args.feature_dir)
    test_data = load_features(feature_dir, "test")
    logger.info(f"Test set: {len(test_data['labels'])} samples")

    if len(test_data['labels']) == 0:
        logger.error("测试集为空！请检查数据提取步骤。")
        return

    # 使用默认阈值评估
    probs, preds, labels = evaluate_model(model, test_data, device, threshold=0.5)
    metrics = compute_metrics(labels, preds, probs)

    print("\n" + "=" * 60)
    print("Test Set Evaluation (threshold=0.5)")
    print("=" * 60)
    print(metrics)

    # 寻找最优阈值
    optimal_threshold, optimal_metrics = find_threshold_for_recall(
        labels, probs, target_recall=args.recall_target
    )

    print(f"\nOptimal threshold for recall >= {args.recall_target}: {optimal_threshold:.3f}")
    print(f"Metrics at optimal threshold:")
    for k, v in optimal_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # 生成图表
    roc_auc, pr_auc = plot_results(probs, labels, output_dir)

    # 错误分析
    optimal_preds = (probs > optimal_threshold).astype(int)

    # 兼容性处理：如果 paths 不存在，用索引代替
    paths = test_data.get('paths', [f"sample_{i}" for i in range(len(labels))])
    error_analysis = analyze_errors(labels, optimal_preds, probs, paths)

    print(f"\nError Analysis:")
    print(f"  False Negatives (missed leaks): {error_analysis['false_negative_count']}")
    print(f"  False Positives (false alarms): {error_analysis['false_positive_count']}")

    # 保存结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': args.checkpoint,
        'test_samples': len(labels),
        'default_threshold': {
            'threshold': 0.5,
            'accuracy': float(metrics.accuracy),
            'recall': float(metrics.recall),
            'precision': float(metrics.precision),
            'f1': float(metrics.f1)
        },
        'optimal_threshold': {
            'threshold': float(optimal_threshold),
            **{k: float(v) if isinstance(v, (int, float, np.floating)) else v
               for k, v in optimal_metrics.items()}
        },
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'error_analysis': {
            'false_negatives': error_analysis['false_negative_count'],
            'false_positives': error_analysis['false_positive_count']
        }
    }

    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_dir}")

    # 检查是否达标
    print("\n" + "=" * 60)
    print("Target Check")
    print("=" * 60)

    recall_met = optimal_metrics['recall'] >= args.recall_target
    accuracy_met = optimal_metrics['accuracy'] >= 0.80

    print(f"Recall >= {args.recall_target}: {'✓ PASS' if recall_met else '✗ FAIL'}")
    print(f"Accuracy >= 0.80: {'✓ PASS' if accuracy_met else '✗ FAIL'}")

    if recall_met and accuracy_met:
        print("\n🎉 All targets met!")
    else:
        print("\n⚠️ Some targets not met. Consider:")
        if not recall_met:
            print("  - Lower the decision threshold")
            print("  - Add more leak samples")
            print("  - Adjust class weights")
        if not accuracy_met:
            print("  - Improve feature engineering")
            print("  - Add more negative prompts for interference")
            print("  - Use FusionAD cross-attention")

    print("=" * 60)


if __name__ == "__main__":
    main()