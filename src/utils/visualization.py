"""
Visualization utilities for oil leak detection.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def visualize_detection(
    rgb: np.ndarray,
    ir: np.ndarray,
    prediction: float,
    label: Optional[int] = None,
    semantic_weight: Optional[float] = None,
    thermal_score: Optional[float] = None,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    可视化检测结果
    
    Args:
        rgb: RGB图像
        ir: 红外图像
        prediction: 预测概率
        label: 真实标签（可选）
        semantic_weight: 语义权重（可选）
        thermal_score: 温度异常分数（可选）
        save_path: 保存路径（可选）
        
    Returns:
        可视化图像
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB图像
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    # 红外图像
    if len(ir.shape) == 2:
        axes[1].imshow(ir, cmap='hot')
    else:
        axes[1].imshow(ir)
    axes[1].set_title('Infrared Image')
    axes[1].axis('off')
    
    # 检测结果
    color = 'red' if prediction > 0.5 else 'green'
    result_text = f"Leak Probability: {prediction:.2%}"
    
    if label is not None:
        gt = "LEAK" if label == 1 else "NORMAL"
        pred = "LEAK" if prediction > 0.5 else "NORMAL"
        correct = "✓" if (label == 1) == (prediction > 0.5) else "✗"
        result_text += f"\nGround Truth: {gt}\nPrediction: {pred} {correct}"
    
    if semantic_weight is not None:
        result_text += f"\nSemantic Weight: {semantic_weight:.3f}"
    
    if thermal_score is not None:
        result_text += f"\nThermal Score: {thermal_score:.3f}"
    
    axes[2].text(0.5, 0.5, result_text, 
                transform=axes[2].transAxes,
                fontsize=14, verticalalignment='center',
                horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    axes[2].set_title('Detection Result')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return img


def plot_feature_distribution(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_k: int = 10,
    save_path: Optional[str] = None
):
    """
    绘制特征分布对比图
    
    Args:
        features: 特征矩阵 (N, D)
        labels: 标签 (N,)
        feature_names: 特征名称列表
        top_k: 显示前k个最具区分力的特征
        save_path: 保存路径
    """
    n_features = features.shape[1]
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    # 计算每个特征的区分度（效果量）
    leak_mask = labels == 1
    normal_mask = labels == 0
    
    effect_sizes = []
    for i in range(n_features):
        leak_values = features[leak_mask, i]
        normal_values = features[normal_mask, i]
        
        pooled_std = np.sqrt((np.var(leak_values) + np.var(normal_values)) / 2)
        if pooled_std > 1e-8:
            effect_size = abs(np.mean(leak_values) - np.mean(normal_values)) / pooled_std
        else:
            effect_size = 0
        effect_sizes.append(effect_size)
    
    # 选择top_k个特征
    top_indices = np.argsort(effect_sizes)[-top_k:][::-1]
    
    # 绘图
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for ax_idx, feat_idx in enumerate(top_indices):
        if ax_idx >= len(axes):
            break
        
        ax = axes[ax_idx]
        
        leak_values = features[leak_mask, feat_idx]
        normal_values = features[normal_mask, feat_idx]
        
        ax.hist(normal_values, bins=30, alpha=0.5, label='Normal', color='green')
        ax.hist(leak_values, bins=30, alpha=0.5, label='Leak', color='red')
        ax.set_title(f'{feature_names[feat_idx]}\n(effect size: {effect_sizes[feat_idx]:.2f})')
        ax.legend()
    
    plt.suptitle('Top Discriminative Features Distribution', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史字典，包含 'train_loss', 'val_loss', 'val_recall' 等
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # 损失曲线
    ax = axes[0]
    if 'train_loss' in history:
        ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 指标曲线
    ax = axes[1]
    if 'val_recall' in history:
        ax.plot(epochs, history['val_recall'], 'b-', label='Recall', linewidth=2)
    if 'val_accuracy' in history:
        ax.plot(epochs, history['val_accuracy'], 'g-', label='Accuracy')
    if 'val_precision' in history:
        ax.plot(epochs, history['val_precision'], 'r-', label='Precision')
    if 'val_f1' in history:
        ax.plot(epochs, history['val_f1'], 'm-', label='F1')
    
    # 目标线
    ax.axhline(y=0.85, color='b', linestyle='--', alpha=0.5, label='Recall Target (0.85)')
    ax.axhline(y=0.80, color='g', linestyle='--', alpha=0.5, label='Accuracy Target (0.80)')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Validation Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_detection_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    在图像上叠加检测掩码
    
    Args:
        image: 原始图像
        mask: 二值掩码
        alpha: 透明度
        color: 叠加颜色 (R, G, B)
        
    Returns:
        叠加后的图像
    """
    overlay = image.copy()
    
    # 创建彩色掩码
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    # 混合
    overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
    
    return overlay
