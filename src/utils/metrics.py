"""
Evaluation Metrics for Oil Leak Detection
=========================================
以召回率为优先的评估指标。

对于油田泄漏检测，漏检比误检更严重，因此：
- 主要指标：召回率（Recall）
- 次要指标：准确率（Accuracy）
- 参考指标：精确率（Precision）、F1
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score
)


@dataclass
class MetricsResult:
    """评估指标结果"""
    accuracy: float
    recall: float
    precision: float
    f1: float
    specificity: float
    confusion_matrix: np.ndarray
    
    # 可选的扩展指标
    roc_auc: Optional[float] = None
    average_precision: Optional[float] = None
    
    def __str__(self):
        return (
            f"Metrics:\n"
            f"  Recall (leak detection rate): {self.recall:.4f}\n"
            f"  Accuracy: {self.accuracy:.4f}\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  F1-Score: {self.f1:.4f}\n"
            f"  Specificity: {self.specificity:.4f}"
        )
    
    def to_dict(self) -> Dict:
        return {
            'accuracy': self.accuracy,
            'recall': self.recall,
            'precision': self.precision,
            'f1': self.f1,
            'specificity': self.specificity,
            'roc_auc': self.roc_auc,
            'average_precision': self.average_precision,
            'confusion_matrix': self.confusion_matrix.tolist()
        }
    
    def meets_targets(
        self, 
        recall_target: float = 0.85,
        accuracy_target: float = 0.80
    ) -> Tuple[bool, str]:
        """检查是否达到目标指标"""
        issues = []
        
        if self.recall < recall_target:
            issues.append(f"Recall {self.recall:.2%} < target {recall_target:.2%}")
        if self.accuracy < accuracy_target:
            issues.append(f"Accuracy {self.accuracy:.2%} < target {accuracy_target:.2%}")
        
        if issues:
            return False, "; ".join(issues)
        return True, "All targets met"


def compute_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    y_prob: Optional[Union[List, np.ndarray]] = None
) -> MetricsResult:
    """
    计算评估指标
    
    Args:
        y_true: 真实标签 (0=正常, 1=泄漏)
        y_pred: 预测标签
        y_prob: 预测概率（可选，用于计算AUC）
        
    Returns:
        MetricsResult对象
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 基本指标
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 特异度 (True Negative Rate)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = 0
    
    # 可选的概率相关指标
    roc_auc = None
    average_precision = None
    
    if y_prob is not None:
        y_prob = np.array(y_prob)
        if len(np.unique(y_true)) > 1:  # 确保有两个类别
            roc_auc = roc_auc_score(y_true, y_prob)
            average_precision = average_precision_score(y_true, y_prob)
    
    return MetricsResult(
        accuracy=accuracy,
        recall=recall,
        precision=precision,
        f1=f1,
        specificity=specificity,
        confusion_matrix=cm,
        roc_auc=roc_auc,
        average_precision=average_precision
    )


class RecallPriorityMetrics:
    """
    召回率优先的指标追踪器
    
    用于训练过程中的模型选择和早停。
    """
    
    def __init__(
        self,
        recall_target: float = 0.85,
        accuracy_target: float = 0.80,
        patience: int = 10
    ):
        self.recall_target = recall_target
        self.accuracy_target = accuracy_target
        self.patience = patience
        
        self.best_recall = 0
        self.best_accuracy_at_recall = 0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        self.history: List[MetricsResult] = []
    
    def update(self, metrics: MetricsResult, epoch: int) -> Tuple[bool, bool]:
        """
        更新指标
        
        Args:
            metrics: 当前epoch的指标
            epoch: 当前epoch
            
        Returns:
            (is_best, should_stop)
        """
        self.history.append(metrics)
        
        is_best = False
        
        # 优先考虑召回率，其次是准确率
        if metrics.recall > self.best_recall:
            self.best_recall = metrics.recall
            self.best_accuracy_at_recall = metrics.accuracy
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            is_best = True
        elif (metrics.recall >= self.best_recall * 0.99 and 
              metrics.accuracy > self.best_accuracy_at_recall):
            # 召回率相近但准确率更高
            self.best_accuracy_at_recall = metrics.accuracy
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            is_best = True
        else:
            self.epochs_without_improvement += 1
        
        should_stop = self.epochs_without_improvement >= self.patience
        
        return is_best, should_stop
    
    def get_summary(self) -> Dict:
        """获取训练摘要"""
        return {
            'best_epoch': self.best_epoch,
            'best_recall': self.best_recall,
            'best_accuracy_at_recall': self.best_accuracy_at_recall,
            'total_epochs': len(self.history),
            'targets_met': (
                self.best_recall >= self.recall_target and 
                self.best_accuracy_at_recall >= self.accuracy_target
            )
        }


def find_threshold_for_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.85
) -> Tuple[float, Dict]:
    """
    寻找满足目标召回率的最优阈值
    
    Args:
        y_true: 真实标签
        y_prob: 预测概率
        target_recall: 目标召回率
        
    Returns:
        (optimal_threshold, metrics_at_threshold)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # 找到满足召回率要求的阈值
    valid_indices = np.where(recalls[:-1] >= target_recall)[0]
    
    if len(valid_indices) == 0:
        # 如果没有阈值能达到目标召回率，返回最低阈值
        best_idx = len(thresholds) - 1
    else:
        # 在满足召回率的阈值中，选择精确率最高的
        best_idx = valid_indices[np.argmax(precisions[valid_indices])]
    
    optimal_threshold = thresholds[best_idx]
    
    # 计算该阈值下的指标
    y_pred = (y_prob >= optimal_threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    return optimal_threshold, metrics.to_dict()


def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    paths: Optional[List[str]] = None
) -> Dict:
    """
    分析错误样本
    
    Returns:
        错误分析结果
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 错误类型
    false_negatives = (y_true == 1) & (y_pred == 0)  # 漏检
    false_positives = (y_true == 0) & (y_pred == 1)  # 误检
    
    result = {
        'false_negative_count': int(false_negatives.sum()),
        'false_positive_count': int(false_positives.sum()),
        'false_negative_rate': float(false_negatives.sum() / max(1, (y_true == 1).sum())),
        'false_positive_rate': float(false_positives.sum() / max(1, (y_true == 0).sum())),
    }
    
    if y_prob is not None:
        y_prob = np.array(y_prob)
        # 分析错误样本的概率分布
        if false_negatives.sum() > 0:
            result['false_negative_prob_mean'] = float(y_prob[false_negatives].mean())
            result['false_negative_prob_std'] = float(y_prob[false_negatives].std())
        if false_positives.sum() > 0:
            result['false_positive_prob_mean'] = float(y_prob[false_positives].mean())
            result['false_positive_prob_std'] = float(y_prob[false_positives].std())
    
    if paths is not None:
        result['false_negative_samples'] = [paths[i] for i in np.where(false_negatives)[0]]
        result['false_positive_samples'] = [paths[i] for i in np.where(false_positives)[0]]
    
    return result
