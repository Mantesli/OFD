#!/usr/bin/env python
"""
Step 4: Train Fusion Classifier (Sampler Version)
=================================================
最终修复版：
1. 启用 WeightedRandomSampler：无需物理复制即可解决样本不平衡。
2. 屏蔽热特征（Ablation）：暂时将热特征置零，强制模型使用我们确信有效的 RGB 特征。
3. 详细日志：打印预测概率均值，监控模型是否"死"了。

Usage:
    python scripts/04_train_classifier.py --feature_dir ./data/features --epochs 100 --lr 0.001
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import h5py
from loguru import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.fusion_classifier import FusionClassifier, FusionClassifierConfig
from src.utils.metrics import compute_metrics, RecallPriorityMetrics, MetricsResult


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)  # 小批量更适合极小样本
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-3)  # 增加正则化
    parser.add_argument("--dropout", type=float, default=0.5)  # 增加 Dropout 防止过拟合
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_features(feature_dir: Path, split: str):
    feature_path = feature_dir / f"{split}_features.h5"
    with h5py.File(feature_path, 'r') as f:
        data = {
            'rgb_clip': f['rgb_clip'][:],
            'ir_clip': f['ir_clip'][:],
            'thermal_texture': f['thermal_texture'][:],
            'semantic_weights': f['semantic_weights'][:],
            'labels': f['labels'][:]
        }
    return data


def create_dataloader(data, batch_size, is_train=True):
    rgb = torch.tensor(data['rgb_clip'], dtype=torch.float32)
    ir = torch.tensor(data['ir_clip'], dtype=torch.float32)
    thermal = torch.tensor(data['thermal_texture'], dtype=torch.float32)
    sem_w = torch.tensor(data['semantic_weights'], dtype=torch.float32)
    labels = torch.tensor(data['labels'], dtype=torch.long)

    # 简单的 zscore 替代
    zscore = thermal[:, 0]

    dataset = TensorDataset(rgb, ir, thermal, zscore, sem_w, labels)

    if is_train:
        # === 核心修复：使用采样器平衡数据 ===
        class_counts = torch.bincount(labels)
        class_weights = 1.0 / class_counts.float()
        # 为每个样本分配权重
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),  # 或者设大一点，比如 len*2
            replacement=True
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    probs_sum = 0
    count = 0

    for batch in loader:
        rgb, ir, thermal, zscore, sem_w, labels = [x.to(device) for x in batch]

        # === 激进策略：屏蔽热特征 ===
        # 既然 Zero-Shot (只用RGB) 有效，我们先强制模型只看 RGB
        # 等模型跑通了，再把这行注释掉
        # thermal = torch.zeros_like(thermal)
        # zscore = torch.zeros_like(zscore)

        optimizer.zero_grad()
        prob = model(rgb, ir, thermal, zscore, sem_w)
        loss = criterion(prob, labels.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        probs_sum += prob.mean().item()
        count += 1

    return total_loss / len(loader), probs_sum / count


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            rgb, ir, thermal, zscore, sem_w, labels = [x.to(device) for x in batch]
            # thermal = torch.zeros_like(thermal) # 保持一致
            # zscore = torch.zeros_like(zscore)

            prob = model(rgb, ir, thermal, zscore, sem_w)
            all_probs.extend(prob.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > 0.5).astype(int)

    return compute_metrics(all_labels, preds, all_probs)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = args.device

    # 1. 加载数据
    feat_dir = Path(args.feature_dir)
    train_data = load_features(feat_dir, "train")
    val_data = load_features(feat_dir, "val")

    train_loader = create_dataloader(train_data, args.batch_size, is_train=True)
    val_loader = create_dataloader(val_data, args.batch_size, is_train=False)

    logger.info(f"Train Raw: {(train_data['labels'] == 1).sum()} leak, {(train_data['labels'] == 0).sum()} normal")

    # 2. 模型
    config = FusionClassifierConfig(
        clip_feature_dim=train_data['rgb_clip'].shape[1],
        thermal_feature_dim=train_data['thermal_texture'].shape[1],
        hidden_dims=[128, 64],  # 简化模型，防止过拟合
        dropout=args.dropout,
        use_zscore_features=True
    )
    model = FusionClassifier(config).to(device)

    # 3. Loss (使用标准 BCE，因为 Sampler 已经处理了平衡)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    tracker = RecallPriorityMetrics(recall_target=0.85, patience=20)

    logger.info("Starting training (Sampler Enabled)...")

    for epoch in range(args.epochs):
        loss, mean_prob = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, device)

        # 包装 metrics
        res_obj = MetricsResult(
            accuracy=metrics.accuracy, recall=metrics.recall,
            precision=metrics.precision, f1=metrics.f1,
            specificity=metrics.specificity, confusion_matrix=np.array([[0, 0], [0, 0]]),
            roc_auc=metrics.roc_auc
        )

        is_best, _ = tracker.update(res_obj, epoch)

        if is_best:
            torch.save({'model_state_dict': model.state_dict(), 'config': config},
                       Path(args.output_dir) / "best_model.pth")

        logger.info(
            f"Ep {epoch + 1}: Loss={loss:.4f}, MeanProb={mean_prob:.3f}, ValRecall={metrics.recall:.4f}, ValAcc={metrics.accuracy:.4f}")


if __name__ == "__main__":
    main()