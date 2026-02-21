"""
FusionAD: Multi-Modal Anomaly Detection with Cross-Modal Attention
===================================================================

本模块实现了基于FusionAD思想的多模态异常检测框架，专门针对RGB+红外的油田泄漏检测任务。

=== 学习指南 ===

1. FusionAD核心思想
-------------------
FusionAD是一种多模态异常检测框架，核心创新点：
- 跨模态特征对齐：RGB和IR特征在同一语义空间对齐
- 跨模态注意力：一个模态的特征可以"查询"另一个模态
- 双向一致性：正常样本双模态特征应该一致，异常样本不一致

2. 为什么适合油田泄漏检测？
-------------------------
- 泄漏：RGB深色 + IR偏热 → 双模态都异常，且空间位置一致
- 黑土：RGB深色 + IR偏热 → 看似相似，但温度分布模式不同
- 阴影：RGB深色 + IR正常 → 双模态不一致，可排除

3. 本实现的简化
---------------
原始FusionAD使用复杂的Transformer编码器和解码器。
考虑到你的时间限制，本实现做了以下简化：
- 使用预训练MobileCLIP作为特征提取器（而非从头训练）
- 使用轻量级跨模态注意力模块
- 保留核心的双模态融合逻辑

4. 代码结构
-----------
- CrossModalAttention: 跨模态注意力层
- FeatureFusionModule: 特征融合模块
- FusionADModel: 完整的融合模型
- MultiModalClassifier: 端到端分类器

=== 参考文献 ===
- FusionAD: https://github.com/xxx/FusionAD (原始仓库)
- Cross-Modal Attention: "Attention is All You Need" 变体
- 多模态学习综述: "Multimodal Machine Learning: A Survey"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger


# =============================================================================
# Part 1: 基础组件
# =============================================================================

class LayerNorm2d(nn.Module):
    """
    2D Layer Normalization
    
    对特征图的每个位置进行归一化，保持空间结构。
    
    学习点：
    - BatchNorm对batch维度归一化，LayerNorm对特征维度归一化
    - LayerNorm在小batch和推理时更稳定
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)  # (B, 1, H, W)
        s = (x - u).pow(2).mean(1, keepdim=True)  # (B, 1, H, W)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# =============================================================================
# Part 2: 跨模态注意力 (Cross-Modal Attention)
# =============================================================================

class CrossModalAttention(nn.Module):
    """
    跨模态注意力模块
    
    核心思想：
    让一个模态的特征（Query）去"关注"另一个模态的特征（Key, Value）
    
    例如：RGB特征作为Query，IR特征作为Key和Value
    - 这样RGB特征可以"询问"IR特征："哪些区域是热的？"
    - 结果是RGB特征被IR信息增强
    
    数学公式：
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
    
    其中：
    - Q: Query，来自模态A
    - K: Key，来自模态B
    - V: Value，来自模态B
    - d: 特征维度
    
    学习点：
    - 这是Transformer的核心组件
    - 多头注意力可以捕捉不同类型的跨模态关系
    """
    
    def __init__(
        self, 
        dim: int,           # 特征维度
        num_heads: int = 8, # 注意力头数
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 缩放因子 1/sqrt(d)
        
        # Query, Key, Value的线性变换
        # 注意：Q来自一个模态，K和V来自另一个模态
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        
        # Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self, 
        query: torch.Tensor,    # 查询模态的特征 (B, N, D)
        key_value: torch.Tensor # 被查询模态的特征 (B, M, D)
    ) -> torch.Tensor:
        """
        Args:
            query: 模态A的特征，将被增强
            key_value: 模态B的特征，提供信息
            
        Returns:
            增强后的模态A特征
        """
        B, N, D = query.shape
        M = key_value.shape[1]
        
        # 计算Q, K, V
        q = self.q_proj(query)      # (B, N, D)
        k = self.k_proj(key_value)  # (B, M, D)
        v = self.v_proj(key_value)  # (B, M, D)
        
        # 重塑为多头形式
        # (B, N, D) -> (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力分数
        # (B, heads, N, head_dim) @ (B, heads, head_dim, M) -> (B, heads, N, M)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力到Value
        # (B, heads, N, M) @ (B, heads, M, head_dim) -> (B, heads, N, head_dim)
        out = attn @ v
        
        # 重塑回原始形状
        # (B, heads, N, head_dim) -> (B, N, heads, head_dim) -> (B, N, D)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        
        # 输出投影
        out = self.out_proj(out)
        out = self.proj_drop(out)
        
        return out


class BidirectionalCrossAttention(nn.Module):
    """
    双向跨模态注意力
    
    同时计算：
    1. RGB → IR：RGB查询IR，获取热信息增强
    2. IR → RGB：IR查询RGB，获取视觉信息增强
    
    这是FusionAD的核心创新之一：双向信息流动
    
    学习点：
    - 单向注意力只能增强一个模态
    - 双向注意力让两个模态互相增强
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # RGB → IR 的注意力
        self.rgb_to_ir_attn = CrossModalAttention(dim, num_heads, attn_drop=dropout)
        
        # IR → RGB 的注意力
        self.ir_to_rgb_attn = CrossModalAttention(dim, num_heads, attn_drop=dropout)
        
        # 残差连接后的LayerNorm
        self.norm_rgb = nn.LayerNorm(dim)
        self.norm_ir = nn.LayerNorm(dim)
        
        # 前馈网络 (FFN)
        self.ffn_rgb = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.ffn_ir = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        self.norm_rgb2 = nn.LayerNorm(dim)
        self.norm_ir2 = nn.LayerNorm(dim)
    
    def forward(
        self, 
        rgb_feat: torch.Tensor,  # (B, N, D)
        ir_feat: torch.Tensor    # (B, M, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rgb_feat: RGB特征
            ir_feat: IR特征
            
        Returns:
            (增强的RGB特征, 增强的IR特征)
        """
        # RGB查询IR（RGB被IR信息增强）
        rgb_enhanced = rgb_feat + self.rgb_to_ir_attn(rgb_feat, ir_feat)
        rgb_enhanced = self.norm_rgb(rgb_enhanced)
        
        # IR查询RGB（IR被RGB信息增强）
        ir_enhanced = ir_feat + self.ir_to_rgb_attn(ir_feat, rgb_feat)
        ir_enhanced = self.norm_ir(ir_enhanced)
        
        # FFN
        rgb_enhanced = rgb_enhanced + self.ffn_rgb(rgb_enhanced)
        rgb_enhanced = self.norm_rgb2(rgb_enhanced)
        
        ir_enhanced = ir_enhanced + self.ffn_ir(ir_enhanced)
        ir_enhanced = self.norm_ir2(ir_enhanced)
        
        return rgb_enhanced, ir_enhanced


# =============================================================================
# Part 3: 特征融合模块
# =============================================================================

class FeatureFusionModule(nn.Module):
    """
    特征融合模块
    
    将跨模态注意力增强后的特征融合为统一表示。
    
    融合策略：
    1. 简单拼接 (concat)
    2. 加权求和 (weighted_sum)
    3. 门控融合 (gated) - 学习自适应权重
    
    学习点：
    - 不同融合策略各有优劣
    - 门控融合最灵活，但参数更多
    - 简单拼接最稳定，适合数据量有限时
    """
    
    def __init__(
        self, 
        dim: int,
        fusion_method: str = "gated",  # "concat", "weighted_sum", "gated"
        dropout: float = 0.1
    ):
        super().__init__()
        self.fusion_method = fusion_method
        
        if fusion_method == "concat":
            # 拼接后降维
            self.fusion_proj = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        elif fusion_method == "weighted_sum":
            # 可学习的权重
            self.rgb_weight = nn.Parameter(torch.tensor(0.5))
            self.ir_weight = nn.Parameter(torch.tensor(0.5))
        
        elif fusion_method == "gated":
            # 门控机制：根据两个模态的内容动态决定权重
            self.gate = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Linear(dim, 2),  # 输出两个权重
                nn.Softmax(dim=-1)
            )
        
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(
        self, 
        rgb_feat: torch.Tensor, 
        ir_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            rgb_feat: RGB特征 (B, N, D) 或 (B, D)
            ir_feat: IR特征 (B, N, D) 或 (B, D)
            
        Returns:
            融合特征 (B, N, D) 或 (B, D)
        """
        if self.fusion_method == "concat":
            # 拼接 + 投影
            concat_feat = torch.cat([rgb_feat, ir_feat], dim=-1)
            return self.fusion_proj(concat_feat)
        
        elif self.fusion_method == "weighted_sum":
            # 归一化权重
            total = torch.abs(self.rgb_weight) + torch.abs(self.ir_weight)
            w_rgb = torch.abs(self.rgb_weight) / total
            w_ir = torch.abs(self.ir_weight) / total
            return w_rgb * rgb_feat + w_ir * ir_feat
        
        elif self.fusion_method == "gated":
            # 门控融合
            concat_feat = torch.cat([rgb_feat, ir_feat], dim=-1)
            weights = self.gate(concat_feat)  # (B, ..., 2)
            
            # 分离权重
            w_rgb = weights[..., 0:1]  # (B, ..., 1)
            w_ir = weights[..., 1:2]   # (B, ..., 1)
            
            return w_rgb * rgb_feat + w_ir * ir_feat


# =============================================================================
# Part 4: 完整的FusionAD模型
# =============================================================================

@dataclass
class FusionADConfig:
    """FusionAD配置"""
    # 特征维度
    feature_dim: int = 512
    
    # 跨模态注意力层数
    num_cross_attn_layers: int = 2
    
    # 注意力头数
    num_heads: int = 8
    
    # 融合方法
    fusion_method: str = "gated"  # "concat", "weighted_sum", "gated"
    
    # Dropout
    dropout: float = 0.1
    
    # 分类器隐藏层
    classifier_hidden_dims: List[int] = None
    
    # 类别数
    num_classes: int = 2
    
    def __post_init__(self):
        if self.classifier_hidden_dims is None:
            self.classifier_hidden_dims = [256, 128]


class FusionADModel(nn.Module):
    """
    FusionAD多模态融合模型
    
    架构：
    1. 特征提取器（使用预训练CLIP）
    2. 多层双向跨模态注意力
    3. 特征融合
    4. 分类头
    
    学习点：
    - 这是端到端的融合模型
    - 可以冻结特征提取器，只训练融合部分
    - 也可以微调整个模型
    """
    
    def __init__(
        self, 
        config: Optional[FusionADConfig] = None,
        feature_extractor = None  # 预训练的特征提取器
    ):
        super().__init__()
        self.config = config or FusionADConfig()
        
        # 特征提取器（可选，如果None则假设输入已是特征）
        self.feature_extractor = feature_extractor
        if feature_extractor is not None:
            # 冻结特征提取器
            for param in feature_extractor.parameters():
                param.requires_grad = False
        
        # 特征投影（确保维度匹配）
        self.rgb_proj = nn.Linear(self.config.feature_dim, self.config.feature_dim)
        self.ir_proj = nn.Linear(self.config.feature_dim, self.config.feature_dim)
        
        # 双向跨模态注意力层
        self.cross_attn_layers = nn.ModuleList([
            BidirectionalCrossAttention(
                dim=self.config.feature_dim,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout
            )
            for _ in range(self.config.num_cross_attn_layers)
        ])
        
        # 特征融合
        self.fusion = FeatureFusionModule(
            dim=self.config.feature_dim,
            fusion_method=self.config.fusion_method,
            dropout=self.config.dropout
        )
        
        # 分类头
        classifier_layers = []
        in_dim = self.config.feature_dim
        for hidden_dim in self.config.classifier_hidden_dims:
            classifier_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout)
            ])
            in_dim = hidden_dim
        classifier_layers.append(nn.Linear(in_dim, self.config.num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
    
    def extract_features(
        self, 
        rgb: torch.Tensor, 
        ir: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取特征"""
        if self.feature_extractor is not None:
            with torch.no_grad():
                rgb_feat = self.feature_extractor.extract_image_features(rgb)
                ir_feat = self.feature_extractor.extract_image_features(ir)
        else:
            # 假设输入已是特征
            rgb_feat = rgb
            ir_feat = ir
        
        return rgb_feat, ir_feat
    
    def forward(
        self, 
        rgb: torch.Tensor,
        ir: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        前向传播
        
        Args:
            rgb: RGB图像或特征 (B, C, H, W) 或 (B, D)
            ir: IR图像或特征 (B, C, H, W) 或 (B, D)
            return_features: 是否返回中间特征
            
        Returns:
            分类logits (B, num_classes)
            如果return_features=True，同时返回中间特征
        """
        # 提取特征
        rgb_feat, ir_feat = self.extract_features(rgb, ir)
        
        # 特征投影
        rgb_feat = self.rgb_proj(rgb_feat)
        ir_feat = self.ir_proj(ir_feat)
        
        # 确保是3D张量 (B, N, D)
        if rgb_feat.dim() == 2:
            rgb_feat = rgb_feat.unsqueeze(1)  # (B, 1, D)
            ir_feat = ir_feat.unsqueeze(1)
        
        # 双向跨模态注意力
        for cross_attn in self.cross_attn_layers:
            rgb_feat, ir_feat = cross_attn(rgb_feat, ir_feat)
        
        # 取平均（如果是序列）
        rgb_feat = rgb_feat.mean(dim=1)  # (B, D)
        ir_feat = ir_feat.mean(dim=1)
        
        # 融合
        fused_feat = self.fusion(rgb_feat, ir_feat)
        
        # 分类
        logits = self.classifier(fused_feat)
        
        if return_features:
            return logits, {
                'rgb_features': rgb_feat,
                'ir_features': ir_feat,
                'fused_features': fused_feat
            }
        
        return logits
    
    def predict_proba(
        self, 
        rgb: torch.Tensor,
        ir: torch.Tensor
    ) -> torch.Tensor:
        """预测概率"""
        logits = self.forward(rgb, ir)
        return F.softmax(logits, dim=-1)
    
    def predict(
        self, 
        rgb: torch.Tensor,
        ir: torch.Tensor
    ) -> torch.Tensor:
        """预测类别"""
        logits = self.forward(rgb, ir)
        return logits.argmax(dim=-1)


# =============================================================================
# Part 5: 端到端分类器（整合所有组件）
# =============================================================================

class MultiModalLeakDetector(nn.Module):
    """
    多模态泄漏检测器
    
    整合：
    1. MobileCLIP特征提取
    2. FusionAD跨模态融合
    3. 语义权重W(x,y)
    4. 最终分类
    
    公式实现：
    P_leak = FusionAD(A_rgb, A_ir) × W
    
    其中：
    - A_rgb, A_ir: 来自FusionAD的融合特征
    - W: MobileCLIP的语义权重
    """
    
    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        fusion_config: Optional[FusionADConfig] = None,
        device: str = "cuda"
    ):
        super().__init__()
        
        from .mobileclip_extractor import MobileCLIPExtractor, CLIPConfig
        
        # CLIP特征提取器
        clip_config = CLIPConfig(model_name=clip_model_name, device=device)
        self.clip_extractor = MobileCLIPExtractor(clip_config)
        
        # FusionAD模型
        self.fusion_config = fusion_config or FusionADConfig()
        self.fusion_model = FusionADModel(self.fusion_config)
        
        # 语义权重是否参与最终决策
        self.use_semantic_weight = True
        
        self.device = device
    
    def forward(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor,
        return_details: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Args:
            rgb: RGB图像
            ir: IR图像
            return_details: 是否返回详细信息
            
        Returns:
            泄漏概率 (B,)
        """
        # 提取CLIP特征
        with torch.no_grad():
            rgb_feat = self.clip_extractor.extract_image_features(rgb)
            ir_feat = self.clip_extractor.extract_image_features(ir)
            semantic_weights = self.clip_extractor.compute_semantic_weights(rgb)
        
        # FusionAD融合
        logits, features = self.fusion_model(
            rgb_feat, ir_feat, 
            return_features=True
        )
        
        # 转换为概率
        probs = F.softmax(logits, dim=-1)
        leak_prob = probs[:, 1]  # 泄漏类的概率
        
        # 应用语义权重
        if self.use_semantic_weight:
            final_prob = leak_prob * semantic_weights
        else:
            final_prob = leak_prob
        
        if return_details:
            return final_prob, {
                'raw_prob': leak_prob,
                'semantic_weight': semantic_weights,
                'rgb_features': features['rgb_features'],
                'ir_features': features['ir_features'],
                'fused_features': features['fused_features']
            }
        
        return final_prob


# =============================================================================
# Part 6: 训练工具
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss
    
    专门用于处理类别不平衡问题。
    
    公式：
    FL(p) = -α(1-p)^γ log(p)
    
    其中：
    - α: 类别权重，用于处理不平衡
    - γ: 聚焦参数，γ越大越关注难分样本
    
    学习点：
    - γ=0时退化为标准交叉熵
    - γ=2是常用值，效果良好
    - α通常设为正负样本比例的倒数
    """
    
    def __init__(
        self, 
        alpha: float = 0.75,  # 正样本权重
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self, 
        inputs: torch.Tensor,  # (B, C) logits
        targets: torch.Tensor  # (B,) 标签
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 预测概率
        
        # Focal权重
        focal_weight = (1 - pt) ** self.gamma
        
        # 类别权重
        alpha_weight = torch.where(
            targets == 1, 
            self.alpha, 
            1 - self.alpha
        )
        
        loss = alpha_weight * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def create_fusion_model(
    feature_dim: int = 512,
    num_cross_attn_layers: int = 2,
    fusion_method: str = "gated",
    num_classes: int = 2
) -> FusionADModel:
    """
    便捷函数：创建FusionAD模型
    """
    config = FusionADConfig(
        feature_dim=feature_dim,
        num_cross_attn_layers=num_cross_attn_layers,
        fusion_method=fusion_method,
        num_classes=num_classes
    )
    return FusionADModel(config)


# =============================================================================
# Part 7: 测试和演示
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FusionAD Module Test")
    print("=" * 60)
    
    # 测试配置
    batch_size = 4
    feature_dim = 512
    
    # 创建模拟特征
    rgb_feat = torch.randn(batch_size, feature_dim)
    ir_feat = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, 2, (batch_size,))
    
    print(f"\nInput shapes:")
    print(f"  RGB features: {rgb_feat.shape}")
    print(f"  IR features: {ir_feat.shape}")
    print(f"  Labels: {labels.shape}")
    
    # 测试跨模态注意力
    print("\n1. Testing CrossModalAttention...")
    cross_attn = CrossModalAttention(dim=feature_dim, num_heads=8)
    rgb_enhanced = cross_attn(rgb_feat.unsqueeze(1), ir_feat.unsqueeze(1))
    print(f"   Output shape: {rgb_enhanced.shape}")
    
    # 测试双向注意力
    print("\n2. Testing BidirectionalCrossAttention...")
    bi_attn = BidirectionalCrossAttention(dim=feature_dim)
    rgb_bi, ir_bi = bi_attn(rgb_feat.unsqueeze(1), ir_feat.unsqueeze(1))
    print(f"   RGB enhanced: {rgb_bi.shape}")
    print(f"   IR enhanced: {ir_bi.shape}")
    
    # 测试融合模块
    print("\n3. Testing FeatureFusionModule...")
    for method in ["concat", "weighted_sum", "gated"]:
        fusion = FeatureFusionModule(dim=feature_dim, fusion_method=method)
        fused = fusion(rgb_feat, ir_feat)
        print(f"   {method}: {fused.shape}")
    
    # 测试完整模型
    print("\n4. Testing FusionADModel...")
    model = create_fusion_model(feature_dim=feature_dim)
    logits = model(rgb_feat, ir_feat)
    print(f"   Output logits: {logits.shape}")
    probs = F.softmax(logits, dim=-1)
    print(f"   Probabilities: {probs}")
    
    # 测试Focal Loss
    print("\n5. Testing FocalLoss...")
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    loss = criterion(logits, labels)
    print(f"   Loss: {loss.item():.4f}")
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n6. Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
