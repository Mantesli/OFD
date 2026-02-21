"""
Fusion Classifier
=================
多模态融合分类器，整合所有组件进行最终泄漏检测。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger


@dataclass
class FusionClassifierConfig:
    """融合分类器配置"""
    clip_model: str = "ViT-B-32"
    clip_feature_dim: int = 512
    thermal_feature_dim: int = 30
    fusion_method: str = "concat"
    hidden_dims: List[int] = None
    dropout: float = 0.3
    use_clip_features: bool = True
    use_thermal_features: bool = True
    use_zscore_features: bool = True
    use_semantic_weight: bool = True
    num_cross_attn_layers: int = 2
    num_heads: int = 8
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class FusionClassifier(nn.Module):
    """多模态融合分类器"""
    
    def __init__(self, config: Optional[FusionClassifierConfig] = None):
        super().__init__()
        self.config = config or FusionClassifierConfig()
        
        # 计算输入特征维度
        input_dim = 0
        if self.config.use_clip_features:
            input_dim += self.config.clip_feature_dim * 2
        if self.config.use_thermal_features:
            input_dim += self.config.thermal_feature_dim
        if self.config.use_zscore_features:
            input_dim += 1
        
        self.input_dim = input_dim
        
        # 构建分类器
        layers = []
        in_dim = input_dim
        
        for hidden_dim in self.config.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 2))
        self.classifier = nn.Sequential(*layers)
        
        if self.config.use_thermal_features:
            self.thermal_norm = nn.LayerNorm(self.config.thermal_feature_dim)
        
        logger.info(f"FusionClassifier initialized with input_dim={input_dim}")
    
    def forward(
        self,
        rgb_clip_feat: Optional[torch.Tensor] = None,
        ir_clip_feat: Optional[torch.Tensor] = None,
        thermal_feat: Optional[torch.Tensor] = None,
        zscore_feat: Optional[torch.Tensor] = None,
        semantic_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        features = []
        
        if self.config.use_clip_features:
            if rgb_clip_feat is not None:
                features.append(rgb_clip_feat)
            if ir_clip_feat is not None:
                features.append(ir_clip_feat)
        
        if self.config.use_thermal_features and thermal_feat is not None:
            thermal_feat = self.thermal_norm(thermal_feat)
            features.append(thermal_feat)
        
        if self.config.use_zscore_features and zscore_feat is not None:
            if zscore_feat.dim() == 1:
                zscore_feat = zscore_feat.unsqueeze(-1)
            features.append(zscore_feat)
        
        if len(features) == 0:
            raise ValueError("No features provided!")
        
        combined_feat = torch.cat(features, dim=-1)
        logits = self.classifier(combined_feat)
        probs = F.softmax(logits, dim=-1)
        leak_prob = probs[:, 1]
        
        if self.config.use_semantic_weight and semantic_weight is not None:
            leak_prob = leak_prob * semantic_weight
        
        return leak_prob


class MultiModalFusionModel(nn.Module):
    """完整的多模态融合模型"""
    
    def __init__(
        self,
        config: Optional[FusionClassifierConfig] = None,
        device: str = "cuda"
    ):
        super().__init__()
        self.config = config or FusionClassifierConfig()
        self.device = device
        self._clip_extractor = None
        self._thermal_extractor = None
        self._zscore_detector = None
        self.classifier = FusionClassifier(self.config)
    
    @property
    def clip_extractor(self):
        if self._clip_extractor is None:
            from .mobileclip_extractor import MobileCLIPExtractor, CLIPConfig
            clip_config = CLIPConfig(
                model_name=self.config.clip_model,
                device=self.device
            )
            self._clip_extractor = MobileCLIPExtractor(clip_config)
        return self._clip_extractor
    
    @property
    def thermal_extractor(self):
        if self._thermal_extractor is None:
            from ..features.thermal_texture import ThermalTextureExtractor
            self._thermal_extractor = ThermalTextureExtractor()
        return self._thermal_extractor
    
    @property
    def zscore_detector(self):
        if self._zscore_detector is None:
            from ..features.zscore_thermal import ZScoreAnomalyDetector
            self._zscore_detector = ZScoreAnomalyDetector()
        return self._zscore_detector
    
    def forward(
        self,
        rgb: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
        ir: Union[torch.Tensor, np.ndarray, List[np.ndarray]]
    ) -> torch.Tensor:
        # 处理输入
        if isinstance(rgb, np.ndarray) and rgb.ndim == 3:
            rgb_list = [rgb]
            ir_list = [ir]
        elif isinstance(rgb, list):
            rgb_list = rgb
            ir_list = ir
        else:
            rgb_list = [rgb[i].cpu().numpy().transpose(1, 2, 0) 
                       for i in range(rgb.shape[0])]
            ir_list = [ir[i].cpu().numpy().transpose(1, 2, 0) 
                      for i in range(ir.shape[0])]
        
        # 提取特征
        with torch.no_grad():
            rgb_clip = self.clip_extractor.extract_image_features(rgb_list)
            ir_clip = self.clip_extractor.extract_image_features(ir_list)
            semantic_w = self.clip_extractor.compute_semantic_weights(rgb_list)
        
        thermal_feats = []
        zscore_scores = []
        for ir_img in ir_list:
            feat_dict = self.thermal_extractor.extract_features(ir_img)
            feat_vec = self.thermal_extractor.features_to_vector(feat_dict)
            thermal_feats.append(feat_vec)
            zscore_scores.append(self.zscore_detector.compute_anomaly_score(ir_img))
        
        thermal = torch.tensor(np.array(thermal_feats), dtype=torch.float32, device=self.device)
        zscore = torch.tensor(zscore_scores, dtype=torch.float32, device=self.device)
        
        return self.classifier(
            rgb_clip_feat=rgb_clip,
            ir_clip_feat=ir_clip,
            thermal_feat=thermal,
            zscore_feat=zscore,
            semantic_weight=semantic_w
        )


if __name__ == "__main__":
    print("Testing FusionClassifier...")
    
    config = FusionClassifierConfig(
        clip_feature_dim=512,
        thermal_feature_dim=30,
        use_zscore_features=True
    )
    
    model = FusionClassifier(config)
    
    batch_size = 4
    rgb_clip = torch.randn(batch_size, 512)
    ir_clip = torch.randn(batch_size, 512)
    thermal = torch.randn(batch_size, 30)
    zscore = torch.randn(batch_size)
    semantic_w = torch.rand(batch_size)
    
    prob = model(
        rgb_clip_feat=rgb_clip,
        ir_clip_feat=ir_clip,
        thermal_feat=thermal,
        zscore_feat=zscore,
        semantic_weight=semantic_w
    )
    
    print(f"Output probability: {prob}")
    print(f"Output shape: {prob.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
