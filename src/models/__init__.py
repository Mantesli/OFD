"""
Models module for oil field leak detection.
"""

from .mobileclip_extractor import (
    MobileCLIPExtractor,
    extract_clip_features,
    compute_semantic_weights,
    CLIPConfig,
    CLIPZeroShotValidator
)
from .fusion_classifier import (
    FusionClassifier,
    FusionClassifierConfig,
    MultiModalFusionModel
)
from .fusion_ad import (
    CrossModalAttention,
    BidirectionalCrossAttention,
    FeatureFusionModule,
    FusionADModel,
    FusionADConfig,
    MultiModalLeakDetector,
    FocalLoss,
    create_fusion_model
)
from .ema_smoother import (
    EMASmoother,
    EMAConfig,
    MultiRegionEMASmoother,
    VideoStreamProcessor,
    smooth_predictions,
    find_stable_detections
)

__all__ = [
    # MobileCLIP
    "MobileCLIPExtractor",
    "extract_clip_features", 
    "compute_semantic_weights",
    "CLIPConfig",
    "CLIPZeroShotValidator",
    # Fusion Classifier
    "FusionClassifier",
    "FusionClassifierConfig",
    "MultiModalFusionModel",
    # FusionAD
    "CrossModalAttention",
    "BidirectionalCrossAttention",
    "FeatureFusionModule",
    "FusionADModel",
    "FusionADConfig",
    "MultiModalLeakDetector",
    "FocalLoss",
    "create_fusion_model",
    # EMA Smoother
    "EMASmoother",
    "EMAConfig",
    "MultiRegionEMASmoother",
    "VideoStreamProcessor",
    "smooth_predictions",
    "find_stable_detections",
]
