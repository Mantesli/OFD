"""
Features module for oil field leak detection.
Handles thermal texture features, Z-score anomaly detection, temperature calibration,
region analysis, and leak discrimination.
"""

from .thermal_texture import (
    ThermalTextureExtractor,
    ThermalConfig,
    extract_thermal_features,
    compute_thermal_anomaly_score,
    ThermalFeatureValidator
)
from .zscore_thermal import (
    ZScoreAnomalyDetector,
    ZScoreConfig,
    AdaptiveZScoreDetector,
    ThermalAnomalyFusion,
    compute_zscore_anomaly,
    detect_hot_regions
)
from .thermal_calibration import (
    ThermalCalibrator,
    CalibrationConfig,
    CalibrationResult,
    DualModalSplitter,
    calibrate_temperature,
    split_dual_modal
)
from .region_analyzer import (
    RegionAnalyzer,
    TemperatureFeatures,
    MorphologyFeatures,
    RegionAnalysisResult,
    AnnotationLoader,
    extract_region_features
)
from .leak_discriminator import (
    LeakDiscriminator,
    DiscriminationThresholds,
    DiscriminationResult,
    AnomalyType,
    IntegratedLeakDetector,
    classify_thermal_anomaly
)

__all__ = [
    # Thermal Texture
    "ThermalTextureExtractor",
    "ThermalConfig",
    "extract_thermal_features",
    "compute_thermal_anomaly_score",
    "ThermalFeatureValidator",
    # Z-Score
    "ZScoreAnomalyDetector",
    "ZScoreConfig",
    "AdaptiveZScoreDetector",
    "ThermalAnomalyFusion",
    "compute_zscore_anomaly",
    "detect_hot_regions",
    # Thermal Calibration
    "ThermalCalibrator",
    "CalibrationConfig",
    "CalibrationResult",
    "DualModalSplitter",
    "calibrate_temperature",
    "split_dual_modal",
    # Region Analyzer
    "RegionAnalyzer",
    "TemperatureFeatures",
    "MorphologyFeatures",
    "RegionAnalysisResult",
    "AnnotationLoader",
    "extract_region_features",
    # Leak Discriminator
    "LeakDiscriminator",
    "DiscriminationThresholds",
    "DiscriminationResult",
    "AnomalyType",
    "IntegratedLeakDetector",
    "classify_thermal_anomaly",
]
