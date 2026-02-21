"""
Leak Discriminator Module
=========================
基于温度特征和形态特征的泄漏判别器。

判别规则（基于物理先验）：

泄漏特征：
├─ 温度：中心高温 + 边缘突然下降（ΔT > 阈值）
├─ 形态：扇形/舌状（方向一致性高）或 线性管状
└─ 梯度：边缘梯度大（突变）

黑色土壤特征：
├─ 温度：整体均匀（ΔT ≈ 0，方差小）
├─ 形态：随机块状（方向一致性低，圆度适中）
└─ 梯度：各处梯度相近

油井设备特征：
├─ 温度：整体高温
├─ 形态：规则形状（圆度高或矩形）
└─ 位置：固定位置

轮胎痕迹特征：
├─ 温度：线性高温
├─ 形态：细长线性（长宽比大）
└─ 位置：沿道路
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
from loguru import logger

from .region_analyzer import (
    RegionAnalyzer, 
    RegionAnalysisResult,
    TemperatureFeatures,
    MorphologyFeatures
)


class AnomalyType(Enum):
    """异常类型"""
    LEAK = "leak"                    # 泄漏
    LEAK_POOL = "leak_pool"          # 液池扩散型泄漏
    LEAK_PIPELINE = "leak_pipeline"  # 管道穿孔型泄漏
    SOIL = "soil"                    # 黑色土壤
    EQUIPMENT = "equipment"          # 油井设备
    TIRE_TRACK = "tire_track"        # 轮胎痕迹
    UNKNOWN = "unknown"              # 未知


@dataclass
class DiscriminationThresholds:
    """判别阈值配置"""
    # 温度特征阈值
    min_delta_t: float = 3.0          # 最小中心-边缘温差（°C），泄漏应 > 此值
    max_delta_t_soil: float = 2.0     # 土壤的最大温差
    min_edge_gradient: float = 1.0    # 最小边缘梯度（突变程度）
    max_center_variance: float = 5.0  # 中心区域最大方差（泄漏中心相对均匀）
    
    # 形态特征阈值
    min_direction_consistency: float = 0.4   # 扇形的最小方向一致性
    max_spread_angle: float = 150.0          # 扇形的最大扩散角度
    min_linearity_pipeline: float = 0.3      # 管道泄漏的最小线性度
    max_circularity_leak: float = 0.85       # 泄漏的最大圆度（不应太圆）
    min_circularity_equipment: float = 0.7   # 设备的最小圆度
    max_aspect_ratio_pool: float = 4.0       # 液池的最大长宽比
    min_aspect_ratio_track: float = 5.0      # 轮胎痕迹的最小长宽比
    
    # 面积阈值
    min_area: int = 100                       # 最小有效面积
    max_area_track: int = 2000               # 轮胎痕迹的最大面积
    
    # 综合权重
    temp_weight: float = 0.5                 # 温度特征权重
    morph_weight: float = 0.3                # 形态特征权重
    gradient_weight: float = 0.2             # 梯度特征权重


@dataclass
class DiscriminationResult:
    """判别结果"""
    region_id: int
    anomaly_type: AnomalyType
    confidence: float                    # 置信度 (0-1)
    leak_score: float                    # 泄漏得分 (0-1)
    
    # 各项得分
    temp_score: float                    # 温度特征得分
    morph_score: float                   # 形态特征得分
    gradient_score: float                # 梯度特征得分
    
    # 判别依据
    reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'region_id': self.region_id,
            'anomaly_type': self.anomaly_type.value,
            'confidence': self.confidence,
            'leak_score': self.leak_score,
            'temp_score': self.temp_score,
            'morph_score': self.morph_score,
            'gradient_score': self.gradient_score,
            'reasons': self.reasons
        }


class LeakDiscriminator:
    """
    泄漏判别器
    
    基于规则和特征的泄漏/干扰分类。
    
    使用方法：
    ```python
    discriminator = LeakDiscriminator()
    result = discriminator.discriminate(region_analysis)
    
    if result.anomaly_type == AnomalyType.LEAK:
        print(f"检测到泄漏，置信度: {result.confidence:.2f}")
    ```
    """
    
    def __init__(self, thresholds: Optional[DiscriminationThresholds] = None):
        self.thresholds = thresholds or DiscriminationThresholds()
    
    def compute_temperature_score(
        self, 
        features: TemperatureFeatures
    ) -> Tuple[float, List[str]]:
        """
        计算温度特征得分
        
        泄漏特征：
        - 中心-边缘温差大 (ΔT > 阈值)
        - 边缘梯度大（突变）
        - 中心温度相对均匀（方差可控）
        
        Returns:
            score: 0-1，越高越像泄漏
            reasons: 判别依据
        """
        score = 0.0
        reasons = []
        
        # 1. 中心-边缘温差
        delta_t = features.delta_t
        if delta_t > self.thresholds.min_delta_t:
            # 温差越大，得分越高
            delta_score = min(1.0, delta_t / (self.thresholds.min_delta_t * 3))
            score += delta_score * 0.4
            reasons.append(f"中心-边缘温差={delta_t:.1f}°C (>阈值{self.thresholds.min_delta_t}°C)")
        elif delta_t < self.thresholds.max_delta_t_soil:
            score -= 0.2
            reasons.append(f"温差过小={delta_t:.1f}°C (像土壤)")
        
        # 2. 边缘梯度（突变程度）
        edge_grad = features.edge_gradient_mean
        if edge_grad > self.thresholds.min_edge_gradient:
            grad_score = min(1.0, edge_grad / (self.thresholds.min_edge_gradient * 5))
            score += grad_score * 0.3
            reasons.append(f"边缘梯度={edge_grad:.2f} (突变明显)")
        else:
            reasons.append(f"边缘梯度={edge_grad:.2f} (渐变/均匀)")
        
        # 3. 中心区域方差
        center_var = features.center_variance
        if center_var < self.thresholds.max_center_variance:
            # 中心均匀是泄漏特征
            var_score = 1.0 - (center_var / self.thresholds.max_center_variance)
            score += var_score * 0.2
            reasons.append(f"中心方差={center_var:.2f} (较均匀)")
        else:
            score -= 0.1
            reasons.append(f"中心方差={center_var:.2f} (噪声大)")
        
        # 4. 温度范围合理性检查
        t_min, t_max = features.temp_range
        if t_max > 5:  # 有明显热源
            score += 0.1
            reasons.append(f"最高温度={t_max:.1f}°C (有热源)")
        
        return np.clip(score, 0, 1), reasons
    
    def compute_morphology_score(
        self,
        features: MorphologyFeatures
    ) -> Tuple[float, List[str], AnomalyType]:
        """
        计算形态特征得分
        
        Returns:
            score: 0-1，越高越像泄漏
            reasons: 判别依据
            sub_type: 细分类型（液池/管道）
        """
        score = 0.0
        reasons = []
        sub_type = AnomalyType.UNKNOWN
        
        # 1. 方向一致性（扇形检测）
        dir_cons = features.direction_consistency
        spread_angle = features.spread_angle
        
        is_fan_shaped = (
            dir_cons > self.thresholds.min_direction_consistency and
            spread_angle < self.thresholds.max_spread_angle
        )
        
        if is_fan_shaped:
            score += 0.4
            reasons.append(f"扇形特征: 方向一致性={dir_cons:.2f}, 扩散角={spread_angle:.0f}°")
            sub_type = AnomalyType.LEAK_POOL
        
        # 2. 线性度（管道检测）
        linearity = features.linearity
        if linearity > self.thresholds.min_linearity_pipeline:
            score += 0.3
            reasons.append(f"线性特征: 线性度={linearity:.2f} (可能是管道泄漏)")
            if sub_type == AnomalyType.UNKNOWN:
                sub_type = AnomalyType.LEAK_PIPELINE
        
        # 3. 圆度检查
        circularity = features.circularity
        if circularity > self.thresholds.min_circularity_equipment:
            # 太圆可能是设备
            score -= 0.2
            reasons.append(f"圆度={circularity:.2f} (可能是设备)")
            sub_type = AnomalyType.EQUIPMENT
        elif circularity < self.thresholds.max_circularity_leak:
            score += 0.1
            reasons.append(f"圆度={circularity:.2f} (不规则，符合泄漏)")
        
        # 4. 长宽比检查
        aspect = features.aspect_ratio
        if aspect > self.thresholds.min_aspect_ratio_track:
            # 太细长可能是轮胎痕迹
            if features.area < self.thresholds.max_area_track:
                score -= 0.3
                reasons.append(f"长宽比={aspect:.1f} (可能是轮胎痕迹)")
                sub_type = AnomalyType.TIRE_TRACK
        elif aspect < self.thresholds.max_aspect_ratio_pool:
            score += 0.1
            reasons.append(f"长宽比={aspect:.1f} (适中)")
        
        # 5. 面积检查
        area = features.area
        if area < self.thresholds.min_area:
            score -= 0.2
            reasons.append(f"面积={area} (过小)")
        else:
            score += 0.1
            reasons.append(f"面积={area}")
        
        return np.clip(score, 0, 1), reasons, sub_type
    
    def discriminate(
        self,
        region_result: RegionAnalysisResult
    ) -> DiscriminationResult:
        """
        对单个区域进行判别
        
        Args:
            region_result: 区域分析结果
        
        Returns:
            DiscriminationResult
        """
        temp_features = region_result.temp_features
        morph_features = region_result.morph_features
        
        all_reasons = []
        
        # 计算各项得分
        temp_score, temp_reasons = self.compute_temperature_score(temp_features)
        morph_score, morph_reasons, sub_type = self.compute_morphology_score(morph_features)
        
        all_reasons.extend(temp_reasons)
        all_reasons.extend(morph_reasons)
        
        # 梯度得分（边缘突变）
        gradient_score = min(1.0, temp_features.edge_gradient_mean / 3.0)
        
        # 综合得分
        w = self.thresholds
        leak_score = (
            w.temp_weight * temp_score +
            w.morph_weight * morph_score +
            w.gradient_weight * gradient_score
        )
        
        # 判定类型
        anomaly_type = self._determine_type(
            leak_score, temp_features, morph_features, sub_type
        )
        
        # 计算置信度
        if anomaly_type in [AnomalyType.LEAK, AnomalyType.LEAK_POOL, AnomalyType.LEAK_PIPELINE]:
            confidence = leak_score
        else:
            confidence = 1.0 - leak_score
        
        return DiscriminationResult(
            region_id=region_result.region_id,
            anomaly_type=anomaly_type,
            confidence=confidence,
            leak_score=leak_score,
            temp_score=temp_score,
            morph_score=morph_score,
            gradient_score=gradient_score,
            reasons=all_reasons
        )
    
    def _determine_type(
        self,
        leak_score: float,
        temp_features: TemperatureFeatures,
        morph_features: MorphologyFeatures,
        sub_type: AnomalyType
    ) -> AnomalyType:
        """
        根据综合特征确定异常类型
        """
        # 高分判定为泄漏
        if leak_score > 0.6:
            # 细分泄漏类型
            if morph_features.linearity > self.thresholds.min_linearity_pipeline:
                return AnomalyType.LEAK_PIPELINE
            elif morph_features.direction_consistency > self.thresholds.min_direction_consistency:
                return AnomalyType.LEAK_POOL
            else:
                return AnomalyType.LEAK
        
        # 中等分数需要进一步判断
        elif leak_score > 0.4:
            # 检查是否更像其他类型
            if sub_type == AnomalyType.EQUIPMENT:
                return AnomalyType.EQUIPMENT
            elif sub_type == AnomalyType.TIRE_TRACK:
                return AnomalyType.TIRE_TRACK
            else:
                return AnomalyType.LEAK  # 倾向于泄漏（宁可误报）
        
        # 低分判定为非泄漏
        else:
            # 温差小 + 无方向 → 土壤
            if (temp_features.delta_t < self.thresholds.max_delta_t_soil and
                morph_features.direction_consistency < self.thresholds.min_direction_consistency):
                return AnomalyType.SOIL
            
            # 圆度高 → 设备
            if morph_features.circularity > self.thresholds.min_circularity_equipment:
                return AnomalyType.EQUIPMENT
            
            # 细长 → 轮胎
            if morph_features.aspect_ratio > self.thresholds.min_aspect_ratio_track:
                return AnomalyType.TIRE_TRACK
            
            return AnomalyType.SOIL
    
    def discriminate_batch(
        self,
        region_results: List[RegionAnalysisResult]
    ) -> List[DiscriminationResult]:
        """
        批量判别多个区域
        """
        return [self.discriminate(r) for r in region_results]


class IntegratedLeakDetector:
    """
    集成泄漏检测器
    
    完整的检测流程：
    1. 温度标定
    2. 热异常检测
    3. 区域分析
    4. 泄漏判别
    """
    
    def __init__(
        self,
        calibration_config: dict = None,
        detection_threshold: float = 0.5,
        discrimination_thresholds: DiscriminationThresholds = None
    ):
        from .thermal_calibration import ThermalCalibrator, CalibrationConfig
        from .zscore_thermal import ZScoreAnomalyDetector
        
        # 初始化组件
        if calibration_config:
            cal_config = CalibrationConfig(**calibration_config)
        else:
            cal_config = CalibrationConfig()
        
        self.calibrator = ThermalCalibrator(cal_config)
        self.anomaly_detector = ZScoreAnomalyDetector()
        self.region_analyzer = RegionAnalyzer()
        self.discriminator = LeakDiscriminator(discrimination_thresholds)
        
        self.detection_threshold = detection_threshold
    
    def detect(
        self,
        ir_image: np.ndarray,
        return_intermediate: bool = False
    ) -> Dict[str, Any]:
        """
        执行完整检测流程
        
        Args:
            ir_image: 红外图像（伪彩色BGR或灰度）
            return_intermediate: 是否返回中间结果
        
        Returns:
            {
                'detections': List[DiscriminationResult],
                'leak_count': int,
                'has_leak': bool,
                'temperature_map': np.ndarray (if return_intermediate),
                'anomaly_map': np.ndarray (if return_intermediate),
                ...
            }
        """
        results = {}
        
        # 1. 温度标定
        cal_result = self.calibrator.calibrate(ir_image)
        temp_map = cal_result.temperature_map
        
        if return_intermediate:
            results['temperature_map'] = temp_map
            results['calibration'] = {
                'r_min': cal_result.r_min,
                'r_max': cal_result.r_max,
                'alpha': cal_result.alpha,
                'beta': cal_result.beta
            }
        
        # 2. 热异常检测
        anomaly_map = self.anomaly_detector.compute_anomaly_map(temp_map)
        
        if return_intermediate:
            results['anomaly_map'] = anomaly_map
        
        # 3. 阈值分割获取候选区域
        binary_mask = (anomaly_map > self.detection_threshold).astype(np.uint8)
        
        # 连通组件分析
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary_mask)
        
        if return_intermediate:
            results['labeled_regions'] = labeled
            results['num_candidates'] = num_features
        
        # 4. 区域分析
        region_results = self.region_analyzer.analyze_multiple_regions(temp_map, labeled)
        
        # 5. 泄漏判别
        disc_results = self.discriminator.discriminate_batch(region_results)
        
        # 统计结果
        leak_detections = [
            d for d in disc_results 
            if d.anomaly_type in [AnomalyType.LEAK, AnomalyType.LEAK_POOL, AnomalyType.LEAK_PIPELINE]
        ]
        
        results['detections'] = disc_results
        results['leak_detections'] = leak_detections
        results['leak_count'] = len(leak_detections)
        results['has_leak'] = len(leak_detections) > 0
        
        if leak_detections:
            results['max_leak_confidence'] = max(d.confidence for d in leak_detections)
        else:
            results['max_leak_confidence'] = 0.0
        
        return results


def classify_thermal_anomaly(
    temp_map: np.ndarray,
    mask: np.ndarray,
    thresholds: DiscriminationThresholds = None
) -> DiscriminationResult:
    """
    便捷函数：分类单个热异常区域
    
    Args:
        temp_map: 温度矩阵
        mask: 区域掩码
        thresholds: 判别阈值
    
    Returns:
        DiscriminationResult
    """
    analyzer = RegionAnalyzer()
    discriminator = LeakDiscriminator(thresholds)
    
    region_result = analyzer.analyze_region(temp_map, mask)
    return discriminator.discriminate(region_result)


# ============ 测试代码 ============
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    
    print("=" * 60)
    print("泄漏判别器测试")
    print("=" * 60)
    
    # 创建三种模拟场景
    h, w = 200, 300
    
    def create_leak_scenario():
        """创建泄漏场景：扇形，中心热，边缘突变"""
        mask = np.zeros((h, w), dtype=np.uint8)
        source = (100, 80)
        
        for angle in range(-40, 50):
            rad = np.radians(angle)
            for r in range(10, 70):
                x = int(source[1] + r * np.cos(rad))
                y = int(source[0] + r * np.sin(rad))
                if 0 <= x < w and 0 <= y < h:
                    mask[y, x] = 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 温度：中心10°C，边缘-10°C，突变
        temp = np.full((h, w), -15.0, dtype=np.float32)
        dist = np.sqrt((np.arange(h)[:, None] - source[0])**2 + 
                      (np.arange(w)[None, :] - source[1])**2)
        leak_temp = 10 - 0.3 * dist
        leak_temp = np.clip(leak_temp, -10, 10)
        leak_temp += np.random.normal(0, 0.3, (h, w))
        
        # 边缘突变
        boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        leak_temp[boundary > 0] = -12
        
        temp = np.where(mask > 0, leak_temp, temp)
        return mask, temp, "泄漏（扇形扩散）"
    
    def create_soil_scenario():
        """创建土壤场景：随机块状，温度均匀"""
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (220, 100), (35, 25), 30, 0, 360, 1, -1)
        
        temp = np.full((h, w), -15.0, dtype=np.float32)
        soil_temp = 3.0 + np.random.normal(0, 0.5, (h, w))  # 均匀温度
        temp = np.where(mask > 0, soil_temp, temp)
        
        return mask, temp, "土壤（均匀加热）"
    
    def create_pipeline_scenario():
        """创建管道泄漏场景：线性，沿管道高温"""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 绘制管道形状
        pts = np.array([[50, 150], [150, 155], [155, 160], [55, 155]], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
        
        temp = np.full((h, w), -15.0, dtype=np.float32)
        pipe_temp = 8.0 + np.random.normal(0, 0.5, (h, w))
        temp = np.where(mask > 0, pipe_temp, temp)
        
        return mask, temp, "管道泄漏（线性）"
    
    # 创建场景
    scenarios = [
        create_leak_scenario(),
        create_soil_scenario(),
        create_pipeline_scenario()
    ]
    
    # 初始化判别器
    discriminator = LeakDiscriminator()
    analyzer = RegionAnalyzer()
    
    # 分析并可视化
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for i, (mask, temp, name) in enumerate(scenarios):
        # 分析区域
        region_result = analyzer.analyze_region(temp, mask, region_id=i)
        disc_result = discriminator.discriminate(region_result)
        
        # 打印结果
        print(f"\n{'='*50}")
        print(f"场景 {i+1}: {name}")
        print(f"{'='*50}")
        print(f"判定类型: {disc_result.anomaly_type.value}")
        print(f"置信度: {disc_result.confidence:.2f}")
        print(f"泄漏得分: {disc_result.leak_score:.2f}")
        print(f"  - 温度得分: {disc_result.temp_score:.2f}")
        print(f"  - 形态得分: {disc_result.morph_score:.2f}")
        print(f"  - 梯度得分: {disc_result.gradient_score:.2f}")
        print("判别依据:")
        for reason in disc_result.reasons:
            print(f"  · {reason}")
        
        # 可视化
        # 温度图
        im = axes[i, 0].imshow(temp, cmap='coolwarm', vmin=-15, vmax=10)
        axes[i, 0].set_title(f'{name}\n温度分布')
        plt.colorbar(im, ax=axes[i, 0], label='°C')
        
        # 掩码
        axes[i, 1].imshow(mask, cmap='Reds')
        axes[i, 1].set_title('区域掩码')
        
        # 判定结果
        result_color = 'red' if 'leak' in disc_result.anomaly_type.value else 'green'
        axes[i, 2].text(0.5, 0.7, f"判定: {disc_result.anomaly_type.value}",
                       ha='center', va='center', fontsize=14, 
                       color=result_color, weight='bold',
                       transform=axes[i, 2].transAxes)
        axes[i, 2].text(0.5, 0.5, f"置信度: {disc_result.confidence:.2f}",
                       ha='center', va='center', fontsize=12,
                       transform=axes[i, 2].transAxes)
        axes[i, 2].text(0.5, 0.3, f"ΔT={region_result.temp_features.delta_t:.1f}°C",
                       ha='center', va='center', fontsize=10,
                       transform=axes[i, 2].transAxes)
        axes[i, 2].text(0.5, 0.15, f"方向一致性={region_result.morph_features.direction_consistency:.2f}",
                       ha='center', va='center', fontsize=10,
                       transform=axes[i, 2].transAxes)
        axes[i, 2].axis('off')
        axes[i, 2].set_title('判定结果')
    
    plt.tight_layout()
    plt.savefig('/tmp/leak_discriminator_test.png', dpi=150)
    print(f"\n可视化保存到: /tmp/leak_discriminator_test.png")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
