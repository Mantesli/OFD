"""
Z-Score Thermal Anomaly Detection
=================================
基于局部Z-score的红外温度异常检测，与纹理特征互补。

核心原理：
- 使用滑动窗口计算局部统计量（均值、标准差）
- 参考背景：当前帧的周围区域
- Z-score = (像素值 - 局部均值) / 局部标准差
- Z-score高的区域为温度异常区域

与纹理特征的区别：
- Z-score: 检测"是否异常热/冷"（温度偏离程度）
- 纹理特征: 检测"热区的形态"（均匀vs分散）

两者结合可以更准确识别泄漏：
- 泄漏 = Z-score异常 + 均匀热区
- 黑土 = Z-score异常 + 分散热块
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import ndimage
from scipy.stats import zscore
from loguru import logger


@dataclass
class ZScoreConfig:
    """Z-score配置"""
    # 局部窗口大小（用于计算局部统计量）
    window_size: int = 64
    
    # 窗口步长（重叠计算）
    stride: int = 32
    
    # Z-score阈值（超过此值认为异常）
    zscore_threshold: float = 2.0
    
    # 是否使用MAD（中位数绝对偏差）代替标准差
    # MAD对异常值更鲁棒
    use_mad: bool = True
    
    # 形态学处理
    morphology_kernel_size: int = 5
    
    # 最小异常区域面积（过滤噪声）
    min_area: int = 100


class ZScoreAnomalyDetector:
    """
    基于Z-score的温度异常检测器
    
    使用滑动窗口计算局部Z-score，检测温度异常区域。
    
    Usage:
        detector = ZScoreAnomalyDetector(config)
        
        # 获取异常分数图
        anomaly_map = detector.compute_anomaly_map(ir_image)
        
        # 获取异常区域
        regions = detector.detect_anomaly_regions(ir_image)
    """
    
    def __init__(self, config: Optional[ZScoreConfig] = None):
        self.config = config or ZScoreConfig()
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理红外图像"""
        # 转换为灰度
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image[:, :, 0]
        else:
            gray = image.copy()
        
        # 转换为float
        gray = gray.astype(np.float32)
        
        return gray
    
    def compute_local_zscore(self, image: np.ndarray) -> np.ndarray:
        """
        计算局部Z-score图
        
        使用滑动窗口，每个像素的Z-score基于其周围区域计算
        
        Args:
            image: 灰度图像
            
        Returns:
            Z-score图，与输入同尺寸
        """
        h, w = image.shape
        ws = self.config.window_size
        
        # 使用均值滤波器计算局部均值
        local_mean = cv2.blur(image, (ws, ws))
        
        # 计算局部标准差或MAD
        if self.config.use_mad:
            # MAD = median(|X - median(X)|)
            # 近似计算：使用局部中值滤波
            local_median = cv2.medianBlur(image.astype(np.float32), ws if ws % 2 == 1 else ws + 1)
            local_deviation = np.abs(image - local_median)
            local_mad = cv2.medianBlur(local_deviation.astype(np.float32), ws if ws % 2 == 1 else ws + 1)
            # MAD to std: std ≈ 1.4826 * MAD
            local_std = 1.4826 * local_mad
        else:
            # 计算局部方差：E[X^2] - E[X]^2
            local_sq_mean = cv2.blur(image ** 2, (ws, ws))
            local_var = local_sq_mean - local_mean ** 2
            local_var = np.maximum(local_var, 0)  # 避免数值误差导致负数
            local_std = np.sqrt(local_var)
        
        # 避免除零
        local_std = np.maximum(local_std, 1e-8)
        
        # 计算Z-score
        zscore_map = (image - local_mean) / local_std
        
        return zscore_map
    
    def compute_anomaly_map(
        self, 
        image: np.ndarray,
        return_zscore: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        计算异常分数图
        
        Args:
            image: 红外图像
            return_zscore: 是否同时返回Z-score图
            
        Returns:
            异常分数图 [0, 1]，高分表示更可能异常
        """
        gray = self.preprocess(image)
        
        # 计算Z-score
        zscore_map = self.compute_local_zscore(gray)
        
        # 转换为异常分数（取绝对值，因为过热和过冷都是异常）
        # 但对于油泄漏，通常是偏热，所以可以只看正向
        # 这里我们主要关注高温异常
        anomaly_map = np.maximum(zscore_map, 0)  # 只保留正向（偏热）
        
        # 归一化到[0, 1]
        # 使用sigmoid风格的归一化
        threshold = self.config.zscore_threshold
        anomaly_map = 1 / (1 + np.exp(-(anomaly_map - threshold)))
        
        # 可选：形态学平滑
        if self.config.morphology_kernel_size > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (self.config.morphology_kernel_size, self.config.morphology_kernel_size)
            )
            anomaly_map = cv2.morphologyEx(anomaly_map, cv2.MORPH_CLOSE, kernel)
            anomaly_map = cv2.morphologyEx(anomaly_map, cv2.MORPH_OPEN, kernel)
        
        if return_zscore:
            return anomaly_map, zscore_map
        return anomaly_map
    
    def detect_anomaly_regions(
        self, 
        image: np.ndarray,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        检测异常区域
        
        Args:
            image: 红外图像
            threshold: 二值化阈值，None则使用配置值
            
        Returns:
            异常区域列表，每个区域包含 {bbox, area, mean_zscore, ...}
        """
        anomaly_map, zscore_map = self.compute_anomaly_map(image, return_zscore=True)
        
        # 二值化
        threshold = threshold or 0.5
        binary = (anomaly_map > threshold).astype(np.uint8)
        
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        regions = []
        for i in range(1, num_labels):  # 跳过背景（标签0）
            area = stats[i, cv2.CC_STAT_AREA]
            
            # 过滤太小的区域
            if area < self.config.min_area:
                continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 该区域的mask
            region_mask = labels == i
            
            # 计算区域统计
            region_zscores = zscore_map[region_mask]
            region_anomaly = anomaly_map[region_mask]
            
            regions.append({
                'id': i,
                'bbox': (x, y, w, h),
                'centroid': (centroids[i][0], centroids[i][1]),
                'area': area,
                'mean_zscore': float(np.mean(region_zscores)),
                'max_zscore': float(np.max(region_zscores)),
                'mean_anomaly_score': float(np.mean(region_anomaly)),
                'mask': region_mask
            })
        
        # 按异常分数排序
        regions.sort(key=lambda r: r['mean_anomaly_score'], reverse=True)
        
        return regions
    
    def compute_anomaly_score(self, image: np.ndarray) -> float:
        """
        计算整体异常分数（用于图像级分类）
        
        Args:
            image: 红外图像
            
        Returns:
            异常分数 [0, 1]
        """
        anomaly_map = self.compute_anomaly_map(image)
        
        # 综合考虑多个因素
        # 1. 最大异常值
        max_anomaly = np.max(anomaly_map)
        
        # 2. 高异常区域的面积占比
        high_anomaly_ratio = np.mean(anomaly_map > 0.5)
        
        # 3. 平均异常值
        mean_anomaly = np.mean(anomaly_map)
        
        # 加权组合
        score = 0.5 * max_anomaly + 0.3 * high_anomaly_ratio + 0.2 * mean_anomaly
        
        return float(np.clip(score, 0, 1))


class AdaptiveZScoreDetector(ZScoreAnomalyDetector):
    """
    自适应Z-score检测器
    
    根据图像内容自适应调整参数：
    - 如果整体温度分布均匀，降低阈值
    - 如果整体温度变化大，提高阈值
    """
    
    def __init__(self, config: Optional[ZScoreConfig] = None):
        super().__init__(config)
        self.history_stats: List[Dict] = []
    
    def compute_adaptive_threshold(self, image: np.ndarray) -> float:
        """计算自适应阈值"""
        gray = self.preprocess(image)
        
        # 全局统计
        global_std = np.std(gray)
        global_range = np.max(gray) - np.min(gray)
        
        # 基础阈值
        base_threshold = self.config.zscore_threshold
        
        # 自适应调整
        # 如果全局变化小，降低阈值（更敏感）
        # 如果全局变化大，提高阈值（更严格）
        if global_std < 20:  # 温度分布均匀
            adaptive_threshold = base_threshold * 0.8
        elif global_std > 50:  # 温度分布变化大
            adaptive_threshold = base_threshold * 1.2
        else:
            adaptive_threshold = base_threshold
        
        return adaptive_threshold
    
    def compute_anomaly_map(
        self, 
        image: np.ndarray,
        return_zscore: bool = False,
        use_adaptive: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """计算异常图（带自适应阈值）"""
        if use_adaptive:
            original_threshold = self.config.zscore_threshold
            self.config.zscore_threshold = self.compute_adaptive_threshold(image)
            result = super().compute_anomaly_map(image, return_zscore)
            self.config.zscore_threshold = original_threshold
            return result
        else:
            return super().compute_anomaly_map(image, return_zscore)


# ============================================================================
# 融合模块：Z-score + 纹理特征
# ============================================================================

class ThermalAnomalyFusion:
    """
    热异常融合检测器
    
    融合Z-score异常和纹理特征，综合判断泄漏
    
    核心逻辑：
    - Z-score高 + 温度均匀 → 高概率泄漏
    - Z-score高 + 温度分散 → 可能是黑土
    - Z-score低 → 正常区域
    """
    
    def __init__(
        self,
        zscore_config: Optional[ZScoreConfig] = None,
        zscore_weight: float = 0.5,
        texture_weight: float = 0.5
    ):
        from .thermal_texture import ThermalTextureExtractor
        
        self.zscore_detector = ZScoreAnomalyDetector(zscore_config)
        self.texture_extractor = ThermalTextureExtractor()
        
        self.zscore_weight = zscore_weight
        self.texture_weight = texture_weight
    
    def compute_fused_score(self, ir_image: np.ndarray) -> Dict:
        """
        计算融合异常分数
        
        Args:
            ir_image: 红外图像
            
        Returns:
            {
                'fused_score': 融合分数,
                'zscore_score': Z-score分数,
                'texture_score': 纹理分数,
                'is_likely_leak': 是否可能泄漏,
                'is_likely_soil': 是否可能黑土
            }
        """
        # Z-score异常分数
        zscore_score = self.zscore_detector.compute_anomaly_score(ir_image)
        
        # 纹理异常分数（均匀性）
        texture_score = self.texture_extractor.compute_anomaly_score(ir_image)
        
        # 融合分数
        fused_score = (
            self.zscore_weight * zscore_score +
            self.texture_weight * texture_score
        )
        
        # 判断类型
        # 泄漏：两者都高
        # 黑土：Z-score高但纹理分数低（不均匀）
        is_likely_leak = zscore_score > 0.5 and texture_score > 0.5
        is_likely_soil = zscore_score > 0.5 and texture_score < 0.4
        
        return {
            'fused_score': float(fused_score),
            'zscore_score': float(zscore_score),
            'texture_score': float(texture_score),
            'is_likely_leak': is_likely_leak,
            'is_likely_soil': is_likely_soil
        }


# ============================================================================
# 便捷函数
# ============================================================================

def compute_zscore_anomaly(
    image: np.ndarray,
    window_size: int = 64,
    threshold: float = 2.0
) -> np.ndarray:
    """
    便捷函数：计算Z-score异常图
    
    Args:
        image: 红外图像
        window_size: 局部窗口大小
        threshold: Z-score阈值
        
    Returns:
        异常分数图 [0, 1]
    """
    config = ZScoreConfig(window_size=window_size, zscore_threshold=threshold)
    detector = ZScoreAnomalyDetector(config)
    return detector.compute_anomaly_map(image)


def detect_hot_regions(
    image: np.ndarray,
    min_area: int = 100,
    threshold: float = 0.5
) -> List[Dict]:
    """
    便捷函数：检测热异常区域
    
    Args:
        image: 红外图像
        min_area: 最小区域面积
        threshold: 检测阈值
        
    Returns:
        区域列表
    """
    config = ZScoreConfig(min_area=min_area)
    detector = ZScoreAnomalyDetector(config)
    return detector.detect_anomaly_regions(image, threshold)


if __name__ == "__main__":
    # 测试代码
    print("Testing Z-Score Anomaly Detector...")
    
    # 创建测试图像
    # 模拟红外图像：背景温度 + 热异常区域
    np.random.seed(42)
    
    # 背景：均值100，标准差10
    background = np.random.normal(100, 10, (256, 256)).astype(np.float32)
    
    # 添加一个均匀热区（模拟泄漏）：左上角，高温且均匀
    background[50:100, 50:120] = np.random.normal(150, 5, (50, 70))
    
    # 添加分散热块（模拟黑土）：右下角，高温但分散
    for _ in range(20):
        x, y = np.random.randint(150, 230), np.random.randint(150, 230)
        size = np.random.randint(5, 15)
        background[y:y+size, x:x+size] = np.random.normal(145, 15, (size, size))
    
    # 归一化到0-255
    test_image = np.clip(background, 0, 255).astype(np.uint8)
    
    # 检测
    detector = ZScoreAnomalyDetector(ZScoreConfig(window_size=32))
    anomaly_map, zscore_map = detector.compute_anomaly_map(test_image, return_zscore=True)
    
    print(f"Z-score range: [{zscore_map.min():.2f}, {zscore_map.max():.2f}]")
    print(f"Anomaly map range: [{anomaly_map.min():.2f}, {anomaly_map.max():.2f}]")
    
    # 检测区域
    regions = detector.detect_anomaly_regions(test_image)
    print(f"\nDetected {len(regions)} anomaly regions:")
    for r in regions[:3]:
        print(f"  Region {r['id']}: area={r['area']}, mean_zscore={r['mean_zscore']:.2f}")
    
    # 整体分数
    score = detector.compute_anomaly_score(test_image)
    print(f"\nOverall anomaly score: {score:.4f}")
