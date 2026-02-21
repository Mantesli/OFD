"""
Module 3: Thermal Texture Feature Extractor
===========================================
从红外图像中提取温度纹理特征，用于区分泄漏和干扰物。

核心原理：
- 油泄漏：温度分布相对均匀，形成连续的热区
- 裸露黑土：温度分布随机，呈现分散的小热块
- 抽油平台：规则形状，温度分布有明显边界

提取的特征：
1. 局部温度统计（方差、熵、梯度）
2. LBP纹理特征
3. 连通域分析
4. 温度一致性指标
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger

from scipy import ndimage
from scipy.stats import entropy
from skimage.feature import local_binary_pattern
from skimage.measure import label, regionprops
from skimage.filters import gabor


@dataclass
class ThermalConfig:
    """温度特征提取配置"""
    # LBP参数
    lbp_radius: int = 3
    lbp_n_points: int = 24
    lbp_method: str = "uniform"
    
    # 统计特征窗口
    window_size: int = 32
    
    # 连通域分析
    min_area: int = 100
    max_area: int = 50000
    
    # 温度阈值（用于二值化）
    # 这些值需要根据实际红外图像的温度范围调整
    hot_threshold_percentile: float = 80  # 高于此百分位数为热区
    
    # Gabor滤波器参数
    gabor_frequencies: List[float] = None
    gabor_thetas: List[float] = None
    
    def __post_init__(self):
        if self.gabor_frequencies is None:
            self.gabor_frequencies = [0.1, 0.2, 0.3]
        if self.gabor_thetas is None:
            self.gabor_thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]


class ThermalTextureExtractor:
    """
    红外温度纹理特征提取器
    
    从红外图像中提取多种纹理和统计特征，用于区分：
    - 油泄漏（均匀热区）
    - 裸露黑土（分散小热块）
    - 抽油平台（规则热区）
    
    Usage:
        extractor = ThermalTextureExtractor(config)
        features = extractor.extract_features(ir_image)
        anomaly_score = extractor.compute_anomaly_score(ir_image)
    """
    
    def __init__(self, config: Optional[ThermalConfig] = None):
        self.config = config or ThermalConfig()
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        预处理红外图像
        
        Args:
            image: 输入图像，可以是彩色或灰度
            
        Returns:
            归一化的灰度图像 [0, 1]
        """
        # 转换为灰度
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image[:, :, 0]
        else:
            gray = image.copy()
        
        # 归一化到[0, 1]
        gray = gray.astype(np.float32)
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        
        return gray
    
    def extract_lbp_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        提取LBP（局部二值模式）纹理特征
        
        LBP对于区分纹理模式非常有效：
        - 均匀纹理（油）：LBP直方图集中在少数bin
        - 随机纹理（黑土）：LBP直方图分散
        
        Args:
            image: 灰度图像 [0, 1]
            
        Returns:
            LBP特征字典
        """
        # 转换为uint8以加速计算
        img_uint8 = (image * 255).astype(np.uint8)
        
        # 计算LBP
        lbp = local_binary_pattern(
            img_uint8, 
            P=self.config.lbp_n_points, 
            R=self.config.lbp_radius, 
            method=self.config.lbp_method
        )
        
        # 计算LBP直方图
        n_bins = self.config.lbp_n_points + 2  # uniform LBP的bin数
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        # 从直方图提取特征
        features = {
            'lbp_entropy': float(entropy(hist + 1e-10)),  # 熵越低，纹理越均匀
            'lbp_uniformity': float(np.sum(hist ** 2)),    # 均匀度，越高越均匀
            'lbp_dominant_ratio': float(hist.max()),       # 主导模式占比
            'lbp_variance': float(np.var(hist)),           # 方差
        }
        
        return features
    
    def extract_statistical_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        提取局部统计特征
        
        关键区分点：
        - 油泄漏：局部方差低（均匀），全局熵低
        - 黑土：局部方差高（随机），全局熵高
        
        Args:
            image: 灰度图像 [0, 1]
            
        Returns:
            统计特征字典
        """
        ws = self.config.window_size
        h, w = image.shape
        
        # 全局统计
        global_mean = float(np.mean(image))
        global_std = float(np.std(image))
        global_skewness = float(self._compute_skewness(image.ravel()))
        global_kurtosis = float(self._compute_kurtosis(image.ravel()))
        
        # 计算图像熵
        hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 1), density=True)
        global_entropy = float(entropy(hist + 1e-10))
        
        # 局部统计（滑动窗口）
        local_vars = []
        local_entropies = []
        
        for i in range(0, h - ws, ws // 2):
            for j in range(0, w - ws, ws // 2):
                window = image[i:i+ws, j:j+ws]
                local_vars.append(np.var(window))
                
                # 局部熵
                local_hist, _ = np.histogram(window.ravel(), bins=32, range=(0, 1), density=True)
                local_entropies.append(entropy(local_hist + 1e-10))
        
        local_vars = np.array(local_vars)
        local_entropies = np.array(local_entropies)
        
        features = {
            'global_mean': global_mean,
            'global_std': global_std,
            'global_entropy': global_entropy,
            'global_skewness': global_skewness,
            'global_kurtosis': global_kurtosis,
            
            # 局部方差统计
            'local_var_mean': float(np.mean(local_vars)),
            'local_var_std': float(np.std(local_vars)),
            'local_var_max': float(np.max(local_vars)),
            
            # 局部熵统计
            'local_entropy_mean': float(np.mean(local_entropies)),
            'local_entropy_std': float(np.std(local_entropies)),
            
            # 温度一致性指标（低方差=高一致性）
            'temperature_consistency': float(1 - np.mean(local_vars) / (global_std + 1e-8)),
        }
        
        return features
    
    def extract_gradient_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        提取梯度特征
        
        区分点：
        - 油泄漏：边缘平滑，梯度较低
        - 黑土：边缘锐利，梯度高
        - 抽油平台：有明显的直线边缘
        
        Args:
            image: 灰度图像 [0, 1]
            
        Returns:
            梯度特征字典
        """
        # 转换为uint8
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Sobel梯度
        grad_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_dir = np.arctan2(grad_y, grad_x)
        
        # Laplacian
        laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
        
        # 梯度方向直方图（检测规则形状）
        dir_hist, _ = np.histogram(grad_dir.ravel(), bins=8, range=(-np.pi, np.pi), density=True)
        dir_entropy = entropy(dir_hist + 1e-10)
        
        features = {
            'gradient_mean': float(np.mean(grad_mag)),
            'gradient_std': float(np.std(grad_mag)),
            'gradient_max': float(np.max(grad_mag)),
            
            # 方向熵（低=有主导方向，如直线边缘）
            'gradient_direction_entropy': float(dir_entropy),
            
            # Laplacian统计
            'laplacian_mean': float(np.mean(np.abs(laplacian))),
            'laplacian_std': float(np.std(laplacian)),
            
            # 边缘平滑度指标
            'edge_smoothness': float(1 - np.mean(grad_mag) / (np.max(grad_mag) + 1e-8)),
        }
        
        return features
    
    def extract_connected_component_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        提取连通域特征
        
        区分点：
        - 油泄漏：大的连续热区
        - 黑土：多个小的分散热块
        
        Args:
            image: 灰度图像 [0, 1]
            
        Returns:
            连通域特征字典
        """
        # 二值化（提取热区）
        threshold = np.percentile(image, self.config.hot_threshold_percentile)
        binary = (image > threshold).astype(np.uint8)
        
        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 连通域分析
        labeled = label(binary)
        regions = regionprops(labeled, intensity_image=image)
        
        if len(regions) == 0:
            return {
                'num_hot_regions': 0,
                'largest_region_area': 0,
                'total_hot_area': 0,
                'region_area_std': 0,
                'mean_region_solidity': 0,
                'mean_region_eccentricity': 0,
                'region_compactness': 0,
            }
        
        # 过滤太小或太大的区域
        valid_regions = [r for r in regions 
                        if self.config.min_area <= r.area <= self.config.max_area]
        
        if len(valid_regions) == 0:
            valid_regions = regions  # 如果全部被过滤，使用原始结果
        
        # 提取特征
        areas = [r.area for r in valid_regions]
        solidities = [r.solidity for r in valid_regions]
        eccentricities = [r.eccentricity for r in valid_regions]
        
        # 计算区域紧凑度（周长^2 / 面积）
        compactnesses = []
        for r in valid_regions:
            if r.perimeter > 0:
                compactness = r.area / (r.perimeter ** 2)
            else:
                compactness = 0
            compactnesses.append(compactness)
        
        features = {
            'num_hot_regions': len(valid_regions),
            'largest_region_area': float(max(areas)),
            'total_hot_area': float(sum(areas)),
            'region_area_std': float(np.std(areas)) if len(areas) > 1 else 0,
            'region_area_ratio': float(max(areas) / (sum(areas) + 1e-8)),  # 最大区域占比
            
            # 形状特征
            'mean_region_solidity': float(np.mean(solidities)),  # 凸性
            'mean_region_eccentricity': float(np.mean(eccentricities)),  # 离心率
            'region_compactness': float(np.mean(compactnesses)),  # 紧凑度
        }
        
        return features
    
    def extract_gabor_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        提取Gabor纹理特征
        
        Gabor滤波器对纹理方向和频率敏感：
        - 油泄漏：低频响应强，各方向均匀
        - 黑土：高频响应强，方向随机
        
        Args:
            image: 灰度图像 [0, 1]
            
        Returns:
            Gabor特征字典
        """
        responses = []
        
        for freq in self.config.gabor_frequencies:
            for theta in self.config.gabor_thetas:
                filt_real, filt_imag = gabor(image, frequency=freq, theta=theta)
                response = np.sqrt(filt_real**2 + filt_imag**2)
                responses.append({
                    'freq': freq,
                    'theta': theta,
                    'mean': np.mean(response),
                    'std': np.std(response)
                })
        
        # 聚合特征
        means = [r['mean'] for r in responses]
        stds = [r['std'] for r in responses]
        
        # 按频率分组
        low_freq_response = np.mean([r['mean'] for r in responses if r['freq'] == min(self.config.gabor_frequencies)])
        high_freq_response = np.mean([r['mean'] for r in responses if r['freq'] == max(self.config.gabor_frequencies)])
        
        features = {
            'gabor_mean': float(np.mean(means)),
            'gabor_std': float(np.mean(stds)),
            'gabor_low_freq': float(low_freq_response),
            'gabor_high_freq': float(high_freq_response),
            'gabor_freq_ratio': float(low_freq_response / (high_freq_response + 1e-8)),  # 低/高频比
            'gabor_direction_uniformity': float(np.std(means)),  # 方向均匀性
        }
        
        return features
    
    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        提取所有特征
        
        Args:
            image: 输入红外图像
            
        Returns:
            特征字典
        """
        # 预处理
        gray = self.preprocess(image)
        
        # 提取各类特征
        features = {}
        
        # LBP纹理特征
        features.update(self.extract_lbp_features(gray))
        
        # 统计特征
        features.update(self.extract_statistical_features(gray))
        
        # 梯度特征
        features.update(self.extract_gradient_features(gray))
        
        # 连通域特征
        features.update(self.extract_connected_component_features(gray))
        
        # Gabor特征
        features.update(self.extract_gabor_features(gray))
        
        return features
    
    def extract_features_batch(self, images: List[np.ndarray]) -> List[Dict[str, float]]:
        """批量提取特征"""
        return [self.extract_features(img) for img in images]
    
    def features_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """将特征字典转换为固定顺序的向量"""
        # 定义固定的特征顺序
        feature_names = sorted(features.keys())
        return np.array([features[name] for name in feature_names])
    
    def compute_anomaly_score(self, image: np.ndarray) -> float:
        """
        计算热异常分数 A_ir(x, y)
        
        基于关键特征的加权组合：
        - 温度一致性高 → 更可能是泄漏
        - 连通域大且少 → 更可能是泄漏
        - 纹理均匀 → 更可能是泄漏
        
        Args:
            image: 输入红外图像
            
        Returns:
            异常分数 [0, 1]，越高越可能是泄漏
        """
        features = self.extract_features(image)
        
        # 关键指标
        consistency = features.get('temperature_consistency', 0)
        uniformity = features.get('lbp_uniformity', 0)
        region_ratio = features.get('region_area_ratio', 0)
        edge_smoothness = features.get('edge_smoothness', 0)
        
        # 反向指标（越低越好）
        num_regions = features.get('num_hot_regions', 1)
        local_var = features.get('local_var_mean', 0)
        
        # 归一化
        num_regions_score = 1 / (1 + num_regions / 5)  # 5个区域时得分0.5
        var_score = 1 / (1 + local_var * 10)
        
        # 加权组合
        score = (
            0.25 * consistency +
            0.15 * uniformity +
            0.20 * region_ratio +
            0.15 * edge_smoothness +
            0.15 * num_regions_score +
            0.10 * var_score
        )
        
        # 裁剪到[0, 1]
        score = np.clip(score, 0, 1)
        
        return float(score)
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-8:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-8:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3)


def extract_thermal_features(
    images: Union[np.ndarray, List[np.ndarray]],
    config: Optional[ThermalConfig] = None
) -> Union[Dict[str, float], List[Dict[str, float]]]:
    """
    便捷函数：提取热特征
    
    Args:
        images: 单张图像或图像列表
        config: 配置
        
    Returns:
        特征字典或列表
    """
    extractor = ThermalTextureExtractor(config)
    
    if isinstance(images, np.ndarray) and len(images.shape) <= 3:
        return extractor.extract_features(images)
    else:
        return extractor.extract_features_batch(images)


def compute_thermal_anomaly_score(
    images: Union[np.ndarray, List[np.ndarray]],
    config: Optional[ThermalConfig] = None
) -> Union[float, List[float]]:
    """
    便捷函数：计算热异常分数
    
    Args:
        images: 单张图像或图像列表
        config: 配置
        
    Returns:
        异常分数或列表
    """
    extractor = ThermalTextureExtractor(config)
    
    if isinstance(images, np.ndarray) and len(images.shape) <= 3:
        return extractor.compute_anomaly_score(images)
    else:
        return [extractor.compute_anomaly_score(img) for img in images]


class ThermalFeatureValidator:
    """
    热特征验证器
    
    用于Week 2实验2：验证温度纹理特征的区分能力
    """
    
    def __init__(self, config: Optional[ThermalConfig] = None):
        self.extractor = ThermalTextureExtractor(config)
    
    def analyze_feature_distributions(
        self,
        leak_images: List[np.ndarray],
        normal_images: List[np.ndarray]
    ) -> Dict[str, Dict]:
        """
        分析泄漏vs正常图像的特征分布差异
        
        Args:
            leak_images: 泄漏图像列表
            normal_images: 正常图像列表
            
        Returns:
            各特征的分布统计
        """
        # 提取特征
        leak_features = self.extractor.extract_features_batch(leak_images)
        normal_features = self.extractor.extract_features_batch(normal_images)
        
        # 获取特征名
        feature_names = list(leak_features[0].keys())
        
        # 分析每个特征的分布
        analysis = {}
        for name in feature_names:
            leak_values = [f[name] for f in leak_features]
            normal_values = [f[name] for f in normal_features]
            
            leak_mean = np.mean(leak_values)
            normal_mean = np.mean(normal_values)
            
            # 计算区分度（效果量）
            pooled_std = np.sqrt((np.var(leak_values) + np.var(normal_values)) / 2)
            if pooled_std > 1e-8:
                effect_size = abs(leak_mean - normal_mean) / pooled_std
            else:
                effect_size = 0
            
            analysis[name] = {
                'leak_mean': float(leak_mean),
                'leak_std': float(np.std(leak_values)),
                'normal_mean': float(normal_mean),
                'normal_std': float(np.std(normal_values)),
                'effect_size': float(effect_size),  # Cohen's d
                'discriminative': effect_size > 0.5  # 中等以上效果
            }
        
        return analysis
    
    def rank_features_by_importance(
        self,
        analysis: Dict[str, Dict]
    ) -> List[Tuple[str, float]]:
        """
        按区分能力排序特征
        
        Args:
            analysis: analyze_feature_distributions的输出
            
        Returns:
            (特征名, 效果量) 列表，按效果量降序排列
        """
        ranked = [(name, info['effect_size']) for name, info in analysis.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Thermal Feature Extractor")
    parser.add_argument("--image", type=str, help="Test IR image path")
    args = parser.parse_args()
    
    extractor = ThermalTextureExtractor()
    
    if args.image:
        # 测试真实图像
        image = cv2.imread(args.image)
        features = extractor.extract_features(image)
        
        print("Extracted features:")
        for name, value in sorted(features.items()):
            print(f"  {name}: {value:.4f}")
        
        score = extractor.compute_anomaly_score(image)
        print(f"\nAnomaly score: {score:.4f}")
    else:
        # 测试合成数据
        print("Testing with synthetic data...")
        
        # 模拟均匀热区（泄漏）
        uniform_image = np.random.normal(0.7, 0.05, (256, 256))
        uniform_image = np.clip(uniform_image, 0, 1)
        
        # 模拟随机热块（黑土）
        random_image = np.random.uniform(0.2, 0.9, (256, 256))
        
        print("\nUniform (leak-like) image:")
        uniform_features = extractor.extract_features(uniform_image)
        print(f"  Temperature consistency: {uniform_features['temperature_consistency']:.4f}")
        print(f"  LBP uniformity: {uniform_features['lbp_uniformity']:.4f}")
        print(f"  Anomaly score: {extractor.compute_anomaly_score(uniform_image):.4f}")
        
        print("\nRandom (soil-like) image:")
        random_features = extractor.extract_features(random_image)
        print(f"  Temperature consistency: {random_features['temperature_consistency']:.4f}")
        print(f"  LBP uniformity: {random_features['lbp_uniformity']:.4f}")
        print(f"  Anomaly score: {extractor.compute_anomaly_score(random_image):.4f}")
