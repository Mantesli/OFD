"""
Region Analyzer Module
======================
分析候选区域的温度特征和形态特征，用于区分泄漏与干扰。

核心特征：
1. 温度特征：
   - 中心-边缘温差 ΔT
   - 区域内温度方差
   - 边缘梯度强度（突变程度）

2. 形态特征：
   - 扇形/舌状检测（方向一致性）
   - 线性结构检测（管道泄漏）
   - 形状描述子（圆度、凸包比等）

物理依据：
- 泄漏：中心高温+边缘突变+扇形扩散方向
- 黑色土壤：均匀温度+无明确方向+随机形状
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from scipy import ndimage
from skimage import measure, morphology
from skimage.feature import peak_local_max
import json
from pathlib import Path
from loguru import logger


@dataclass
class TemperatureFeatures:
    """温度特征"""
    center_temp: float          # 中心区域平均温度
    edge_temp: float            # 边缘区域平均温度
    delta_t: float              # 中心-边缘温差
    center_variance: float      # 中心区域方差
    edge_variance: float        # 边缘区域方差
    overall_variance: float     # 整体方差
    edge_gradient_mean: float   # 边缘梯度均值（突变程度）
    edge_gradient_std: float    # 边缘梯度标准差
    temp_range: Tuple[float, float]  # 区域温度范围
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'center_temp': self.center_temp,
            'edge_temp': self.edge_temp,
            'delta_t': self.delta_t,
            'center_variance': self.center_variance,
            'edge_variance': self.edge_variance,
            'overall_variance': self.overall_variance,
            'edge_gradient_mean': self.edge_gradient_mean,
            'edge_gradient_std': self.edge_gradient_std,
            'temp_min': self.temp_range[0],
            'temp_max': self.temp_range[1]
        }


@dataclass
class MorphologyFeatures:
    """形态特征"""
    area: int                   # 面积（像素数）
    perimeter: float            # 周长
    circularity: float          # 圆度 = 4π × area / perimeter²
    convexity: float            # 凸包填充率 = area / convex_hull_area
    aspect_ratio: float         # 长宽比
    eccentricity: float         # 偏心率 (0=圆, 1=线)
    orientation: float          # 主轴方向（弧度）
    solidity: float             # 实心度 = area / convex_hull_area
    extent: float               # 范围 = area / bounding_box_area
    
    # 扇形/舌状特征
    direction_consistency: float  # 方向一致性 (0-1)
    spread_angle: float           # 扩散角度
    
    # 线性特征（管道检测）
    linearity: float              # 线性度 (0-1)
    skeleton_length: float        # 骨架长度
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'area': self.area,
            'perimeter': self.perimeter,
            'circularity': self.circularity,
            'convexity': self.convexity,
            'aspect_ratio': self.aspect_ratio,
            'eccentricity': self.eccentricity,
            'orientation': self.orientation,
            'solidity': self.solidity,
            'extent': self.extent,
            'direction_consistency': self.direction_consistency,
            'spread_angle': self.spread_angle,
            'linearity': self.linearity,
            'skeleton_length': self.skeleton_length
        }


@dataclass
class RegionAnalysisResult:
    """区域分析结果"""
    region_id: int
    bbox: Tuple[int, int, int, int]  # (y_min, x_min, y_max, x_max)
    centroid: Tuple[float, float]     # (y, x)
    mask: np.ndarray                  # 区域掩码
    temp_features: TemperatureFeatures
    morph_features: MorphologyFeatures
    
    # 综合判定
    leak_score: float = 0.0           # 泄漏置信度 (0-1)
    classification: str = "unknown"   # leak / soil / equipment / unknown
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'region_id': self.region_id,
            'bbox': self.bbox,
            'centroid': self.centroid,
            'temp_features': self.temp_features.to_dict(),
            'morph_features': self.morph_features.to_dict(),
            'leak_score': self.leak_score,
            'classification': self.classification
        }


class RegionAnalyzer:
    """
    区域分析器
    
    对候选热异常区域进行温度和形态特征分析。
    """
    
    def __init__(
        self,
        erosion_kernel_size: int = 5,  # 腐蚀核大小（用于提取中心）
        edge_width: int = 3,           # 边缘宽度（像素）
        min_region_area: int = 50,     # 最小区域面积
    ):
        self.erosion_kernel_size = erosion_kernel_size
        self.edge_width = edge_width
        self.min_region_area = min_region_area
    
    def compute_temperature_features(
        self,
        temp_map: np.ndarray,
        mask: np.ndarray
    ) -> TemperatureFeatures:
        """
        计算区域的温度特征
        
        Args:
            temp_map: 温度矩阵
            mask: 区域掩码 (bool或0/1)
        """
        mask = mask.astype(bool)
        
        # 提取区域温度
        region_temps = temp_map[mask]
        
        if len(region_temps) == 0:
            logger.warning("空区域，返回默认特征")
            return TemperatureFeatures(
                center_temp=0, edge_temp=0, delta_t=0,
                center_variance=0, edge_variance=0, overall_variance=0,
                edge_gradient_mean=0, edge_gradient_std=0,
                temp_range=(0, 0)
            )
        
        # 整体统计
        overall_variance = np.var(region_temps)
        temp_range = (region_temps.min(), region_temps.max())
        
        # 中心区域（腐蚀后的区域）
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.erosion_kernel_size, self.erosion_kernel_size)
        )
        center_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=2)
        center_mask = center_mask.astype(bool)
        
        # 边缘区域（原区域 - 中心区域）
        edge_mask = mask & ~center_mask
        
        # 如果腐蚀后中心为空，使用整个区域的中心点
        if center_mask.sum() < 10:
            # 使用距离变换找中心
            dist_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            center_mask = dist_transform > dist_transform.max() * 0.5
            edge_mask = mask & ~center_mask
        
        # 计算中心和边缘温度
        if center_mask.sum() > 0:
            center_temps = temp_map[center_mask]
            center_temp = np.mean(center_temps)
            center_variance = np.var(center_temps)
        else:
            center_temp = np.mean(region_temps)
            center_variance = overall_variance
        
        if edge_mask.sum() > 0:
            edge_temps = temp_map[edge_mask]
            edge_temp = np.mean(edge_temps)
            edge_variance = np.var(edge_temps)
        else:
            edge_temp = center_temp
            edge_variance = center_variance
        
        delta_t = center_temp - edge_temp
        
        # 边缘梯度（温度突变程度）
        # 使用Sobel计算温度梯度
        grad_x = cv2.Sobel(temp_map.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(temp_map.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 提取边缘处的梯度
        # 使用形态学梯度找边界
        boundary = cv2.morphologyEx(
            mask.astype(np.uint8), 
            cv2.MORPH_GRADIENT, 
            np.ones((3, 3), np.uint8)
        )
        boundary_mask = boundary > 0
        
        if boundary_mask.sum() > 0:
            edge_gradients = gradient_magnitude[boundary_mask]
            edge_gradient_mean = np.mean(edge_gradients)
            edge_gradient_std = np.std(edge_gradients)
        else:
            edge_gradient_mean = 0
            edge_gradient_std = 0
        
        return TemperatureFeatures(
            center_temp=center_temp,
            edge_temp=edge_temp,
            delta_t=delta_t,
            center_variance=center_variance,
            edge_variance=edge_variance,
            overall_variance=overall_variance,
            edge_gradient_mean=edge_gradient_mean,
            edge_gradient_std=edge_gradient_std,
            temp_range=temp_range
        )
    
    def compute_morphology_features(
        self,
        mask: np.ndarray,
        temp_map: Optional[np.ndarray] = None
    ) -> MorphologyFeatures:
        """
        计算区域的形态特征
        
        Args:
            mask: 区域掩码
            temp_map: 温度矩阵（用于计算温度加权的方向）
        """
        mask = mask.astype(np.uint8)
        
        # 基本形状特征
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return self._default_morph_features()
        
        # 使用最大轮廓
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area < self.min_region_area or perimeter < 1:
            return self._default_morph_features()
        
        # 圆度
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # 凸包
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        solidity = convexity  # 同义
        
        # 拟合椭圆（需要至少5个点）
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center_x, center_y), (width, height), angle = ellipse
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1
            eccentricity = np.sqrt(1 - (min(width, height) / max(width, height))**2) if max(width, height) > 0 else 0
            orientation = np.radians(angle)
        else:
            # 使用边界框
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
            eccentricity = 0.5
            orientation = 0
        
        # 范围
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        extent = area / bbox_area if bbox_area > 0 else 0
        
        # 方向一致性（扇形/舌状检测）
        direction_consistency, spread_angle = self._compute_direction_features(
            mask, contour, temp_map
        )
        
        # 线性度（管道检测）
        linearity, skeleton_length = self._compute_linearity(mask)
        
        return MorphologyFeatures(
            area=int(area),
            perimeter=perimeter,
            circularity=circularity,
            convexity=convexity,
            aspect_ratio=aspect_ratio,
            eccentricity=eccentricity,
            orientation=orientation,
            solidity=solidity,
            extent=extent,
            direction_consistency=direction_consistency,
            spread_angle=spread_angle,
            linearity=linearity,
            skeleton_length=skeleton_length
        )
    
    def _compute_direction_features(
        self,
        mask: np.ndarray,
        contour: np.ndarray,
        temp_map: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        计算扇形/舌状的方向特征
        
        扇形特征：
        1. 有明确的扩散源点（通常是温度最高点）
        2. 从源点向外呈放射状扩散
        3. 边界点相对于源点的方向分布集中
        
        Returns:
            direction_consistency: 方向一致性 (0-1, 越高越像扇形)
            spread_angle: 扩散角度（度）
        """
        if len(contour) < 10:
            return 0.0, 360.0
        
        mask_bool = mask.astype(bool)
        
        # 找到扩散源点（温度最高点 或 重心）
        if temp_map is not None:
            # 使用温度最高点作为源点
            masked_temp = np.where(mask_bool, temp_map, -np.inf)
            source_y, source_x = np.unravel_index(np.argmax(masked_temp), masked_temp.shape)
        else:
            # 使用重心
            M = cv2.moments(mask)
            if M['m00'] > 0:
                source_x = M['m10'] / M['m00']
                source_y = M['m01'] / M['m00']
            else:
                return 0.0, 360.0
        
        # 计算轮廓点相对于源点的方向
        contour_points = contour.reshape(-1, 2)  # (N, 2) - [x, y]
        
        # 向量：源点 -> 轮廓点
        vectors = contour_points - np.array([source_x, source_y])
        
        # 计算角度
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])  # [-π, π]
        
        # 方向一致性：使用圆形方差
        # 如果点都在相似方向，方差小
        mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        
        # 计算角度偏差
        angle_diff = np.abs(angles - mean_angle)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)  # 处理环绕
        
        # 方向一致性 = 1 - 归一化方差
        angle_std = np.std(angle_diff)
        direction_consistency = max(0, 1 - angle_std / (np.pi / 2))
        
        # 扩散角度：包含90%点的角度范围
        sorted_angles = np.sort(angles)
        n = len(sorted_angles)
        p5, p95 = sorted_angles[int(n * 0.05)], sorted_angles[int(n * 0.95)]
        spread_angle = np.degrees(p95 - p5)
        
        # 处理跨越±π的情况
        if spread_angle < 0:
            spread_angle += 360
        
        return direction_consistency, spread_angle
    
    def _compute_linearity(self, mask: np.ndarray) -> Tuple[float, float]:
        """
        计算线性度（用于管道泄漏检测）
        
        线性度高 = 骨架长度 / 面积 高，且宽度均匀
        
        Returns:
            linearity: 线性度 (0-1)
            skeleton_length: 骨架长度
        """
        mask_bool = mask.astype(bool)
        
        # 骨架化
        skeleton = morphology.skeletonize(mask_bool)
        skeleton_length = np.sum(skeleton)
        
        # 计算面积
        area = np.sum(mask_bool)
        
        if area == 0:
            return 0.0, 0.0
        
        # 理想线条：宽度为w的线条，面积 = w * length
        # 估计平均宽度
        if skeleton_length > 0:
            avg_width = area / skeleton_length
        else:
            avg_width = np.sqrt(area)
        
        # 线性度 = 骨架长度² / 面积
        # 对于细长形状，这个值更大
        linearity = (skeleton_length ** 2) / area if area > 0 else 0
        
        # 归一化到0-1（经验值）
        linearity = min(1.0, linearity / 10.0)
        
        return linearity, float(skeleton_length)
    
    def _default_morph_features(self) -> MorphologyFeatures:
        """返回默认形态特征"""
        return MorphologyFeatures(
            area=0, perimeter=0, circularity=0, convexity=0,
            aspect_ratio=1, eccentricity=0, orientation=0,
            solidity=0, extent=0, direction_consistency=0,
            spread_angle=360, linearity=0, skeleton_length=0
        )
    
    def analyze_region(
        self,
        temp_map: np.ndarray,
        mask: np.ndarray,
        region_id: int = 0
    ) -> RegionAnalysisResult:
        """
        分析单个区域
        
        Args:
            temp_map: 温度矩阵
            mask: 区域掩码
            region_id: 区域ID
        
        Returns:
            RegionAnalysisResult
        """
        mask = mask.astype(bool)
        
        # 计算bbox和centroid
        coords = np.where(mask)
        if len(coords[0]) == 0:
            bbox = (0, 0, 0, 0)
            centroid = (0.0, 0.0)
        else:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            bbox = (y_min, x_min, y_max, x_max)
            centroid = (coords[0].mean(), coords[1].mean())
        
        # 计算特征
        temp_features = self.compute_temperature_features(temp_map, mask)
        morph_features = self.compute_morphology_features(mask, temp_map)
        
        return RegionAnalysisResult(
            region_id=region_id,
            bbox=bbox,
            centroid=centroid,
            mask=mask,
            temp_features=temp_features,
            morph_features=morph_features
        )
    
    def analyze_multiple_regions(
        self,
        temp_map: np.ndarray,
        label_map: np.ndarray
    ) -> List[RegionAnalysisResult]:
        """
        分析多个区域（从标签图）
        
        Args:
            temp_map: 温度矩阵
            label_map: 区域标签图（0=背景，1,2,3...=不同区域）
        
        Returns:
            List of RegionAnalysisResult
        """
        results = []
        unique_labels = np.unique(label_map)
        
        for label in unique_labels:
            if label == 0:  # 跳过背景
                continue
            
            mask = label_map == label
            if mask.sum() < self.min_region_area:
                continue
            
            result = self.analyze_region(temp_map, mask, region_id=int(label))
            results.append(result)
        
        return results


class AnnotationLoader:
    """
    JSON标注加载器
    
    支持常见的标注格式：
    - LabelMe格式
    - COCO格式
    - VIA格式
    """
    
    @staticmethod
    def load_labelme(json_path: str) -> List[Dict]:
        """
        加载LabelMe格式标注
        
        Returns:
            List of {'label': str, 'points': List[List[float]], 'mask': np.ndarray}
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        annotations = []
        image_height = data.get('imageHeight', 0)
        image_width = data.get('imageWidth', 0)
        
        for shape in data.get('shapes', []):
            label = shape.get('label', 'unknown')
            points = shape.get('points', [])
            shape_type = shape.get('shape_type', 'polygon')
            
            if shape_type == 'polygon' and len(points) >= 3:
                # 转换为掩码
                mask = np.zeros((image_height, image_width), dtype=np.uint8)
                pts = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 1)
                
                annotations.append({
                    'label': label,
                    'points': points,
                    'mask': mask.astype(bool)
                })
        
        return annotations
    
    @staticmethod
    def load_coco(json_path: str, image_id: int = None) -> List[Dict]:
        """
        加载COCO格式标注
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 构建类别映射
        categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
        
        # 构建图像尺寸映射
        images = {img['id']: (img['height'], img['width']) for img in data.get('images', [])}
        
        annotations = []
        for ann in data.get('annotations', []):
            if image_id is not None and ann['image_id'] != image_id:
                continue
            
            img_h, img_w = images.get(ann['image_id'], (0, 0))
            if img_h == 0 or img_w == 0:
                continue
            
            label = categories.get(ann['category_id'], 'unknown')
            
            # 处理分割标注
            segmentation = ann.get('segmentation', [])
            if isinstance(segmentation, list) and len(segmentation) > 0:
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                for seg in segmentation:
                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
                
                annotations.append({
                    'label': label,
                    'points': segmentation,
                    'mask': mask.astype(bool),
                    'bbox': ann.get('bbox', [])
                })
        
        return annotations
    
    @staticmethod
    def polygon_to_mask(
        points: List[List[float]], 
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        将多边形点转换为掩码
        
        Args:
            points: [[x1,y1], [x2,y2], ...]
            image_shape: (height, width)
        
        Returns:
            bool掩码
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
        return mask.astype(bool)


def extract_region_features(
    temp_map: np.ndarray,
    mask: np.ndarray
) -> Dict[str, float]:
    """
    便捷函数：提取区域特征
    
    Returns:
        特征字典
    """
    analyzer = RegionAnalyzer()
    result = analyzer.analyze_region(temp_map, mask)
    
    features = {}
    features.update(result.temp_features.to_dict())
    features.update(result.morph_features.to_dict())
    
    return features


# ============ 测试代码 ============
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("区域分析模块测试")
    print("=" * 60)
    
    # 创建模拟数据
    h, w = 200, 300
    
    # 1. 模拟泄漏区域（扇形，中心高温，边缘低温突变）
    print("\n1. 创建模拟泄漏区域...")
    leak_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 扇形：从点(150, 100)向右下扩散
    source = (100, 150)
    for angle in range(-30, 60):  # 90度扇形
        rad = np.radians(angle)
        for r in range(10, 80):
            x = int(source[1] + r * np.cos(rad))
            y = int(source[0] + r * np.sin(rad))
            if 0 <= x < w and 0 <= y < h:
                leak_mask[y, x] = 1
    
    # 形态学闭运算填充
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leak_mask = cv2.morphologyEx(leak_mask, cv2.MORPH_CLOSE, kernel)
    
    # 创建温度分布（中心热，边缘冷，边缘突变）
    leak_temp = np.zeros((h, w), dtype=np.float32)
    dist_from_source = np.sqrt(
        (np.arange(h)[:, None] - source[0])**2 + 
        (np.arange(w)[None, :] - source[1])**2
    )
    leak_temp = 10 - 0.3 * dist_from_source  # 中心10°C
    leak_temp = np.clip(leak_temp, -15, 10)
    leak_temp += np.random.normal(0, 0.5, (h, w))  # 添加噪声
    
    # 边缘突变
    leak_boundary = cv2.dilate(leak_mask, kernel) - leak_mask
    leak_temp[leak_boundary > 0] = -15  # 边缘突然降温
    
    # 2. 模拟土壤区域（随机形状，温度均匀）
    print("2. 创建模拟土壤区域...")
    soil_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(soil_mask, (220, 50), (30, 20), 45, 0, 360, 1, -1)
    
    soil_temp = np.zeros((h, w), dtype=np.float32)
    soil_temp[soil_mask > 0] = 5 + np.random.normal(0, 0.3, soil_mask.sum())  # 均匀温度
    
    # 合并温度图
    temp_map = np.full((h, w), -10.0, dtype=np.float32)  # 背景温度
    temp_map = np.where(leak_mask > 0, leak_temp, temp_map)
    temp_map = np.where(soil_mask > 0, soil_temp[soil_mask > 0].mean(), temp_map)
    
    # 3. 分析区域
    print("\n3. 分析区域特征...")
    analyzer = RegionAnalyzer()
    
    leak_result = analyzer.analyze_region(temp_map, leak_mask, region_id=1)
    soil_result = analyzer.analyze_region(temp_map, soil_mask, region_id=2)
    
    # 打印特征对比
    print("\n" + "-" * 60)
    print(f"{'特征':<25} {'泄漏区域':<15} {'土壤区域':<15}")
    print("-" * 60)
    
    # 温度特征
    print("【温度特征】")
    print(f"{'中心温度 (°C)':<25} {leak_result.temp_features.center_temp:<15.2f} {soil_result.temp_features.center_temp:<15.2f}")
    print(f"{'边缘温度 (°C)':<25} {leak_result.temp_features.edge_temp:<15.2f} {soil_result.temp_features.edge_temp:<15.2f}")
    print(f"{'中心-边缘温差 ΔT':<25} {leak_result.temp_features.delta_t:<15.2f} {soil_result.temp_features.delta_t:<15.2f}")
    print(f"{'边缘梯度强度':<25} {leak_result.temp_features.edge_gradient_mean:<15.2f} {soil_result.temp_features.edge_gradient_mean:<15.2f}")
    print(f"{'整体方差':<25} {leak_result.temp_features.overall_variance:<15.2f} {soil_result.temp_features.overall_variance:<15.2f}")
    
    # 形态特征
    print("\n【形态特征】")
    print(f"{'面积 (像素)':<25} {leak_result.morph_features.area:<15} {soil_result.morph_features.area:<15}")
    print(f"{'圆度':<25} {leak_result.morph_features.circularity:<15.3f} {soil_result.morph_features.circularity:<15.3f}")
    print(f"{'方向一致性':<25} {leak_result.morph_features.direction_consistency:<15.3f} {soil_result.morph_features.direction_consistency:<15.3f}")
    print(f"{'扩散角度 (度)':<25} {leak_result.morph_features.spread_angle:<15.1f} {soil_result.morph_features.spread_angle:<15.1f}")
    print(f"{'线性度':<25} {leak_result.morph_features.linearity:<15.3f} {soil_result.morph_features.linearity:<15.3f}")
    print("-" * 60)
    
    # 判别结论
    print("\n【判别结论】")
    print(f"泄漏区域: ΔT={leak_result.temp_features.delta_t:.1f}°C, 方向一致性={leak_result.morph_features.direction_consistency:.2f}")
    print(f"  → 中心热、边缘冷、有扩散方向 → 符合泄漏特征")
    print(f"土壤区域: ΔT={soil_result.temp_features.delta_t:.1f}°C, 方向一致性={soil_result.morph_features.direction_consistency:.2f}")
    print(f"  → 温度均匀、无扩散方向 → 符合土壤特征")
    
    # 4. 可视化
    print("\n4. 生成可视化...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 温度图
    im = axes[0, 0].imshow(temp_map, cmap='coolwarm', vmin=-15, vmax=10)
    axes[0, 0].set_title('温度分布')
    plt.colorbar(im, ax=axes[0, 0], label='°C')
    
    # 泄漏区域
    axes[0, 1].imshow(leak_mask, cmap='Reds')
    axes[0, 1].set_title('泄漏区域掩码')
    axes[0, 1].plot(source[1], source[0], 'g*', markersize=15, label='源点')
    axes[0, 1].legend()
    
    # 土壤区域
    axes[0, 2].imshow(soil_mask, cmap='Greens')
    axes[0, 2].set_title('土壤区域掩码')
    
    # 泄漏区域温度
    leak_temp_vis = np.ma.masked_where(leak_mask == 0, temp_map)
    im2 = axes[1, 0].imshow(leak_temp_vis, cmap='hot', vmin=-15, vmax=10)
    axes[1, 0].set_title(f'泄漏温度 (ΔT={leak_result.temp_features.delta_t:.1f}°C)')
    plt.colorbar(im2, ax=axes[1, 0], label='°C')
    
    # 土壤区域温度
    soil_temp_vis = np.ma.masked_where(soil_mask == 0, temp_map)
    im3 = axes[1, 1].imshow(soil_temp_vis, cmap='hot', vmin=-15, vmax=10)
    axes[1, 1].set_title(f'土壤温度 (ΔT={soil_result.temp_features.delta_t:.1f}°C)')
    plt.colorbar(im3, ax=axes[1, 1], label='°C')
    
    # 特征对比柱状图
    features = ['ΔT', '边缘梯度', '方向一致性', '圆度']
    leak_vals = [
        leak_result.temp_features.delta_t / 10,  # 归一化
        leak_result.temp_features.edge_gradient_mean / 5,
        leak_result.morph_features.direction_consistency,
        leak_result.morph_features.circularity
    ]
    soil_vals = [
        soil_result.temp_features.delta_t / 10,
        soil_result.temp_features.edge_gradient_mean / 5,
        soil_result.morph_features.direction_consistency,
        soil_result.morph_features.circularity
    ]
    
    x = np.arange(len(features))
    width = 0.35
    axes[1, 2].bar(x - width/2, leak_vals, width, label='泄漏', color='red', alpha=0.7)
    axes[1, 2].bar(x + width/2, soil_vals, width, label='土壤', color='green', alpha=0.7)
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(features)
    axes[1, 2].set_ylabel('归一化值')
    axes[1, 2].set_title('特征对比')
    axes[1, 2].legend()
    axes[1, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('/tmp/region_analysis_test.png', dpi=150)
    print("   保存到: /tmp/region_analysis_test.png")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
