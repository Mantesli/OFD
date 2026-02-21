"""
EMA Temporal Smoother
=====================
时序EMA平滑模块，用于视频流中的泄漏检测结果平滑。

核心原理：
- 真实泄漏：持续存在，连续多帧检测结果稳定
- 瞬时干扰（飞鸟、阴影变化）：仅出现1-2帧，被平滑后削弱

公式：
P_final(t) = α × P_leak(t) + (1-α) × P_final(t-1)

其中：
- α = 0.7 表示当前帧权重70%，历史权重30%
- α 越大，响应越快但平滑效果越弱
- α 越小，平滑效果越强但响应越慢
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import deque
from loguru import logger


@dataclass
class EMAConfig:
    """EMA配置"""
    # EMA系数：当前帧权重
    alpha: float = 0.7
    
    # 最小连续帧数：只有连续N帧超过阈值才确认检测
    min_consecutive_frames: int = 3
    
    # 检测阈值
    detection_threshold: float = 0.5
    
    # 是否启用空间平滑（对热力图）
    spatial_smoothing: bool = True
    spatial_kernel_size: int = 5


class EMASmoother:
    """
    EMA时序平滑器
    
    用于视频流中的逐帧检测结果平滑，过滤瞬时干扰。
    
    Usage:
        smoother = EMASmoother(alpha=0.7)
        
        for frame in video:
            raw_score = model.predict(frame)
            smoothed_score = smoother.update(raw_score)
            is_leak = smoother.is_confirmed_detection()
    """
    
    def __init__(self, config: Optional[EMAConfig] = None):
        self.config = config or EMAConfig()
        
        # 状态变量
        self.smoothed_value: Optional[float] = None
        self.smoothed_map: Optional[np.ndarray] = None  # 用于热力图平滑
        
        # 连续检测计数
        self.consecutive_detections = 0
        self.consecutive_non_detections = 0
        
        # 历史记录（用于分析）
        self.history: deque = deque(maxlen=100)
        
        # 帧计数
        self.frame_count = 0
    
    def reset(self):
        """重置状态（新视频开始时调用）"""
        self.smoothed_value = None
        self.smoothed_map = None
        self.consecutive_detections = 0
        self.consecutive_non_detections = 0
        self.history.clear()
        self.frame_count = 0
    
    def update(self, current_value: float) -> float:
        """
        更新EMA平滑值（标量版本）
        
        Args:
            current_value: 当前帧的检测分数 [0, 1]
            
        Returns:
            平滑后的分数
        """
        self.frame_count += 1
        
        if self.smoothed_value is None:
            # 第一帧，直接使用当前值
            self.smoothed_value = current_value
        else:
            # EMA更新
            self.smoothed_value = (
                self.config.alpha * current_value + 
                (1 - self.config.alpha) * self.smoothed_value
            )
        
        # 更新连续检测计数
        if self.smoothed_value >= self.config.detection_threshold:
            self.consecutive_detections += 1
            self.consecutive_non_detections = 0
        else:
            self.consecutive_non_detections += 1
            self.consecutive_detections = 0
        
        # 记录历史
        self.history.append({
            'frame': self.frame_count,
            'raw': current_value,
            'smoothed': self.smoothed_value,
            'consecutive': self.consecutive_detections
        })
        
        return self.smoothed_value
    
    def update_map(self, current_map: np.ndarray) -> np.ndarray:
        """
        更新EMA平滑热力图（二维版本）
        
        Args:
            current_map: 当前帧的检测热力图 (H, W)，值域 [0, 1]
            
        Returns:
            平滑后的热力图
        """
        self.frame_count += 1
        
        # 可选的空间平滑
        if self.config.spatial_smoothing:
            import cv2
            k = self.config.spatial_kernel_size
            current_map = cv2.GaussianBlur(current_map, (k, k), 0)
        
        if self.smoothed_map is None:
            self.smoothed_map = current_map.copy()
        else:
            # 确保尺寸匹配
            if self.smoothed_map.shape != current_map.shape:
                import cv2
                self.smoothed_map = cv2.resize(
                    self.smoothed_map, 
                    (current_map.shape[1], current_map.shape[0])
                )
            
            # EMA更新
            self.smoothed_map = (
                self.config.alpha * current_map + 
                (1 - self.config.alpha) * self.smoothed_map
            )
        
        return self.smoothed_map
    
    def is_confirmed_detection(self) -> bool:
        """
        判断是否为确认的检测（满足连续帧要求）
        
        Returns:
            是否确认为泄漏
        """
        return self.consecutive_detections >= self.config.min_consecutive_frames
    
    def get_confidence(self) -> float:
        """
        获取检测置信度
        
        综合考虑平滑值和连续帧数
        
        Returns:
            置信度 [0, 1]
        """
        if self.smoothed_value is None:
            return 0.0
        
        # 基础置信度：平滑后的分数
        base_confidence = self.smoothed_value
        
        # 连续性加成：连续检测越多，置信度越高
        continuity_bonus = min(
            self.consecutive_detections / self.config.min_consecutive_frames,
            1.0
        ) * 0.2  # 最多加成0.2
        
        return min(base_confidence + continuity_bonus, 1.0)
    
    def get_state(self) -> Dict:
        """获取当前状态"""
        return {
            'frame_count': self.frame_count,
            'smoothed_value': self.smoothed_value,
            'consecutive_detections': self.consecutive_detections,
            'is_confirmed': self.is_confirmed_detection(),
            'confidence': self.get_confidence()
        }
    
    def get_history_stats(self) -> Dict:
        """获取历史统计"""
        if not self.history:
            return {}
        
        raw_values = [h['raw'] for h in self.history]
        smoothed_values = [h['smoothed'] for h in self.history]
        
        return {
            'total_frames': len(self.history),
            'raw_mean': float(np.mean(raw_values)),
            'raw_std': float(np.std(raw_values)),
            'smoothed_mean': float(np.mean(smoothed_values)),
            'smoothed_std': float(np.std(smoothed_values)),
            'max_consecutive': max(h['consecutive'] for h in self.history)
        }


class MultiRegionEMASmoother:
    """
    多区域EMA平滑器
    
    当检测输出是多个候选区域时使用，为每个区域独立维护EMA状态。
    """
    
    def __init__(self, config: Optional[EMAConfig] = None):
        self.config = config or EMAConfig()
        self.region_smoothers: Dict[int, EMASmoother] = {}
        self.frame_count = 0
    
    def update(
        self, 
        regions: List[Dict]
    ) -> List[Dict]:
        """
        更新多区域检测结果
        
        Args:
            regions: 区域列表，每个区域包含 {'id': int, 'score': float, 'bbox': tuple}
            
        Returns:
            平滑后的区域列表
        """
        self.frame_count += 1
        
        smoothed_regions = []
        active_ids = set()
        
        for region in regions:
            region_id = region['id']
            active_ids.add(region_id)
            
            # 获取或创建该区域的平滑器
            if region_id not in self.region_smoothers:
                self.region_smoothers[region_id] = EMASmoother(self.config)
            
            smoother = self.region_smoothers[region_id]
            smoothed_score = smoother.update(region['score'])
            
            smoothed_regions.append({
                **region,
                'raw_score': region['score'],
                'smoothed_score': smoothed_score,
                'is_confirmed': smoother.is_confirmed_detection(),
                'confidence': smoother.get_confidence()
            })
        
        # 清理不再活跃的区域（可选：保留一段时间）
        # 这里简单处理：超过10帧未出现则移除
        inactive_ids = set(self.region_smoothers.keys()) - active_ids
        for region_id in inactive_ids:
            smoother = self.region_smoothers[region_id]
            if self.frame_count - smoother.frame_count > 10:
                del self.region_smoothers[region_id]
        
        return smoothed_regions


class VideoStreamProcessor:
    """
    视频流处理器
    
    整合检测模型和EMA平滑，提供端到端的视频处理接口。
    
    Usage:
        processor = VideoStreamProcessor(model, ema_config)
        
        for frame in video:
            result = processor.process_frame(frame)
            if result['is_leak']:
                alert(result)
    """
    
    def __init__(
        self,
        detection_model,  # 检测模型（需要有predict方法）
        ema_config: Optional[EMAConfig] = None,
        alert_callback=None
    ):
        self.model = detection_model
        self.smoother = EMASmoother(ema_config)
        self.alert_callback = alert_callback
        
        # 状态
        self.is_alerting = False
        self.alert_start_frame = None
    
    def process_frame(
        self, 
        rgb_frame: np.ndarray,
        ir_frame: Optional[np.ndarray] = None
    ) -> Dict:
        """
        处理单帧
        
        Args:
            rgb_frame: RGB图像
            ir_frame: 红外图像（可选）
            
        Returns:
            处理结果字典
        """
        # 模型推理
        if ir_frame is not None:
            raw_score = self.model.predict(rgb_frame, ir_frame)
        else:
            raw_score = self.model.predict(rgb_frame)
        
        # EMA平滑
        smoothed_score = self.smoother.update(raw_score)
        
        # 判断是否确认检测
        is_confirmed = self.smoother.is_confirmed_detection()
        
        # 警报状态管理
        if is_confirmed and not self.is_alerting:
            self.is_alerting = True
            self.alert_start_frame = self.smoother.frame_count
            if self.alert_callback:
                self.alert_callback('ALERT_START', self.smoother.get_state())
        elif not is_confirmed and self.is_alerting:
            # 需要连续多帧非检测才解除警报
            if self.smoother.consecutive_non_detections >= self.smoother.config.min_consecutive_frames:
                self.is_alerting = False
                if self.alert_callback:
                    self.alert_callback('ALERT_END', self.smoother.get_state())
        
        return {
            'frame': self.smoother.frame_count,
            'raw_score': raw_score,
            'smoothed_score': smoothed_score,
            'is_leak': is_confirmed,
            'confidence': self.smoother.get_confidence(),
            'is_alerting': self.is_alerting,
            'state': self.smoother.get_state()
        }
    
    def reset(self):
        """重置处理器状态"""
        self.smoother.reset()
        self.is_alerting = False
        self.alert_start_frame = None


# ============================================================================
# 便捷函数
# ============================================================================

def smooth_predictions(
    predictions: List[float],
    alpha: float = 0.7
) -> List[float]:
    """
    对预测序列进行EMA平滑
    
    Args:
        predictions: 原始预测分数列表
        alpha: EMA系数
        
    Returns:
        平滑后的预测列表
    """
    if not predictions:
        return []
    
    smoothed = [predictions[0]]
    for i in range(1, len(predictions)):
        smoothed_value = alpha * predictions[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(smoothed_value)
    
    return smoothed


def find_stable_detections(
    predictions: List[float],
    threshold: float = 0.5,
    min_consecutive: int = 3,
    alpha: float = 0.7
) -> List[Tuple[int, int]]:
    """
    找出稳定的检测区间
    
    Args:
        predictions: 预测分数列表
        threshold: 检测阈值
        min_consecutive: 最小连续帧数
        alpha: EMA系数
        
    Returns:
        检测区间列表 [(start, end), ...]
    """
    # 先平滑
    smoothed = smooth_predictions(predictions, alpha)
    
    # 找连续超过阈值的区间
    intervals = []
    start = None
    
    for i, score in enumerate(smoothed):
        if score >= threshold:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= min_consecutive:
                    intervals.append((start, i - 1))
                start = None
    
    # 处理最后一个区间
    if start is not None and len(smoothed) - start >= min_consecutive:
        intervals.append((start, len(smoothed) - 1))
    
    return intervals


if __name__ == "__main__":
    # 测试代码
    print("Testing EMA Smoother...")
    
    # 模拟检测序列：包含瞬时干扰和持续泄漏
    # 帧 0-10: 正常（低分）
    # 帧 11-13: 瞬时干扰（高分但只有3帧）
    # 帧 14-20: 正常
    # 帧 21-35: 持续泄漏（高分持续15帧）
    # 帧 36-50: 正常
    
    raw_predictions = (
        [0.1] * 10 +           # 正常
        [0.8, 0.9, 0.7] +      # 瞬时干扰（3帧）
        [0.1] * 7 +            # 正常
        [0.85] * 15 +          # 持续泄漏
        [0.1] * 15             # 正常
    )
    
    smoother = EMASmoother(EMAConfig(alpha=0.7, min_consecutive_frames=5))
    
    print("\nFrame | Raw   | Smoothed | Consecutive | Confirmed")
    print("-" * 55)
    
    for i, raw in enumerate(raw_predictions):
        smoothed = smoother.update(raw)
        state = smoother.get_state()
        confirmed = "YES" if state['is_confirmed'] else "NO"
        print(f"{i:5d} | {raw:.3f} | {smoothed:.3f}    | {state['consecutive_detections']:11d} | {confirmed}")
    
    print("\n" + "=" * 55)
    print("稳定检测区间:", find_stable_detections(raw_predictions, threshold=0.5, min_consecutive=5))
    print("\n分析：瞬时干扰（帧11-13）被过滤，持续泄漏（帧21-35）被保留")
