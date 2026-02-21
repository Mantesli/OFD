"""
Thermal Calibration Module
==========================
伪彩色红外图像的温度反演与标定。

核心功能：
1. 逐帧自适应温度标定（R通道 → 温度）
2. 温度矩阵生成
3. 标定参数管理

物理背景：
- DJI红外相机输出伪彩色图像，红色通道值与温度正相关
- 由于不同帧的R值范围不稳定，需要逐帧自适应标定
- 标定公式：T = T_min + (R - R_min) / (R_max - R_min) × (T_max - T_min)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Union, Dict
from pathlib import Path
from loguru import logger


@dataclass
class CalibrationConfig:
    """温度标定配置"""
    # 温度范围（冬季油田环境）
    t_min: float = -20.0  # 最低温度 °C
    t_max: float = 15.0   # 最高温度 °C
    
    # R通道处理
    use_percentile: bool = True  # 使用百分位数而非极值
    percentile_low: float = 1.0  # 低百分位（去除噪声）
    percentile_high: float = 99.0  # 高百分位
    
    # 自动裁剪黑边
    auto_crop: bool = True
    black_threshold: int = 10  # 黑色像素阈值


@dataclass 
class CalibrationResult:
    """标定结果"""
    temperature_map: np.ndarray  # 温度矩阵
    r_min: float  # R通道最小值
    r_max: float  # R通道最大值
    alpha: float  # 线性系数 (斜率)
    beta: float   # 线性系数 (截距)
    t_range: Tuple[float, float]  # 温度范围
    
    def temperature_at(self, r_value: float) -> float:
        """根据R值计算温度"""
        return self.alpha * r_value + self.beta
    
    def r_value_at(self, temperature: float) -> float:
        """根据温度反算R值"""
        return (temperature - self.beta) / self.alpha


class ThermalCalibrator:
    """
    温度标定器
    
    将伪彩色红外图像转换为温度矩阵。
    
    使用方法：
    ```python
    calibrator = ThermalCalibrator()
    result = calibrator.calibrate(ir_image)
    temp_map = result.temperature_map
    ```
    """
    
    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
    
    def auto_crop_content(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        自动裁剪图像中非黑边的有效内容区域
        
        Returns:
            cropped_image: 裁剪后的图像
            bbox: (y_min, y_max, x_min, x_max) 裁剪区域
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        mask = gray > self.config.black_threshold
        coords = np.where(mask)
        
        if coords[0].size == 0:
            logger.warning("图像中无有效内容，返回原图")
            h, w = image.shape[:2]
            return image, (0, h, 0, w)
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        cropped = image[y_min:y_max + 1, x_min:x_max + 1]
        return cropped, (y_min, y_max + 1, x_min, x_max + 1)
    
    def extract_red_channel(self, image: np.ndarray) -> np.ndarray:
        """
        提取红色通道
        
        注意：OpenCV默认BGR格式，红色通道在索引2
        """
        if len(image.shape) == 2:
            # 已经是单通道
            return image.astype(np.float32)
        
        # BGR -> R
        red_channel = image[:, :, 2].astype(np.float32)
        return red_channel
    
    def compute_calibration_params(
        self, 
        red_channel: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        计算标定参数
        
        Returns:
            r_min, r_max, alpha, beta
        """
        if self.config.use_percentile:
            r_min = np.percentile(red_channel, self.config.percentile_low)
            r_max = np.percentile(red_channel, self.config.percentile_high)
        else:
            r_min = red_channel.min()
            r_max = red_channel.max()
        
        # 防止除零
        if abs(r_max - r_min) < 1e-6:
            logger.warning(f"R值范围过小: [{r_min}, {r_max}]，使用默认范围")
            r_min, r_max = 0, 255
        
        # 线性标定: T = alpha * R + beta
        # T_min = alpha * R_min + beta
        # T_max = alpha * R_max + beta
        t_min, t_max = self.config.t_min, self.config.t_max
        alpha = (t_max - t_min) / (r_max - r_min)
        beta = t_min - alpha * r_min
        
        return r_min, r_max, alpha, beta
    
    def calibrate(
        self, 
        image: np.ndarray,
        crop: bool = None
    ) -> CalibrationResult:
        """
        执行温度标定
        
        Args:
            image: 输入图像 (BGR格式或灰度)
            crop: 是否裁剪黑边，None表示使用配置
        
        Returns:
            CalibrationResult: 包含温度矩阵和标定参数
        """
        # 是否裁剪
        if crop is None:
            crop = self.config.auto_crop
        
        if crop:
            image, bbox = self.auto_crop_content(image)
        
        # 提取红色通道
        red_channel = self.extract_red_channel(image)
        
        # 计算标定参数
        r_min, r_max, alpha, beta = self.compute_calibration_params(red_channel)
        
        # 温度反演
        temperature_map = alpha * red_channel + beta
        
        # 裁剪到有效温度范围（可选）
        temperature_map = np.clip(
            temperature_map, 
            self.config.t_min, 
            self.config.t_max
        )
        
        logger.debug(
            f"标定完成: R=[{r_min:.1f}, {r_max:.1f}] → "
            f"T=[{self.config.t_min}°C, {self.config.t_max}°C], "
            f"α={alpha:.4f}, β={beta:.2f}"
        )
        
        return CalibrationResult(
            temperature_map=temperature_map,
            r_min=r_min,
            r_max=r_max,
            alpha=alpha,
            beta=beta,
            t_range=(self.config.t_min, self.config.t_max)
        )
    
    def calibrate_with_references(
        self,
        image: np.ndarray,
        ref_points: list,  # [(x1, y1, t1), (x2, y2, t2)]
    ) -> CalibrationResult:
        """
        使用参考点进行标定（交互式标定）
        
        当知道图像中某些点的真实温度时使用此方法。
        
        Args:
            image: 输入图像
            ref_points: 参考点列表 [(x, y, temperature), ...]
        
        Returns:
            CalibrationResult
        """
        if len(ref_points) < 2:
            raise ValueError("至少需要2个参考点进行标定")
        
        red_channel = self.extract_red_channel(image)
        
        # 提取参考点的R值和温度
        r_values = []
        t_values = []
        for x, y, t in ref_points:
            r_values.append(red_channel[y, x])
            t_values.append(t)
        
        r_values = np.array(r_values)
        t_values = np.array(t_values)
        
        # 最小二乘拟合
        # T = alpha * R + beta
        A = np.vstack([r_values, np.ones(len(r_values))]).T
        alpha, beta = np.linalg.lstsq(A, t_values, rcond=None)[0]
        
        # 计算R范围
        r_min, r_max = red_channel.min(), red_channel.max()
        
        # 温度反演
        temperature_map = alpha * red_channel + beta
        
        logger.info(
            f"参考点标定完成: α={alpha:.4f}, β={beta:.2f}, "
            f"T范围=[{temperature_map.min():.1f}°C, {temperature_map.max():.1f}°C]"
        )
        
        return CalibrationResult(
            temperature_map=temperature_map,
            r_min=r_min,
            r_max=r_max,
            alpha=alpha,
            beta=beta,
            t_range=(temperature_map.min(), temperature_map.max())
        )


class DualModalSplitter:
    """
    双模态图像分割器
    
    处理DJI Matrice 300 RTK等设备的左右拼接图像：
    - 左半部分：红外图像
    - 右半部分：可见光图像
    """
    
    def __init__(
        self, 
        layout: str = "horizontal",  # horizontal: 左右拼接, vertical: 上下拼接
        ir_position: str = "left"    # left/right 或 top/bottom
    ):
        self.layout = layout
        self.ir_position = ir_position
    
    def split(
        self, 
        image: np.ndarray,
        auto_crop: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        分割双模态图像
        
        Returns:
            ir_image: 红外图像
            rgb_image: 可见光图像
        """
        # 自动裁剪黑边
        if auto_crop:
            calibrator = ThermalCalibrator()
            image, _ = calibrator.auto_crop_content(image)
        
        h, w = image.shape[:2]
        
        if self.layout == "horizontal":
            mid = w // 2
            left = image[:, :mid]
            right = image[:, mid:]
            
            if self.ir_position == "left":
                return left, right
            else:
                return right, left
        else:
            mid = h // 2
            top = image[:mid, :]
            bottom = image[mid:, :]
            
            if self.ir_position == "top":
                return top, bottom
            else:
                return bottom, top


def calibrate_temperature(
    image: np.ndarray,
    t_min: float = -20.0,
    t_max: float = 15.0
) -> np.ndarray:
    """
    便捷函数：温度标定
    
    Args:
        image: 输入图像
        t_min: 最低温度
        t_max: 最高温度
    
    Returns:
        温度矩阵
    """
    config = CalibrationConfig(t_min=t_min, t_max=t_max)
    calibrator = ThermalCalibrator(config)
    result = calibrator.calibrate(image)
    return result.temperature_map


def split_dual_modal(
    image: np.ndarray,
    ir_position: str = "left"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    便捷函数：分割双模态图像
    
    Returns:
        (ir_image, rgb_image)
    """
    splitter = DualModalSplitter(ir_position=ir_position)
    return splitter.split(image)


# ============ 测试代码 ============
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("温度标定模块测试")
    print("=" * 60)
    
    # 创建模拟的伪彩色红外图像
    # 模拟一个泄漏场景：中心高温，边缘低温
    h, w = 200, 300
    
    # 创建温度分布（中心热，边缘冷）
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # 模拟温度：中心10°C，边缘-15°C
    true_temp = 10 - 25 * (distance / distance.max())
    true_temp += np.random.normal(0, 1, (h, w))  # 添加噪声
    
    # 模拟R通道值（非线性映射，模拟真实情况）
    r_channel = ((true_temp + 20) / 35 * 200 + 30).astype(np.uint8)
    r_channel = np.clip(r_channel, 0, 255)
    
    # 创建伪彩色图像 (BGR)
    fake_ir = np.zeros((h, w, 3), dtype=np.uint8)
    fake_ir[:, :, 2] = r_channel  # R通道
    fake_ir[:, :, 1] = (r_channel * 0.3).astype(np.uint8)  # G通道
    fake_ir[:, :, 0] = (255 - r_channel) * 0.5  # B通道
    
    # 测试标定
    print("\n1. 测试自适应标定...")
    calibrator = ThermalCalibrator()
    result = calibrator.calibrate(fake_ir)
    
    print(f"   R范围: [{result.r_min:.1f}, {result.r_max:.1f}]")
    print(f"   标定参数: α={result.alpha:.4f}, β={result.beta:.2f}")
    print(f"   温度范围: [{result.temperature_map.min():.1f}°C, {result.temperature_map.max():.1f}°C]")
    
    # 验证标定精度
    temp_error = np.abs(result.temperature_map - true_temp)
    print(f"   平均标定误差: {temp_error.mean():.2f}°C")
    
    # 测试双模态分割
    print("\n2. 测试双模态分割...")
    # 创建左右拼接图像
    fake_rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    dual_image = np.hstack([fake_ir, fake_rgb])
    
    splitter = DualModalSplitter()
    ir_split, rgb_split = splitter.split(dual_image, auto_crop=False)
    
    print(f"   原始尺寸: {dual_image.shape}")
    print(f"   IR尺寸: {ir_split.shape}")
    print(f"   RGB尺寸: {rgb_split.shape}")
    
    # 可视化
    print("\n3. 生成可视化...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始伪彩色
    axes[0, 0].imshow(cv2.cvtColor(fake_ir, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("输入：伪彩色红外图像")
    axes[0, 0].axis('off')
    
    # R通道
    axes[0, 1].imshow(r_channel, cmap='hot')
    axes[0, 1].set_title("R通道提取")
    axes[0, 1].axis('off')
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], label='R value')
    
    # 真实温度
    im1 = axes[0, 2].imshow(true_temp, cmap='coolwarm', vmin=-20, vmax=15)
    axes[0, 2].set_title("真实温度分布")
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], label='°C')
    
    # 标定温度
    im2 = axes[1, 0].imshow(result.temperature_map, cmap='coolwarm', vmin=-20, vmax=15)
    axes[1, 0].set_title("标定温度分布")
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], label='°C')
    
    # 误差图
    im3 = axes[1, 1].imshow(temp_error, cmap='Reds', vmin=0, vmax=5)
    axes[1, 1].set_title(f"标定误差 (平均={temp_error.mean():.2f}°C)")
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], label='°C')
    
    # 温度直方图
    axes[1, 2].hist(true_temp.flatten(), bins=50, alpha=0.5, label='真实', color='blue')
    axes[1, 2].hist(result.temperature_map.flatten(), bins=50, alpha=0.5, label='标定', color='red')
    axes[1, 2].set_xlabel('温度 (°C)')
    axes[1, 2].set_ylabel('像素数')
    axes[1, 2].set_title('温度分布直方图')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('/tmp/thermal_calibration_test.png', dpi=150)
    print("   保存到: /tmp/thermal_calibration_test.png")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
