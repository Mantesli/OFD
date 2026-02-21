#!/usr/bin/env python
"""
Step 6: Visualize Test Set (Smart Matching Version)
===================================================
专门用于测试集的可视化分析脚本。
它能自动处理图片文件名(xxx.jpg)与标注文件名(xxx_ir.json)不一致的问题。

Usage:
    python scripts/06_visualize_test.py
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.thermal_calibration import ThermalCalibrator, CalibrationConfig, DualModalSplitter
from src.features.region_analyzer import RegionAnalyzer, AnnotationLoader
from src.features.leak_discriminator import LeakDiscriminator, DiscriminationThresholds

# === 配置路径 ===
IMAGE_DIR = project_root / "data" / "sampled" / "test" / "normal"
ANNOTATION_DIR = project_root / "data" / "annotations"
OUTPUT_DIR = project_root / "results" / "thermal_analysis_test2"


def find_annotation(img_path: Path, anno_dir: Path) -> Path:
    """智能查找标注文件（处理 _ir 后缀）"""
    stem = img_path.stem

    # 尝试1: 完全匹配 (xxx.json)
    cand1 = anno_dir / f"{stem}.json"
    if cand1.exists(): return cand1

    # 尝试2: 加 _ir 后缀 (xxx_ir.json)
    cand2 = anno_dir / f"{stem}_ir.json"
    if cand2.exists(): return cand2

    # 尝试3: 如果图片名里已经有 _copy 之类的后缀，尝试去除
    # (针对之前数据增强产生的文件名)
    base_name = stem.split('_copy')[0].split('_2')[0].split(' -')[0]
    cand3 = anno_dir / f"{base_name}_ir.json"
    if cand3.exists(): return cand3

    return None


def analyze_and_draw(img_path, json_path, output_dir, calibrator, splitter, analyzer, discriminator):
    # 1. 读取图像
    # 处理 Windows 中文路径问题
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ 无法读取: {img_path.name}")
        return

    # 2. 读取标注
    try:
        annotations = AnnotationLoader.load_labelme(str(json_path))
    except Exception as e:
        print(f"⚠️ 标注读取失败 {json_path.name}: {e}")
        return

    # 3. 分割 RGB/IR
    # 假设是拼接图，根据您的设置调整 (horizontal/left)
    ir_img, rgb_img = splitter.split(img)

    # 4. 温度反演
    cal_res = calibrator.calibrate(ir_img)
    temp_map = cal_res.temperature_map

    # 5. 可视化画布
    plt.figure(figsize=(15, 5))

    # 子图1: 红外原图 + 标注
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB))
    plt.title(f"IR Source: {img_path.name}")
    plt.axis('off')

    # 子图2: 温度热力图
    plt.subplot(1, 3, 2)
    im = plt.imshow(temp_map, cmap='inferno')
    plt.colorbar(im, label='Temperature (°C)')
    plt.title("Thermal Analysis")
    plt.axis('off')

    # 子图3: 检测结果
    overlay = ir_img.copy()

    for i, ann in enumerate(annotations):
        mask = ann.get('mask')
        if mask is None: continue

        # 调整mask大小
        if mask.shape != temp_map.shape:
            mask = cv2.resize(mask.astype(np.uint8), (temp_map.shape[1], temp_map.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

        # 分析区域
        res = analyzer.analyze_region(temp_map, mask)
        disc = discriminator.discriminate(res)

        # 绘图颜色
        if 'leak' in disc.anomaly_type.value:
            color = (0, 0, 255)  # 红
            label = f"LEAK ({disc.confidence:.2f})"
        else:
            color = (0, 255, 0)  # 绿
            label = "Normal"

        # 画轮廓
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

        # 画标签
        if contours:
            c = contours[0]
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(overlay, label, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(overlay, label, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Detection Result")
    plt.axis('off')

    # 保存
    out_path = output_dir / f"{img_path.stem}_vis.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def main():
    if not IMAGE_DIR.exists():
        print(f"❌ 图片目录不存在: {IMAGE_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 初始化工具链
    calibrator = ThermalCalibrator(CalibrationConfig(t_min=-20, t_max=15))
    splitter = DualModalSplitter(layout="horizontal")  # 假设是左右拼接
    analyzer = RegionAnalyzer()
    discriminator = LeakDiscriminator(DiscriminationThresholds(min_delta_t=3.0))

    images = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png"))
    print(f"🚀 开始分析 {len(images)} 张测试集图片...")

    success_count = 0
    for img_path in tqdm(images):
        # 1. 找标注
        json_path = find_annotation(img_path, ANNOTATION_DIR)

        if not json_path:
            # 如果是 _copy 的图片可能没有对应 json，尝试找原图 json
            # print(f"⚠️ 跳过（无标注）: {img_path.name}")
            continue

        # 2. 执行分析
        analyze_and_draw(img_path, json_path, OUTPUT_DIR, calibrator, splitter, analyzer, discriminator)
        success_count += 1

    print(f"\n✅ 分析完成！成功生成 {success_count} 张可视化图。")
    print(f"📂 结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()