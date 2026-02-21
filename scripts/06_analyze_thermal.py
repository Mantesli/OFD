#!/usr/bin/env python
"""
Step 6: Thermal Analysis Validation
===================================
验证温度反演和泄漏判别的完整流程。

功能：
1. 加载红外图像和JSON标注
2. 温度反演
3. 提取标注区域的特征
4. 泄漏/干扰判别
5. 生成分析报告

Usage:
    # 分析单张图像
    python scripts/06_analyze_thermal.py \
        --image ./data/samples/001.jpg \
        --annotation ./data/annotations/001.json \
        --output_dir ./results/thermal_analysis

    # 批量分析
    python scripts/06_analyze_thermal.py \
        --image_dir ./data/images \
        --annotation_dir ./data/annotations \
        --output_dir ./results/thermal_analysis
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from loguru import logger

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.thermal_calibration import (
    ThermalCalibrator, 
    CalibrationConfig,
    DualModalSplitter
)
from src.features.region_analyzer import (
    RegionAnalyzer,
    AnnotationLoader
)
from src.features.leak_discriminator import (
    LeakDiscriminator,
    DiscriminationThresholds,
    AnomalyType
)


def parse_args():
    parser = argparse.ArgumentParser(description="Thermal analysis validation")
    
    # 输入
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--annotation", type=str, help="Single annotation path")
    parser.add_argument("--image_dir", type=str, help="Image directory for batch")
    parser.add_argument("--annotation_dir", type=str, help="Annotation directory for batch")
    
    # 输出
    parser.add_argument("--output_dir", type=str, default="./results/thermal_analysis")
    
    # 标定参数
    parser.add_argument("--t_min", type=float, default=-20.0, help="Min temperature")
    parser.add_argument("--t_max", type=float, default=15.0, help="Max temperature")
    
    # 图像布局
    parser.add_argument("--layout", type=str, default="horizontal", 
                       choices=["horizontal", "vertical", "ir_only"])
    parser.add_argument("--ir_position", type=str, default="left",
                       choices=["left", "right", "top", "bottom"])
    
    # 判别阈值
    parser.add_argument("--min_delta_t", type=float, default=3.0)
    parser.add_argument("--min_direction_consistency", type=float, default=0.4)
    
    # 标注格式
    parser.add_argument("--annotation_format", type=str, default="labelme",
                       choices=["labelme", "coco"])
    
    return parser.parse_args()


def load_image_and_annotation(
    image_path: str,
    annotation_path: str,
    annotation_format: str = "labelme"
) -> tuple:
    """加载图像和标注"""
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    
    # 加载标注
    if annotation_format == "labelme":
        annotations = AnnotationLoader.load_labelme(annotation_path)
    elif annotation_format == "coco":
        annotations = AnnotationLoader.load_coco(annotation_path)
    else:
        raise ValueError(f"不支持的标注格式: {annotation_format}")
    
    return image, annotations


def analyze_single_image(
    image: np.ndarray,
    annotations: list,
    calibrator: ThermalCalibrator,
    splitter: DualModalSplitter,
    analyzer: RegionAnalyzer,
    discriminator: LeakDiscriminator,
    layout: str = "horizontal"
) -> dict:
    """
    分析单张图像
    
    Returns:
        分析结果字典
    """
    results = {
        'regions': [],
        'summary': {}
    }
    
    # 1. 分割双模态图像（如果需要）
    if layout != "ir_only":
        ir_image, rgb_image = splitter.split(image)
    else:
        ir_image = image
        rgb_image = None
    
    # 2. 温度标定
    cal_result = calibrator.calibrate(ir_image)
    temp_map = cal_result.temperature_map
    
    results['calibration'] = {
        'r_min': float(cal_result.r_min),
        'r_max': float(cal_result.r_max),
        'alpha': float(cal_result.alpha),
        'beta': float(cal_result.beta),
        'temp_range': [float(temp_map.min()), float(temp_map.max())]
    }
    
    # 3. 分析每个标注区域
    leak_count = 0
    soil_count = 0
    other_count = 0
    
    for i, ann in enumerate(annotations):
        label = ann.get('label', 'unknown')
        mask = ann.get('mask')
        
        if mask is None:
            continue
        
        # 确保mask尺寸匹配
        if mask.shape != temp_map.shape:
            # 可能需要调整大小
            mask = cv2.resize(
                mask.astype(np.uint8), 
                (temp_map.shape[1], temp_map.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        
        # 分析区域
        region_result = analyzer.analyze_region(temp_map, mask, region_id=i)
        
        # 判别
        disc_result = discriminator.discriminate(region_result)
        
        # 统计
        if disc_result.anomaly_type in [AnomalyType.LEAK, AnomalyType.LEAK_POOL, AnomalyType.LEAK_PIPELINE]:
            leak_count += 1
        elif disc_result.anomaly_type == AnomalyType.SOIL:
            soil_count += 1
        else:
            other_count += 1
        
        # 记录结果
        results['regions'].append({
            'region_id': i,
            'label': label,
            'predicted_type': disc_result.anomaly_type.value,
            'confidence': float(disc_result.confidence),
            'leak_score': float(disc_result.leak_score),
            'temp_features': region_result.temp_features.to_dict(),
            'morph_features': region_result.morph_features.to_dict(),
            'reasons': disc_result.reasons
        })
    
    # 汇总
    results['summary'] = {
        'total_regions': len(annotations),
        'leak_count': leak_count,
        'soil_count': soil_count,
        'other_count': other_count,
        'has_leak': leak_count > 0
    }
    
    # 保存中间数据
    results['_temp_map'] = temp_map
    results['_ir_image'] = ir_image
    results['_rgb_image'] = rgb_image
    
    return results


def visualize_results(
    results: dict,
    annotations: list,
    output_path: str,
    image_name: str = ""
):
    """可视化分析结果"""
    temp_map = results['_temp_map']
    ir_image = results['_ir_image']
    rgb_image = results.get('_rgb_image')
    
    n_cols = 3 if rgb_image is not None else 2
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
    
    # 第一行：原始图像
    # IR图像
    axes[0, 0].imshow(cv2.cvtColor(ir_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('红外图像 (伪彩色)')
    axes[0, 0].axis('off')
    
    # 温度图
    im = axes[0, 1].imshow(temp_map, cmap='coolwarm', 
                          vmin=results['calibration']['temp_range'][0],
                          vmax=results['calibration']['temp_range'][1])
    axes[0, 1].set_title('温度反演')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], label='°C')
    
    # RGB图像（如果有）
    if rgb_image is not None:
        axes[0, 2].imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('可见光图像')
        axes[0, 2].axis('off')
    
    # 第二行：分析结果
    # 标注区域叠加
    overlay = ir_image.copy()
    for i, (ann, region) in enumerate(zip(annotations, results['regions'])):
        mask = ann.get('mask')
        if mask is None:
            continue
        
        # 确保mask尺寸匹配
        if mask.shape[:2] != overlay.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (overlay.shape[1], overlay.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # 根据判别结果选择颜色
        pred_type = region['predicted_type']
        if 'leak' in pred_type:
            color = (0, 0, 255)  # 红色 - 泄漏
        elif pred_type == 'soil':
            color = (0, 255, 0)  # 绿色 - 土壤
        else:
            color = (255, 255, 0)  # 青色 - 其他
        
        # 绘制轮廓
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 2)
        
        # 添加标签
        if len(contours) > 0:
            M = cv2.moments(contours[0])
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(overlay, f"{pred_type}", (cx-30, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    axes[1, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('检测结果\n红=泄漏, 绿=土壤, 青=其他')
    axes[1, 0].axis('off')
    
    # 特征统计
    if results['regions']:
        # 温度特征对比
        delta_ts = [r['temp_features']['delta_t'] for r in results['regions']]
        dir_cons = [r['morph_features']['direction_consistency'] for r in results['regions']]
        types = [r['predicted_type'] for r in results['regions']]
        
        colors = ['red' if 'leak' in t else 'green' if t == 'soil' else 'blue' 
                 for t in types]
        
        axes[1, 1].scatter(delta_ts, dir_cons, c=colors, s=100, alpha=0.7)
        axes[1, 1].axhline(y=0.4, color='gray', linestyle='--', alpha=0.5, label='方向阈值')
        axes[1, 1].axvline(x=3.0, color='gray', linestyle='--', alpha=0.5, label='温差阈值')
        axes[1, 1].set_xlabel('中心-边缘温差 ΔT (°C)')
        axes[1, 1].set_ylabel('方向一致性')
        axes[1, 1].set_title('特征分布')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 汇总信息
    summary = results['summary']
    summary_text = (
        f"分析汇总\n"
        f"{'='*30}\n"
        f"总区域数: {summary['total_regions']}\n"
        f"泄漏: {summary['leak_count']}\n"
        f"土壤: {summary['soil_count']}\n"
        f"其他: {summary['other_count']}\n"
        f"{'='*30}\n"
        f"判定: {'有泄漏!' if summary['has_leak'] else '无泄漏'}"
    )
    
    if n_cols > 2:
        ax_text = axes[1, 2]
    else:
        ax_text = axes[1, 1]
    
    axes[1, -1].text(0.5, 0.5, summary_text, 
                    transform=axes[1, -1].transAxes,
                    fontsize=12, verticalalignment='center',
                    horizontalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, -1].axis('off')
    axes[1, -1].set_title('分析汇总')
    
    plt.suptitle(f'热分析结果: {image_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"可视化保存到: {output_path}")


def main():
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化组件
    cal_config = CalibrationConfig(t_min=args.t_min, t_max=args.t_max)
    calibrator = ThermalCalibrator(cal_config)
    
    splitter = DualModalSplitter(
        layout=args.layout if args.layout != "ir_only" else "horizontal",
        ir_position=args.ir_position
    )
    
    analyzer = RegionAnalyzer()
    
    disc_thresholds = DiscriminationThresholds(
        min_delta_t=args.min_delta_t,
        min_direction_consistency=args.min_direction_consistency
    )
    discriminator = LeakDiscriminator(disc_thresholds)
    
    # 确定处理模式
    if args.image:
        # 单张图像模式
        image_paths = [Path(args.image)]
        annotation_paths = [Path(args.annotation)] if args.annotation else [None]
    elif args.image_dir:
        # 批量模式
        image_dir = Path(args.image_dir)
        annotation_dir = Path(args.annotation_dir) if args.annotation_dir else None
        
        image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
        
        if annotation_dir:
            annotation_paths = []
            for img_path in image_paths:
                ann_path = annotation_dir / f"{img_path.stem}.json"
                annotation_paths.append(ann_path if ann_path.exists() else None)
        else:
            annotation_paths = [None] * len(image_paths)
    else:
        logger.error("请指定 --image 或 --image_dir")
        return
    
    # 处理图像
    all_results = []
    
    for img_path, ann_path in zip(image_paths, annotation_paths):
        logger.info(f"处理: {img_path}")
        
        try:
            # 加载图像
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"无法读取图像: {img_path}")
                continue
            
            # 加载标注
            if ann_path and ann_path.exists():
                annotations = AnnotationLoader.load_labelme(str(ann_path))
            else:
                # 无标注时，使用整图作为区域
                logger.warning(f"无标注文件: {ann_path}, 使用整图分析")
                h, w = image.shape[:2]
                if args.layout != "ir_only":
                    w = w // 2
                annotations = [{
                    'label': 'full_image',
                    'mask': np.ones((h, w), dtype=bool)
                }]
            
            # 分析
            results = analyze_single_image(
                image=image,
                annotations=annotations,
                calibrator=calibrator,
                splitter=splitter,
                analyzer=analyzer,
                discriminator=discriminator,
                layout=args.layout
            )
            
            # 可视化
            vis_path = output_dir / f"{img_path.stem}_analysis.png"
            visualize_results(results, annotations, str(vis_path), img_path.name)
            
            # 保存JSON结果
            json_result = {
                'image': str(img_path),
                'annotation': str(ann_path) if ann_path else None,
                'calibration': results['calibration'],
                'summary': results['summary'],
                'regions': results['regions']
            }
            
            json_path = output_dir / f"{img_path.stem}_result.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_result, f, indent=2, ensure_ascii=False)
            
            all_results.append(json_result)
            
            # 打印结果
            print(f"\n{'='*50}")
            print(f"图像: {img_path.name}")
            print(f"{'='*50}")
            print(f"区域数: {results['summary']['total_regions']}")
            print(f"泄漏数: {results['summary']['leak_count']}")
            print(f"判定: {'⚠️ 检测到泄漏!' if results['summary']['has_leak'] else '✓ 无泄漏'}")
            
            for region in results['regions']:
                print(f"\n  区域 {region['region_id']} ({region['label']}):")
                print(f"    预测类型: {region['predicted_type']}")
                print(f"    置信度: {region['confidence']:.2f}")
                print(f"    ΔT: {region['temp_features']['delta_t']:.1f}°C")
                print(f"    方向一致性: {region['morph_features']['direction_consistency']:.2f}")
        
        except Exception as e:
            logger.error(f"处理 {img_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存汇总报告
    if len(all_results) > 1:
        summary_report = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(all_results),
            'images_with_leak': sum(1 for r in all_results if r['summary']['has_leak']),
            'total_regions': sum(r['summary']['total_regions'] for r in all_results),
            'total_leaks': sum(r['summary']['leak_count'] for r in all_results),
            'results': all_results
        }
        
        with open(output_dir / 'summary_report.json', 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*50}")
        print("批量处理完成")
        print(f"{'='*50}")
        print(f"总图像数: {summary_report['total_images']}")
        print(f"有泄漏图像: {summary_report['images_with_leak']}")
        print(f"总区域数: {summary_report['total_regions']}")
        print(f"总泄漏数: {summary_report['total_leaks']}")
        print(f"结果保存到: {output_dir}")


if __name__ == "__main__":
    main()
