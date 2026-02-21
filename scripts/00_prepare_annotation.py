#!/usr/bin/env python
"""
Step 0: Prepare Images for Annotation
=====================================
预处理双模态图像，分割出IR和RGB部分用于标注。

功能：
1. 自动裁剪黑边
2. 分割左右拼接的IR/RGB图像
3. 保存分割后的图像用于标注

Usage:
    python scripts/00_prepare_annotation.py \
        --input_dir ./data/raw \
        --output_dir ./data/for_annotation \
        --layout horizontal \
        --ir_position left
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def auto_crop_black_border(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """自动裁剪黑边"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    
    coords = np.where(mask)
    if coords[0].size == 0:
        return image
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    return image[y_min:y_max + 1, x_min:x_max + 1]


def split_dual_modal(
    image: np.ndarray, 
    layout: str = "horizontal",
    ir_position: str = "left"
) -> tuple:
    """
    分割双模态图像
    
    Returns:
        (ir_image, rgb_image)
    """
    h, w = image.shape[:2]
    
    if layout == "horizontal":
        mid = w // 2
        left = image[:, :mid]
        right = image[:, mid:]
        
        if ir_position == "left":
            return left, right
        else:
            return right, left
    else:  # vertical
        mid = h // 2
        top = image[:mid, :]
        bottom = image[mid:, :]
        
        if ir_position == "top":
            return top, bottom
        else:
            return bottom, top


def process_images(
    input_dir: Path,
    output_dir: Path,
    layout: str = "horizontal",
    ir_position: str = "left",
    save_rgb: bool = False
):
    """处理所有图像"""
    
    # 创建输出目录
    ir_dir = output_dir / "ir_images"
    ir_dir.mkdir(parents=True, exist_ok=True)
    
    if save_rgb:
        rgb_dir = output_dir / "rgb_images"
        rgb_dir.mkdir(parents=True, exist_ok=True)
    
    # 原始图像备份目录
    original_dir = output_dir / "original"
    original_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像
    image_files = (
        list(input_dir.glob("*.jpg")) + 
        list(input_dir.glob("*.jpeg")) + 
        list(input_dir.glob("*.png"))
    )
    
    print(f"找到 {len(image_files)} 张图像")
    
    stats = {
        'success': 0,
        'failed': 0,
        'skipped': 0
    }
    
    for img_path in tqdm(image_files, desc="处理图像"):
        try:
            # 读取图像
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"⚠️ 无法读取: {img_path}")
                stats['failed'] += 1
                continue
            
            # 自动裁剪黑边
            cropped = auto_crop_black_border(image)
            
            # 检查是否是双模态图像（宽度应该是高度的约2倍）
            h, w = cropped.shape[:2]
            aspect_ratio = w / h
            
            if aspect_ratio < 1.5:
                # 可能已经是单模态图像，直接复制
                print(f"⚠️ 图像比例异常 ({aspect_ratio:.2f})，可能已是单模态: {img_path}")
                cv2.imwrite(str(ir_dir / img_path.name), cropped)
                stats['skipped'] += 1
                continue
            
            # 分割
            ir_image, rgb_image = split_dual_modal(cropped, layout, ir_position)
            
            # 保存IR图像（用于标注）
            ir_filename = f"{img_path.stem}_ir{img_path.suffix}"
            cv2.imwrite(str(ir_dir / ir_filename), ir_image)
            
            # 保存RGB图像（可选）
            if save_rgb:
                rgb_filename = f"{img_path.stem}_rgb{img_path.suffix}"
                cv2.imwrite(str(rgb_dir / rgb_filename), rgb_image)
            
            # 复制原始图像
            cv2.imwrite(str(original_dir / img_path.name), image)
            
            stats['success'] += 1
            
        except Exception as e:
            print(f"❌ 处理失败 {img_path}: {e}")
            stats['failed'] += 1
    
    return stats


def create_annotation_template(output_dir: Path):
    """创建标注模板和说明"""
    
    # 创建annotations目录
    ann_dir = output_dir / "annotations"
    ann_dir.mkdir(exist_ok=True)
    
    # 创建说明文件
    readme_content = """# 标注说明

## 目录结构

```
for_annotation/
├── ir_images/          # IR图像（在此目录中标注）
│   ├── 001_ir.jpg
│   └── ...
├── rgb_images/         # RGB图像（参考用）
│   └── ...
├── annotations/        # 标注文件保存位置
│   ├── 001_ir.json
│   └── ...
└── original/           # 原始拼接图像
```

## 标注步骤

1. 安装LabelMe: `pip install labelme`

2. 启动LabelMe:
   ```bash
   labelme ir_images/ --output annotations/
   ```

3. 标注类别:
   - `leak` - 泄漏（扇形/舌状）
   - `leak_pipeline` - 管道泄漏（线性）
   - `soil` - 黑色土壤（干扰）
   - `equipment` - 设备（干扰）
   - `tire_track` - 轮胎痕迹（干扰）
   - `other` - 其他

4. 保存标注到 annotations/ 目录

## 标注要点

- 仅标注IR图像
- 使用多边形勾勒热异常边界
- 泄漏特征：中心热、边缘突变、扇形/舌状
- 土壤特征：温度均匀、形状随机
"""
    
    with open(output_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✓ 已创建标注说明: {output_dir / 'README.md'}")


def main():
    parser = argparse.ArgumentParser(description="准备标注图像")
    parser.add_argument("--input_dir", type=str, required=True, help="原始图像目录")
    parser.add_argument("--output_dir", type=str, default="./data/for_annotation")
    parser.add_argument("--layout", type=str, default="horizontal", 
                       choices=["horizontal", "vertical"])
    parser.add_argument("--ir_position", type=str, default="left",
                       choices=["left", "right", "top", "bottom"])
    parser.add_argument("--save_rgb", action="store_true", help="是否保存RGB图像")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    print("=" * 60)
    print("准备标注图像")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"布局: {args.layout}, IR位置: {args.ir_position}")
    print()
    
    # 处理图像
    stats = process_images(
        input_dir=input_dir,
        output_dir=output_dir,
        layout=args.layout,
        ir_position=args.ir_position,
        save_rgb=args.save_rgb
    )
    
    # 创建标注模板
    create_annotation_template(output_dir)
    
    # 打印结果
    print()
    print("=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"成功: {stats['success']}")
    print(f"跳过: {stats['skipped']}")
    print(f"失败: {stats['failed']}")
    print()
    print("下一步:")
    print(f"1. 安装LabelMe: pip install labelme")
    print(f"2. 启动标注: labelme {output_dir / 'ir_images'} --output {output_dir / 'annotations'}")
    print(f"3. 阅读标注规范: {output_dir / 'README.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
