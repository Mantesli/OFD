#!/usr/bin/env python
"""
Step 1: Sample Frames from Videos
=================================
从视频中抽帧采样，按类别分层采样，划分训练/验证/测试集。

Usage:
    # 从视频采样
    python scripts/01_sample_frames.py \
        --video_dir /path/to/videos \
        --output_dir ./data/sampled \
        --annotation /path/to/annotations.json \
        --sample_interval 30

    # 从已有图像目录采样
    python scripts/01_sample_frames.py \
        --image_dir /path/to/images \
        --output_dir ./data/sampled \
        --mode image
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.video_sampler import (
    VideoSampler, 
    ImageSampler,
    SamplingConfig,
    sample_from_videos
)
from src.data.frame_splitter import batch_split_images, SplitConfig
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Sample frames for oil leak detection")
    
    # 输入源（二选一）
    parser.add_argument("--video_dir", type=str, help="Directory containing video files")
    parser.add_argument("--image_dir", type=str, help="Directory containing image files")
    
    # 输出
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    # 标注文件
    parser.add_argument("--annotation", type=str, help="Annotation file (JSON or CSV)")
    
    # 采样配置
    parser.add_argument("--sample_interval", type=int, default=30, 
                       help="Sample every N frames from video")
    parser.add_argument("--num_leak_train", type=int, default=160)
    parser.add_argument("--num_normal_train", type=int, default=1600)
    parser.add_argument("--num_leak_val", type=int, default=20)
    parser.add_argument("--num_normal_val", type=int, default=600)
    parser.add_argument("--num_leak_test", type=int, default=20)
    parser.add_argument("--num_normal_test", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # 图像配置
    parser.add_argument("--image_size", type=int, nargs=2, default=[384, 384])
    parser.add_argument("--split_mode", type=str, default="horizontal",
                       choices=["horizontal", "vertical", "auto"])
    parser.add_argument("--rgb_position", type=str, default="right",
                       choices=["left", "right", "top", "bottom"])
    
    # 其他选项
    parser.add_argument("--split_images", action="store_true",
                       help="Also save split RGB and IR images separately")
    parser.add_argument("--mode", type=str, default="video",
                       choices=["video", "image"],
                       help="Input mode: video or image directory")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建采样配置
    config = SamplingConfig(
        sample_interval=args.sample_interval,
        num_leak_train=args.num_leak_train,
        num_normal_train=args.num_normal_train,
        num_leak_val=args.num_leak_val,
        num_normal_val=args.num_normal_val,
        num_leak_test=args.num_leak_test,
        num_normal_test=args.num_normal_test,
        random_seed=args.seed,
        image_size=tuple(args.image_size),
        split_mode=args.split_mode,
        rgb_position=args.rgb_position
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "video":
        # 从视频采样
        if not args.video_dir:
            logger.error("--video_dir is required for video mode")
            sys.exit(1)
        
        logger.info(f"Sampling from videos in {args.video_dir}")
        
        sampler = VideoSampler(config)
        sampler.scan_videos(args.video_dir)
        
        if args.annotation:
            sampler.load_annotations(args.annotation)
        else:
            logger.warning("No annotation file provided. "
                          "All frames will be treated as 'normal'.")
        
        result = sampler.sample_and_save(
            output_dir, 
            save_split_images=args.split_images
        )
        
    else:
        # 从图像目录采样
        if not args.image_dir:
            logger.error("--image_dir is required for image mode")
            sys.exit(1)
        
        logger.info(f"Sampling from images in {args.image_dir}")
        
        sampler = ImageSampler(config)
        result = sampler.sample_from_directory(
            args.image_dir,
            output_dir
        )
    
    # 如果需要，分离RGB和IR
    if args.split_images and args.mode == "image":
        logger.info("Splitting RGB and IR images...")
        split_config = SplitConfig(
            split_mode=args.split_mode,
            rgb_position=args.rgb_position
        )
        
        for split_name in ['train', 'val', 'test']:
            split_dir = output_dir / split_name
            if split_dir.exists():
                for label in ['leak', 'normal']:
                    label_dir = split_dir / label
                    if label_dir.exists():
                        batch_split_images(
                            label_dir,
                            label_dir,
                            split_config
                        )
    
    logger.info(f"Sampling complete!")
    logger.info(f"Total frames: {result['total_frames'] if 'total_frames' in result else result.get('total_images', 0)}")
    logger.info(f"Metadata saved to: {result['metadata_path']}")
    
    # 打印数据集统计
    print("\n" + "="*50)
    print("Dataset Statistics")
    print("="*50)
    
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        if split_dir.exists():
            leak_count = len(list((split_dir / 'leak').glob('*.jpg'))) if (split_dir / 'leak').exists() else 0
            normal_count = len(list((split_dir / 'normal').glob('*.jpg'))) if (split_dir / 'normal').exists() else 0
            print(f"{split:>8}: {leak_count:>4} leak + {normal_count:>5} normal = {leak_count + normal_count:>5} total")
    
    print("="*50)


if __name__ == "__main__":
    main()
