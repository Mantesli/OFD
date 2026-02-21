#!/usr/bin/env python
"""
Step 3: Extract Features
========================
提取双模态特征（MobileCLIP + 温度纹理），保存为特征文件供训练使用。

这是Week 2实验2和Week 3的准备工作。

Usage:
    python scripts/03_extract_features.py \
        --data_dir ./data/sampled \
        --output_dir ./data/features \
        --device cuda
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import h5py
from tqdm import tqdm
from loguru import logger

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.mobileclip_extractor import MobileCLIPExtractor, CLIPConfig
from src.features.thermal_texture import ThermalTextureExtractor, ThermalConfig
from src.data.frame_splitter import split_rgb_ir, SplitConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Extract features for oil leak detection")
    
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Root data directory (with train/val/test subdirs)")
    parser.add_argument("--output_dir", type=str, default="./data/features",
                       help="Output directory for features")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split_mode", type=str, default="horizontal")
    parser.add_argument("--rgb_position", type=str, default="right")
    
    return parser.parse_args()


def extract_features_for_split(
    split_dir: Path,
    clip_extractor: MobileCLIPExtractor,
    thermal_extractor: ThermalTextureExtractor,
    split_config: SplitConfig,
    batch_size: int
) -> dict:
    """为单个数据集划分提取特征"""
    
    all_features = {
        'rgb_clip': [],
        'ir_clip': [],
        'thermal_texture': [],
        'semantic_weights': [],
        'labels': [],
        'paths': []
    }
    
    # 收集所有图像
    images_info = []
    for label_name, label_value in [('leak', 1), ('normal', 0)]:
        label_dir = split_dir / label_name
        if not label_dir.exists():
            continue
        
        for img_path in label_dir.glob('*.jpg'):
            images_info.append((img_path, label_value))
    
    logger.info(f"Processing {len(images_info)} images from {split_dir.name}")
    
    # 批量处理
    for i in tqdm(range(0, len(images_info), batch_size)):
        batch_info = images_info[i:i + batch_size]
        
        rgb_batch = []
        ir_batch = []
        labels_batch = []
        paths_batch = []
        
        for img_path, label in batch_info:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 分离RGB和IR
            rgb, ir = split_rgb_ir(img, split_config.split_mode, split_config.rgb_position)
            
            rgb_batch.append(rgb)
            ir_batch.append(ir)
            labels_batch.append(label)
            paths_batch.append(str(img_path))
        
        if len(rgb_batch) == 0:
            continue
        
        # 提取CLIP特征
        rgb_features = clip_extractor.extract_image_features(rgb_batch)
        ir_features = clip_extractor.extract_image_features(ir_batch)
        
        # 计算语义权重
        semantic_weights = clip_extractor.compute_semantic_weights(rgb_batch)
        
        # 提取温度纹理特征
        thermal_features = []
        for ir_img in ir_batch:
            features = thermal_extractor.extract_features(ir_img)
            feature_vector = thermal_extractor.features_to_vector(features)
            thermal_features.append(feature_vector)
        
        # 保存
        all_features['rgb_clip'].append(rgb_features.cpu().numpy())
        all_features['ir_clip'].append(ir_features.cpu().numpy())
        all_features['thermal_texture'].append(np.array(thermal_features))
        all_features['semantic_weights'].append(semantic_weights.cpu().numpy())
        all_features['labels'].extend(labels_batch)
        all_features['paths'].extend(paths_batch)
    
    # 合并所有批次
    result = {
        'rgb_clip': np.concatenate(all_features['rgb_clip'], axis=0),
        'ir_clip': np.concatenate(all_features['ir_clip'], axis=0),
        'thermal_texture': np.concatenate(all_features['thermal_texture'], axis=0),
        'semantic_weights': np.concatenate(all_features['semantic_weights'], axis=0),
        'labels': np.array(all_features['labels']),
        'paths': all_features['paths']
    }
    
    return result


def save_features_hdf5(features: dict, output_path: Path):
    """保存特征为HDF5格式"""
    with h5py.File(output_path, 'w') as f:
        for key, value in features.items():
            if key == 'paths':
                # 字符串需要特殊处理
                dt = h5py.special_dtype(vlen=str)
                f.create_dataset(key, data=value, dtype=dt)
            else:
                f.create_dataset(key, data=value, compression='gzip')
    
    logger.info(f"Saved features to {output_path}")


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化提取器
    logger.info(f"Initializing CLIP extractor: {args.clip_model}")
    clip_config = CLIPConfig(model_name=args.clip_model, device=args.device)
    clip_extractor = MobileCLIPExtractor(clip_config)
    
    logger.info("Initializing thermal feature extractor")
    thermal_config = ThermalConfig()
    thermal_extractor = ThermalTextureExtractor(thermal_config)
    
    split_config = SplitConfig(
        split_mode=args.split_mode,
        rgb_position=args.rgb_position
    )
    
    data_dir = Path(args.data_dir)
    
    # 为每个数据集划分提取特征
    for split_name in ['train', 'val', 'test']:
        split_dir = data_dir / split_name
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue
        
        logger.info(f"Processing {split_name} split...")
        
        features = extract_features_for_split(
            split_dir,
            clip_extractor,
            thermal_extractor,
            split_config,
            args.batch_size
        )
        
        # 保存特征
        output_path = output_dir / f'{split_name}_features.h5'
        save_features_hdf5(features, output_path)
        
        # 打印统计信息
        n_leak = (features['labels'] == 1).sum()
        n_normal = (features['labels'] == 0).sum()
        logger.info(f"{split_name}: {n_leak} leak + {n_normal} normal = {len(features['labels'])} total")
        logger.info(f"  RGB CLIP shape: {features['rgb_clip'].shape}")
        logger.info(f"  Thermal texture shape: {features['thermal_texture'].shape}")
    
    # 保存配置
    config_info = {
        'timestamp': datetime.now().isoformat(),
        'clip_model': args.clip_model,
        'split_mode': args.split_mode,
        'rgb_position': args.rgb_position,
        'feature_dims': {
            'rgb_clip': int(clip_config.feature_dim),
            'thermal_texture': int(features['thermal_texture'].shape[1])
        }
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_info, f, indent=2)
    
    logger.info("Feature extraction complete!")


if __name__ == "__main__":
    main()
