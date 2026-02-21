"""
Module 2: MobileCLIP Feature Extractor
======================================
使用MobileCLIP提取RGB和红外图像的视觉特征，并利用文本编码进行语义引导。

核心功能：
1. 提取RGB图像的CLIP视觉特征
2. 提取红外图像的CLIP视觉特征（将IR视为灰度图像）
3. 计算语义权重W(x,y)：用于抑制干扰物
4. 零样本泄漏检测能力验证

技术要点：
- 使用MobileCLIP-S0/S1/S2变体（轻量级）
- 支持批量特征提取
- 文本prompt设计用于区分泄漏vs干扰物
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    logger.warning("open_clip not available, some features will be disabled")

from PIL import Image
import torchvision.transforms as T


@dataclass
class CLIPConfig:
    """MobileCLIP配置"""
    # 模型名称 - 支持的变体:
    # "ViT-B-32" (OpenAI CLIP), "MobileCLIP-S0", "MobileCLIP-S1", "MobileCLIP-S2"
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    
    # 特征维度（根据模型自动设置）
    feature_dim: int = 512
    
    # 输入图像尺寸
    image_size: int = 224
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 是否使用混合精度
    use_fp16: bool = True


# 预定义的文本提示词
DEFAULT_POSITIVE_PROMPTS = [
    "oil spill on ground",
    "oil leakage on soil", 
    "petroleum leak",
    "dark oil stain on earth",
    "crude oil contamination",
    "oil pollution on ground",
    "black oil puddle",
    "liquid oil spill",
]

DEFAULT_NEGATIVE_PROMPTS = [
    "dark soil",
    "bare ground", 
    "black earth",
    "shadow on ground",
    "tire tracks on snow",
    "vehicle wheel marks",
    "metal equipment",
    "oil pump station",
    "industrial machinery",
    "dry dead grass",
    "brown vegetation",
    "dirt road",
    "mud puddle",
    "frozen ground",
]


class MobileCLIPExtractor(nn.Module):
    """
    MobileCLIP特征提取器
    
    用于提取RGB和红外图像的CLIP特征，并计算语义权重。
    
    核心功能：
    1. extract_image_features(): 提取图像的视觉特征
    2. extract_text_features(): 提取文本的语义特征
    3. compute_semantic_weights(): 计算语义抑制权重W(x,y)
    4. zero_shot_classify(): 零样本分类
    
    Usage:
        extractor = MobileCLIPExtractor(config)
        
        # 提取特征
        rgb_features = extractor.extract_image_features(rgb_images)
        ir_features = extractor.extract_image_features(ir_images)
        
        # 计算语义权重
        weights = extractor.compute_semantic_weights(rgb_images)
        
        # 零样本分类
        probs = extractor.zero_shot_classify(rgb_images)
    """
    
    def __init__(self, config: Optional[CLIPConfig] = None):
        super().__init__()
        
        self.config = config or CLIPConfig()
        self.device = torch.device(self.config.device)
        
        # 加载模型
        self._load_model()
        
        # 预计算文本特征
        self.positive_prompts = DEFAULT_POSITIVE_PROMPTS
        self.negative_prompts = DEFAULT_NEGATIVE_PROMPTS
        self._precompute_text_features()
        
        logger.info(f"MobileCLIPExtractor initialized with {self.config.model_name}")
    
    def _load_model(self):
        """加载CLIP模型"""
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("open_clip is required. Install with: pip install open_clip_torch")
        
        # 尝试加载MobileCLIP，如果失败则使用标准CLIP
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.config.model_name,
                pretrained=self.config.pretrained
            )
            logger.info(f"Loaded {self.config.model_name} with {self.config.pretrained}")
        except Exception as e:
            logger.warning(f"Failed to load {self.config.model_name}: {e}")
            logger.info("Falling back to ViT-B-32")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="openai"
            )
        
        self.tokenizer = open_clip.get_tokenizer(self.config.model_name)
        
        # 移动到设备
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _precompute_text_features(self):
        """预计算文本特征（正向和负向提示）"""
        with torch.no_grad():
            # 正向提示特征
            pos_tokens = self.tokenizer(self.positive_prompts).to(self.device)
            self.positive_text_features = self.model.encode_text(pos_tokens)
            self.positive_text_features = F.normalize(self.positive_text_features, dim=-1)
            
            # 负向提示特征
            neg_tokens = self.tokenizer(self.negative_prompts).to(self.device)
            self.negative_text_features = self.model.encode_text(neg_tokens)
            self.negative_text_features = F.normalize(self.negative_text_features, dim=-1)
            
            logger.info(f"Precomputed {len(self.positive_prompts)} positive and "
                       f"{len(self.negative_prompts)} negative text features")
    
    def set_prompts(
        self, 
        positive_prompts: Optional[List[str]] = None,
        negative_prompts: Optional[List[str]] = None
    ):
        """
        设置自定义提示词并重新计算文本特征
        
        Args:
            positive_prompts: 正向提示词列表（指示泄漏）
            negative_prompts: 负向提示词列表（指示干扰物）
        """
        if positive_prompts:
            self.positive_prompts = positive_prompts
        if negative_prompts:
            self.negative_prompts = negative_prompts
        
        self._precompute_text_features()
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image: 输入图像，支持numpy数组、PIL Image或Tensor
            
        Returns:
            预处理后的Tensor (1, C, H, W)
        """
        if isinstance(image, torch.Tensor):
            # 假设已经是正确格式
            if image.dim() == 3:
                image = image.unsqueeze(0)
            return image.to(self.device)
        
        if isinstance(image, np.ndarray):
            # 处理灰度图（红外）
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=-1)
            
            image = Image.fromarray(image.astype(np.uint8))
        
        # 应用CLIP预处理
        image = self.preprocess(image).unsqueeze(0)
        return image.to(self.device)
    
    def preprocess_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> torch.Tensor:
        """
        批量预处理图像
        
        Args:
            images: 图像列表
            
        Returns:
            预处理后的Tensor (B, C, H, W)
        """
        processed = [self.preprocess_image(img) for img in images]
        return torch.cat(processed, dim=0)
    
    @torch.no_grad()
    def extract_image_features(
        self, 
        images: Union[torch.Tensor, List[np.ndarray], np.ndarray]
    ) -> torch.Tensor:
        """
        提取图像的CLIP视觉特征
        
        Args:
            images: 输入图像，可以是:
                - Tensor (B, C, H, W)
                - 图像列表 List[np.ndarray]
                - 单张图像 np.ndarray
                
        Returns:
            归一化的特征向量 (B, feature_dim)
        """
        # 预处理
        if isinstance(images, (list, np.ndarray)) and not isinstance(images, torch.Tensor):
            if isinstance(images, np.ndarray) and len(images.shape) <= 3:
                images = [images]
            images = self.preprocess_batch(images)
        elif isinstance(images, torch.Tensor) and images.dim() == 3:
            images = images.unsqueeze(0)
        
        images = images.to(self.device)

        # 使用混合精度
        if self.config.use_fp16 and self.device.type == 'cuda':
            # 修改前: with torch.cuda.amp.autocast():
            with torch.amp.autocast('cuda'):  # 修改后
                features = self.model.encode_image(images)
        
        # L2归一化
        features = F.normalize(features, dim=-1)
        
        return features
    
    @torch.no_grad()
    def extract_text_features(self, texts: List[str]) -> torch.Tensor:
        """
        提取文本的CLIP语义特征
        
        Args:
            texts: 文本列表
            
        Returns:
            归一化的特征向量 (N, feature_dim)
        """
        tokens = self.tokenizer(texts).to(self.device)

        if self.config.use_fp16 and self.device.type == 'cuda':
            # 修改前: with torch.cuda.amp.autocast():
            with torch.amp.autocast('cuda'):  # 修改后
                features = self.model.encode_text(tokens)
        
        features = F.normalize(features, dim=-1)
        return features
    
    @torch.no_grad()
    def compute_similarity(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor,
        temperature: float = 100.0
    ) -> torch.Tensor:
        """
        计算图像-文本相似度
        
        Args:
            image_features: 图像特征 (B, D)
            text_features: 文本特征 (N, D)
            temperature: 温度参数
            
        Returns:
            相似度矩阵 (B, N)
        """
        # 强制转换为 float32 以避免混合精度导致的类型不匹配 (Half vs Float)
        image_features = image_features.float()
        text_features = text_features.float()
        # 余弦相似度
        similarity = image_features @ text_features.T
        return similarity * temperature
    
    @torch.no_grad()
    def compute_semantic_weights(
        self, 
        images: Union[torch.Tensor, List[np.ndarray]],
        return_details: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算语义抑制权重 W(x,y)
        
        核心逻辑：与干扰物（负向提示）相似度越高，权重越低
        
        W(x,y) = 1 - max(CosSim(V_image, T_negative))
        
        Args:
            images: 输入图像
            return_details: 是否返回详细信息
            
        Returns:
            语义权重 (B,)，范围[0, 1]
            如果return_details=True，返回包含详细信息的字典
        """
        # 提取图像特征
        image_features = self.extract_image_features(images)
        
        # 计算与负向提示的相似度
        neg_similarity = self.compute_similarity(
            image_features, 
            self.negative_text_features,
            temperature=1.0  # 不缩放
        )
        
        # 取每个图像与所有负向提示的最大相似度
        max_neg_sim, max_neg_idx = neg_similarity.max(dim=-1)
        
        # 将相似度映射到[0, 1]范围
        # 相似度范围大约在[-1, 1]，使用sigmoid进行非线性映射
        max_neg_sim_normalized = torch.sigmoid(max_neg_sim * 3)  # 缩放因子可调
        
        # 计算权重：相似度越高，权重越低
        weights = 1 - max_neg_sim_normalized
        
        if return_details:
            # 同时计算正向相似度
            pos_similarity = self.compute_similarity(
                image_features,
                self.positive_text_features,
                temperature=1.0
            )
            max_pos_sim, max_pos_idx = pos_similarity.max(dim=-1)
            
            return {
                'weights': weights,
                'max_negative_similarity': max_neg_sim,
                'max_positive_similarity': max_pos_sim,
                'matched_negative_prompts': [self.negative_prompts[i] for i in max_neg_idx.cpu().tolist()],
                'matched_positive_prompts': [self.positive_prompts[i] for i in max_pos_idx.cpu().tolist()],
                'image_features': image_features
            }
        
        return weights
    
    @torch.no_grad()
    def zero_shot_classify(
        self, 
        images: Union[torch.Tensor, List[np.ndarray]],
        return_scores: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        零样本泄漏检测
        
        比较图像与正向/负向提示的相似度，判断是否为泄漏
        
        Args:
            images: 输入图像
            return_scores: 是否返回详细分数
            
        Returns:
            预测概率 (B,)，表示泄漏的概率
        """
        # 提取图像特征
        image_features = self.extract_image_features(images)
        
        # 计算与正向和负向提示的相似度
        pos_similarity = self.compute_similarity(
            image_features,
            self.positive_text_features,
            temperature=1.0
        )
        neg_similarity = self.compute_similarity(
            image_features,
            self.negative_text_features,
            temperature=1.0
        )
        
        # 聚合相似度（取最大值）
        pos_score = pos_similarity.max(dim=-1)[0]
        neg_score = neg_similarity.max(dim=-1)[0]
        
        # 使用softmax计算概率
        scores = torch.stack([neg_score, pos_score], dim=-1)
        probs = F.softmax(scores * 2, dim=-1)  # 温度缩放
        
        leak_prob = probs[:, 1]  # 泄漏概率
        
        if return_scores:
            return leak_prob, {
                'positive_score': pos_score,
                'negative_score': neg_score,
                'all_positive_similarities': pos_similarity,
                'all_negative_similarities': neg_similarity
            }
        
        return leak_prob
    
    def forward(
        self, 
        rgb_images: torch.Tensor,
        ir_images: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播：提取双模态特征
        
        Args:
            rgb_images: RGB图像 (B, 3, H, W)
            ir_images: 红外图像 (B, 3, H, W) 或 (B, 1, H, W)
            
        Returns:
            {
                'rgb_features': (B, D),
                'ir_features': (B, D) if ir_images provided,
                'semantic_weights': (B,)
            }
        """
        result = {}
        
        # RGB特征
        result['rgb_features'] = self.extract_image_features(rgb_images)
        
        # 红外特征（如果提供）
        if ir_images is not None:
            # 如果是单通道，扩展到3通道
            if ir_images.shape[1] == 1:
                ir_images = ir_images.repeat(1, 3, 1, 1)
            result['ir_features'] = self.extract_image_features(ir_images)
        
        # 语义权重
        result['semantic_weights'] = self.compute_semantic_weights(rgb_images)
        
        return result


def extract_clip_features(
    images: Union[List[np.ndarray], np.ndarray],
    model_name: str = "ViT-B-32",
    device: str = "cuda"
) -> np.ndarray:
    """
    便捷函数：提取CLIP特征
    
    Args:
        images: 图像列表或单张图像
        model_name: CLIP模型名称
        device: 计算设备
        
    Returns:
        特征数组 (N, D)
    """
    config = CLIPConfig(model_name=model_name, device=device)
    extractor = MobileCLIPExtractor(config)
    
    if isinstance(images, np.ndarray) and len(images.shape) == 3:
        images = [images]
    
    features = extractor.extract_image_features(images)
    return features.cpu().numpy()


def compute_semantic_weights(
    images: Union[List[np.ndarray], np.ndarray],
    positive_prompts: Optional[List[str]] = None,
    negative_prompts: Optional[List[str]] = None,
    model_name: str = "ViT-B-32",
    device: str = "cuda"
) -> np.ndarray:
    """
    便捷函数：计算语义权重
    
    Args:
        images: 图像列表
        positive_prompts: 正向提示词
        negative_prompts: 负向提示词
        model_name: CLIP模型名称
        device: 计算设备
        
    Returns:
        权重数组 (N,)
    """
    config = CLIPConfig(model_name=model_name, device=device)
    extractor = MobileCLIPExtractor(config)
    
    if positive_prompts or negative_prompts:
        extractor.set_prompts(positive_prompts, negative_prompts)
    
    if isinstance(images, np.ndarray) and len(images.shape) == 3:
        images = [images]
    
    weights = extractor.compute_semantic_weights(images)
    return weights.cpu().numpy()


class CLIPZeroShotValidator:
    """
    CLIP零样本能力验证器
    
    用于Week 2实验1：验证CLIP能否区分泄漏vs干扰物
    
    Usage:
        validator = CLIPZeroShotValidator()
        results = validator.validate(data_dir)
        validator.generate_report(results, output_path)
    """
    
    def __init__(self, config: Optional[CLIPConfig] = None):
        self.extractor = MobileCLIPExtractor(config)
    
    def validate_single(
        self, 
        image: np.ndarray, 
        label: int
    ) -> Dict:
        """验证单张图像"""
        prob, scores = self.extractor.zero_shot_classify([image], return_scores=True)
        
        pred = (prob > 0.5).int().item()
        correct = (pred == label)
        
        return {
            'prob': prob.item(),
            'pred': pred,
            'label': label,
            'correct': correct,
            'positive_score': scores['positive_score'].item(),
            'negative_score': scores['negative_score'].item()
        }
    
    def validate_batch(
        self, 
        images: List[np.ndarray], 
        labels: List[int],
        threshold: float = 0.5
    ) -> Dict:
        """
        批量验证
        
        Args:
            images: 图像列表
            labels: 标签列表（0=正常, 1=泄漏）
            threshold: 分类阈值
            
        Returns:
            验证结果统计
        """
        probs, scores = self.extractor.zero_shot_classify(images, return_scores=True)
        
        probs_np = probs.cpu().numpy()
        labels_np = np.array(labels)
        
        preds = (probs_np > threshold).astype(int)
        
        # 计算指标
        correct = (preds == labels_np)
        accuracy = correct.mean()
        
        # 按类别统计
        leak_mask = labels_np == 1
        normal_mask = labels_np == 0
        
        # 召回率（泄漏检出率）
        recall = preds[leak_mask].mean() if leak_mask.sum() > 0 else 0
        
        # 精确率
        precision = labels_np[preds == 1].mean() if (preds == 1).sum() > 0 else 0
        
        # 特异度（正常样本正确率）
        specificity = (1 - preds[normal_mask]).mean() if normal_mask.sum() > 0 else 0
        
        return {
            'accuracy': float(accuracy),
            'recall': float(recall),
            'precision': float(precision),
            'specificity': float(specificity),
            'f1': float(2 * precision * recall / (precision + recall + 1e-8)),
            'predictions': preds.tolist(),
            'probabilities': probs_np.tolist(),
            'threshold': threshold
        }
    
    def find_optimal_threshold(
        self, 
        images: List[np.ndarray], 
        labels: List[int],
        recall_target: float = 0.85
    ) -> Tuple[float, Dict]:
        """
        寻找满足目标召回率的最优阈值
        
        Args:
            images: 图像列表
            labels: 标签列表
            recall_target: 目标召回率
            
        Returns:
            (最优阈值, 该阈值下的指标)
        """
        probs = self.extractor.zero_shot_classify(images).cpu().numpy()
        labels_np = np.array(labels)
        
        best_threshold = 0.5
        best_metrics = None
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            preds = (probs > threshold).astype(int)
            
            leak_mask = labels_np == 1
            recall = preds[leak_mask].mean() if leak_mask.sum() > 0 else 0
            
            if recall >= recall_target:
                precision = labels_np[preds == 1].mean() if (preds == 1).sum() > 0 else 0
                accuracy = (preds == labels_np).mean()
                
                if best_metrics is None or precision > best_metrics.get('precision', 0):
                    best_threshold = threshold
                    best_metrics = {
                        'accuracy': float(accuracy),
                        'recall': float(recall),
                        'precision': float(precision),
                        'threshold': float(threshold)
                    }
        
        return best_threshold, best_metrics or {}


if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MobileCLIP Extractor")
    parser.add_argument("--image", type=str, help="Test image path")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # 初始化提取器
    config = CLIPConfig(device=args.device)
    extractor = MobileCLIPExtractor(config)
    
    if args.image:
        # 测试单张图像
        import cv2
        image = cv2.imread(args.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 提取特征
        features = extractor.extract_image_features([image])
        print(f"Feature shape: {features.shape}")
        
        # 零样本分类
        prob, scores = extractor.zero_shot_classify([image], return_scores=True)
        print(f"Leak probability: {prob.item():.4f}")
        print(f"Positive score: {scores['positive_score'].item():.4f}")
        print(f"Negative score: {scores['negative_score'].item():.4f}")
        
        # 语义权重
        weights = extractor.compute_semantic_weights([image], return_details=True)
        print(f"Semantic weight: {weights['weights'].item():.4f}")
        print(f"Matched negative prompt: {weights['matched_negative_prompts'][0]}")
    else:
        # 简单测试
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        features = extractor.extract_image_features([dummy_image])
        print(f"Test passed! Feature shape: {features.shape}")
