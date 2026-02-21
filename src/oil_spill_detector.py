"""
Oil Spill Detection System for Winter Aerial Drone Monitoring
===============================================================

A two-stage detection system combining traditional CV with Vision-Language Models:
- Stage 1: Fast proposal generation using OpenCV (threshold + contour analysis)
- Stage 2: VLM verification for semantic understanding

Author: System Architect
Date: 2025
"""

import cv2
import numpy as np
import base64
import json
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Proposal:
    """Represents a candidate region detected in Stage 1"""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area: float
    ir_temp: float  # average temperature in IR
    rgb_color: Tuple[int, int, int]  # average RGB color

    contour: np.ndarray  # contour for shape analysis


@dataclass
class VLMResult:
    """Represents the verification result from Stage 2"""
    is_leak: bool
    confidence: str  # "high", "medium", "low"
    reason: str


class OilSpillDetector:
    """
    Oil Spill Detector using VLA-inspired two-stage approach.
    
    Stage 1: Traditional CV for fast proposal generation
    Stage 2: VLM for semantic verification
    """
    
    def __init__(
        self,
        ir_threshold: int = 200,
        rgb_dark_threshold: int = 80,
        min_area: int = 500,
        max_area: int = 100000,
        vlm_api_key: Optional[str] = None,
        vlm_model: str = "gpt-4o",
        clahe_clip_limit: float = 3.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        gaussian_blur_kernel: int = 5,
        morph_kernel_size: int = 20,
        min_crop_size: int = 64,
        padding_size: int = 10,
    ):
        """
        Initialize the Oil Spill Detector.
        
        Args:
            ir_threshold: Threshold for IR hotspot detection (0-255)
            rgb_dark_threshold: Threshold for RGB dark color filtering (0-255)
            min_area: Minimum contour area to consider
            max_area: Maximum contour area to consider
            vlm_api_key: API key for VLM service
            vlm_model: Model name for VLM (e.g., "gpt-4o", "llava-v1.6")
            clahe_clip_limit: CLAHE clip limit for IR enhancement
            clahe_grid_size: CLAHE grid size for IR enhancement
            gaussian_blur_kernel: Kernel size for Gaussian blur
            morph_kernel_size: Kernel size for morphological operations
            min_crop_size: Minimum crop size for VLM input
            padding_size: Padding size around crops for context
        """
        self.ir_threshold = ir_threshold
        self.rgb_dark_threshold = rgb_dark_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.vlm_api_key = vlm_api_key
        self.vlm_model = vlm_model
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.gaussian_blur_kernel = gaussian_blur_kernel
        self.morph_kernel_size = morph_kernel_size
        self.min_crop_size = min_crop_size
        self.padding_size = padding_size
        
        # Initialize CLAHE for IR enhancement
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size
        )
        
        # Morphological kernel for de-fragmentation
        self.morph_kernel = np.ones(
            (self.morph_kernel_size, self.morph_kernel_size),
            np.uint8
        )
    
    def preprocess_images(
        self,
        ir_frame: np.ndarray,
        rgb_frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess IR and RGB frames for detection.
        
        Args:
            ir_frame: Infrared frame (single channel or BGR)
            rgb_frame: RGB frame (BGR format)
        
        Returns:
            Tuple of preprocessed (ir_frame, rgb_frame)
        """
        # Ensure IR is single channel
        if len(ir_frame.shape) == 3:
            ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
        
        # Normalize IR to 0-255 range
        ir_frame = cv2.normalize(ir_frame, None, 0, 255, cv2.NORM_MINMAX)
        
        # Ensure RGB is BGR format
        if len(rgb_frame.shape) == 2:
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_GRAY2BGR)
        
        # Check if frames have same dimensions
        if ir_frame.shape != rgb_frame.shape[:2]:
            print(f"⚠️ Warning: IR and RGB frames have different dimensions.")
            print(f"   IR: {ir_frame.shape}, RGB: {rgb_frame.shape[:2]}")
            # Resize RGB to match IR
            rgb_frame = cv2.resize(rgb_frame, (ir_frame.shape[1], ir_frame.shape[0]))
        
        return ir_frame, rgb_frame
    
    def _filter_regular_shapes(
        self,
        contour: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> bool:
        """
        Filter out regular geometric shapes (rectangles, circles, etc.)
        that are likely to be machinery, vehicles, or shadows.
        
        Args:
            contour: Contour points
            bbox: Bounding box (x, y, w, h)
        
        Returns:
            True if shape is irregular (keep), False if regular (discard)
        """
        x, y, w, h = bbox
        
        # Calculate contour area and bounding box area
        contour_area = cv2.contourArea(contour)
        bbox_area = w * h
        
        # Avoid division by zero
        if bbox_area == 0:
            return False
        
        # Solidity: contour_area / bbox_area
        # Oil spills have lower solidity (irregular shapes)
        # Machinery/vehicles have higher solidity (rectangular)
        solidity = contour_area / bbox_area
        
        # Filter out very solid shapes (likely machinery)
        if solidity > 0.85:
            return False
        
        # Aspect ratio filter
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Very elongated shapes (pipes, shadows)
        if aspect_ratio > 5.0 or aspect_ratio < 0.2:
            return False
        
        # Calculate convex hull and hull area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            return False
        
        # Convexity: contour_area / hull_area
        # Oil spills have lower convexity (irregular boundaries)
        convexity = contour_area / hull_area
        
        # Filter out very convex shapes (likely regular objects)
        if convexity > 0.95:
            return False
        
        # Calculate moments for circularity
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return False
        
        # Circularity: 4 * pi * area / perimeter^2
        # Perfect circle = 1.0, irregular shapes < 1.0
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
        
        circularity = 4 * np.pi * contour_area / (perimeter ** 2)
        
        # Filter out very circular shapes (likely round machinery)
        if circularity > 0.85:
            return False
        
        # Keep irregular shapes
        return True
    
    def generate_proposals(
        self,
        ir_frame: np.ndarray,
        rgb_frame: np.ndarray
    ) -> List[Proposal]:
        """
        Stage 1: Generate candidate proposals using traditional CV.
        
        Process:
        1. IR stream: CLAHE enhancement -> Gaussian blur -> Threshold
        2. RGB stream: Grayscale -> Threshold (dark regions)
        3. Mask fusion: bitwise AND
        4. Contour extraction and shape filtering
        
        Args:
            ir_frame: Preprocessed IR frame (single channel)
            rgb_frame: Preprocessed RGB frame (BGR)
        
        Returns:
            List of Proposal objects
        """
        print(f"[Stage 1] Generating proposals...")
        
        # === IR Stream ===
        # Apply CLAHE for contrast enhancement
        ir_enhanced = self.clahe.apply(ir_frame)
        
        # Gaussian blur to reduce noise
        ir_blur = cv2.GaussianBlur(
            ir_enhanced,
            (self.gaussian_blur_kernel, self.gaussian_blur_kernel),
            0
        )
        
        # Threshold to extract hotspots
        _, ir_mask = cv2.threshold(
            ir_blur,
            self.ir_threshold,
            255,
            cv2.THRESH_BINARY
        )
        
        # === RGB Stream ===
        # Convert to grayscale
        rgb_gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold to extract dark regions (oil appears dark)
        _, rgb_mask = cv2.threshold(
            rgb_gray,
            self.rgb_dark_threshold,
            255,
            cv2.THRESH_BINARY_INV
        )
        
        # === Mask Fusion ===
        # Combine IR and RGB masks
        final_mask = cv2.bitwise_and(ir_mask, rgb_mask)
        
        # Morphological operations for de-fragmentation
        final_mask = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_DILATE,
            self.morph_kernel
        )
        
        # === Contour Extraction ===
        contours, _ = cv2.findContours(
            final_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        proposals = []
        
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, w, h)
            
            # Area filter
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            
            # Shape regularity filter
            if not self._filter_regular_shapes(contour, bbox):
                continue
            
            # Calculate average IR temperature in region
            ir_roi = ir_frame[y:y+h, x:x+w]
            ir_temp = float(np.mean(ir_roi)) if ir_roi.size > 0 else 0.0
            
            # Calculate average RGB color in region
            rgb_roi = rgb_frame[y:y+h, x:x+w]
            rgb_color = tuple(map(int, np.mean(rgb_roi, axis=(0, 1)))) if rgb_roi.size > 0 else (0, 0, 0)
            
            # Create proposal
            proposal = Proposal(
                bbox=bbox,
                area=area,
                ir_temp=ir_temp,
                rgb_color=rgb_color,
                contour=contour
            )
            proposals.append(proposal)
        
        print(f"[Stage 1] Generated {len(proposals)} proposals")
        return proposals
    
    def _enhance_small_crop(
        self,
        crop_ir: np.ndarray,
        crop_rgb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhance small crops for better VLM understanding.
        "Active Observation" - pad or resize small regions.
        
        Args:
            crop_ir: Cropped IR region
            crop_rgb: Cropped RGB region
        
        Returns:
            Tuple of enhanced (crop_ir, crop_rgb)
        """
        h_ir, w_ir = crop_ir.shape[:2]
        h_rgb, w_rgb = crop_rgb.shape[:2]
        
        # Check if crop is too small
        if h_ir < self.min_crop_size or w_ir < self.min_crop_size:
            # Calculate scale factor
            scale = max(
                self.min_crop_size / h_ir,
                self.min_crop_size / w_ir
            )
            
            # Resize crops
            new_h_ir = int(h_ir * scale)
            new_w_ir = int(w_ir * scale)
            crop_ir = cv2.resize(crop_ir, (new_w_ir, new_h_ir), interpolation=cv2.INTER_CUBIC)
            
            new_h_rgb = int(h_rgb * scale)
            new_w_rgb = int(w_rgb * scale)
            crop_rgb = cv2.resize(crop_rgb, (new_w_rgb, new_h_rgb), interpolation=cv2.INTER_CUBIC)
            
            print(f"[Active Observation] Resized crop from ({w_ir}x{h_ir}) to ({new_w_ir}x{new_h_ir})")
        
        return crop_ir, crop_rgb
    
    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        """Encode image to base64 for API transmission"""
        _, buffer = cv2.imencode('.jpg', image)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    
    def verify_with_vlm(
        self,
        crop_ir: np.ndarray,
        crop_rgb: np.ndarray
    ) -> VLMResult:
        """
        Stage 2: Verify candidate using Vision-Language Model.
        
        Args:
            crop_ir: Cropped IR region
            crop_rgb: Cropped RGB region
        
        Returns:
            VLMResult with is_leak, confidence, and reason
        """
        # Enhance small crops (Active Observation)
        crop_ir, crop_rgb = self._enhance_small_crop(crop_ir, crop_rgb)
        
        # Convert IR to BGR if single channel
        if len(crop_ir.shape) == 2:
            crop_ir = cv2.cvtColor(crop_ir, cv2.COLOR_GRAY2BGR)
        
        # Combine IR and RGB horizontally for VLM input
        combined_img = np.hstack((crop_ir, crop_rgb))
        
        # Encode to base64
        base64_img = self._encode_image_to_base64(combined_img)
        
        # System Prompt for VLM
        # This is the KEY to distinguishing soil, machinery, and oil spills
        system_prompt = """
你是一位专业的油田工业视觉专家。擅长从高空无人机视角识别冬季环境下的石油泄漏。

【图像说明】
- 左侧图像：红外热成像（高亮区域代表高温，暗色区域代表低温）
- 右侧图像：可见光RGB图像

【任务】
请判断图像中是否包含【地面石油泄漏】。

【判断标准】

1. 石油泄漏特征：
   - 红外特征：中心高温，边缘低温，呈放射状梯度分布
   - RGB特征：黑色或深褐色，形状不规则（非几何形状）
   - 边缘特征：边缘呈锯齿状、羽化状或渗透状
   - 扩散特征：看起来像液体在地面渗透或扩散
   - 温度特征：红外有明显的温差分布

2. 干扰物特征：
   - 裸露土壤：
     * RGB：棕褐色，但边缘锐利，无液体扩散感
     * 红外：可能呈现暖色，但温度分布均匀，无放射状梯度
     * 形状：通常为大面积连续区域，边界自然但规则
   
   - 管道/设备：
     * RGB：笔直的线条、规则的几何长条或矩形
     * 红外：高温区域，但形状规则（直线、矩形）
     * 边缘：边缘锐利、几何形状规则
   
   - 车辆/机械：
     * RGB：孤立的固体形态，有明显的轮廓和结构
     * 红外：高温点，但形状规则（矩形、圆形等）
     * 形状：有明显的几何形状，边缘清晰
   
   - 阴影：
     * RGB：黑色或深灰色，但边缘锐利、几何形状规则
     * 红外：低温区域，温度分布均匀
     * 形状：通常为长条形或规则几何形状

【输出格式】
请仅输出JSON格式：
{"is_leak": true/false, "confidence": "high/medium/low", "reason": "简述判断依据"}
"""
        
        # Qwen2.5-VL-72B-Instruct API call
        try:
            from openai import OpenAI
            import os
            
            client = OpenAI(
                api_key=self.vlm_api_key or os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            
            response = client.chat.completions.create(
                model=self.vlm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": base64_img}},
                            {"type": "text", "text": system_prompt},
                        ],
                    },
                ],
                temperature=0.01,
            )
            
            content = response.choices[0].message.content
            # Clean JSON response (remove markdown symbols)
            content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)
            
            return VLMResult(
                is_leak=result.get("is_leak", False),
                confidence=result.get("confidence", "low"),
                reason=result.get("reason", "No reason provided")
            )
        except Exception as e:
            print(f"VLM API call failed: {e}")
            # Return mock result on error
            return VLMResult(
                is_leak=False,
                confidence="low",
                reason=f"API Error: {str(e)}"
            )
    
    def detect(
        self,
        ir_frame: np.ndarray,
        rgb_frame: np.ndarray
    ) -> Tuple[List[Proposal], List[VLMResult]]:
        """
        Main detection pipeline coordinating Stage 1 and Stage 2.
        
        Args:
            ir_frame: Infrared frame
            rgb_frame: RGB frame
        
        Returns:
            Tuple of (proposals, verified_results)
        """
        print("=" * 60)
        print("Starting Oil Spill Detection Pipeline")
        print("=" * 60)
        
        # Preprocess images
        ir_frame, rgb_frame = self.preprocess_images(ir_frame, rgb_frame)
        
        # Stage 1: Generate proposals
        proposals = self.generate_proposals(ir_frame, rgb_frame)
        
        if not proposals:
            print("No proposals generated. Detection complete.")
            return [], []
        
        # Stage 2: Verify proposals with VLM
        print(f"\n[Stage 2] Starting VLM verification...")
        verified_results = []
        
        for i, proposal in enumerate(proposals):
            print(f"\n--- Processing proposal {i + 1}/{len(proposals)} ---")
            
            x, y, w, h = proposal.bbox
            
            # Crop with padding for context
            x1, y1 = max(0, x - self.padding_size), max(0, y - self.padding_size)
            x2, y2 = min(rgb_frame.shape[1], x + w + self.padding_size), min(rgb_frame.shape[0], y + h + self.padding_size)
            
            crop_rgb = rgb_frame[y1:y2, x1:x2]
            crop_ir = ir_frame[y1:y2, x1:x2]
            
            # Verify with VLM
            result = self.verify_with_vlm(crop_ir, crop_rgb)
            verified_results.append(result)
            
            print(f"Result: is_leak={result.is_leak}, confidence={result.confidence}, reason={result.reason}")
        
        print(f"\n[Stage 2] Verification complete. {sum(1 for r in verified_results if r.is_leak)} leaks confirmed.")
        
        return proposals, verified_results
    
    def visualize_results(
        self,
        rgb_frame: np.ndarray,
        proposals: List[Proposal],
        verified_results: List[VLMResult],
        show: bool = True,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detection results on the RGB frame.
        
        Args:
            rgb_frame: Original RGB frame
            proposals: List of proposals from Stage 1
            verified_results: List of VLM results from Stage 2
            show: Whether to display the result
            save_path: Path to save the result image
        
        Returns:
            Result image with annotations
        """
        result_img = rgb_frame.copy()
        
        for proposal, result in zip(proposals, verified_results):
            x, y, w, h = proposal.bbox
            
            if result.is_leak:
                # Confirmed leak: Green box with thick border
                color = (0, 255, 0)
                label = f"LEAK ({result.confidence})"
                thickness = 3
            else:
                # Rejected proposal: Red box with thinner border
                color = (0, 0, 255)
                label = "Ignored"
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label
            cv2.putText(
                result_img,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        if show:
            cv2.imshow("Oil Spill Detection Results", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        if save_path:
            # Create directory if it doesn't exist
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_path, result_img)
            print(f"Result saved to: {save_path}")
        
        return result_img
