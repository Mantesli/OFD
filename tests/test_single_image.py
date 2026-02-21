"""
Test script to detect oil spills in a single image.

This script processes a combined RGBT image (left IR + right RGB)
and runs the oil spill detection pipeline.
"""

import cv2
import sys
import os
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.oil_spill_detector import OilSpillDetector


def main():
    # ================= Configuration =================
    # Path to the combined RGBT image
    IMAGE_PATH = r"E:\work\oilfield-leak-detection-v4\data\original\000001.jpg"
    OUTPUT_PATH = r"E:\work\oilfield-leak-detection-v4\results\test_000001_result.jpg"
    
    # VLM API Configuration
    VLM_API_KEY = "sk-ef7db77064064747936dd65767cbd794"
    VLM_MODEL = "qwen2.5-vl-72b-instruct"
    
    # ================= Initialize Detector =================
    detector = OilSpillDetector(
        # Threshold parameters
        ir_threshold=200,
        rgb_dark_threshold=80,
        
        # Area filters
        min_area=500,
        max_area=100000,
        
        # VLM configuration
        vlm_api_key=VLM_API_KEY,
        vlm_model=VLM_MODEL,
        
        # IR enhancement
        clahe_clip_limit=3.0,
        clahe_grid_size=(8, 8),
        
        # Morphological operations
        gaussian_blur_kernel=5,
        morph_kernel_size=20,
        
        # Active observation
        min_crop_size=64,
        padding_size=10,
    )
    
    # ================= Load Image =================
    print(f"Loading image: {IMAGE_PATH}")
    full_img = cv2.imread(IMAGE_PATH)
    
    if full_img is None:
        print(f"Error: Could not load image from {IMAGE_PATH}")
        return
    
    print(f"Image shape: {full_img.shape}")
    
    # ================= Split Combined Image =================
    # Assuming the image is combined: left IR + right RGB
    h, w = full_img.shape[:2]
    mid = w // 2
    
    # Split into IR and RGB parts
    ir_part = full_img[:, :mid]
    rgb_part = full_img[:, mid:]
    
    print(f"Split image:")
    print(f"  IR part shape: {ir_part.shape}")
    print(f"  RGB part shape: {rgb_part.shape}")
    
    # ================= Run Detection =================
    print("\n" + "=" * 60)
    print("Starting Oil Spill Detection")
    print("=" * 60)
    
    proposals, verified_results = detector.detect(ir_part, rgb_part)
    
    # ================= Print Results =================
    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)
    
    if not proposals:
        print("No oil spills detected.")
    else:
        confirmed_leaks = sum(1 for r in verified_results if r.is_leak)
        print(f"Total proposals: {len(proposals)}")
        print(f"Confirmed leaks: {confirmed_leaks}")
        print(f"Rejected (false positives): {len(proposals) - confirmed_leaks}")
        
        print("\nDetailed Results:")
        for i, (proposal, result) in enumerate(zip(proposals, verified_results)):
            x, y, w, h = proposal.bbox
            print(f"\n  [{i+1}] BBox: ({x}, {y}, {w}, {h})")
            print(f"      Area: {proposal.area:.1f}")
            print(f"      IR Temp: {proposal.ir_temp:.1f}")
            print(f"      RGB Color: {proposal.rgb_color}")
            print(f"      Is Leak: {result.is_leak}")
            print(f"      Confidence: {result.confidence}")
            print(f"      Reason: {result.reason}")
    
    # ================= Visualize Results =================
    print("\n" + "=" * 60)
    print("VISUALIZING RESULTS")
    print("=" * 60)
    
    # Draw results on the RGB part
    result_img = rgb_part.copy()
    
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
    
    # Save result
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, result_img)
    print(f"\nResult saved to: {OUTPUT_PATH}")
    
    print("\nDetection complete!")


if __name__ == "__main__":
    main()
