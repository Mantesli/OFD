"""
Example script demonstrating how to use the OilSpillDetector.

This script shows how to:
1. Initialize the detector with custom parameters
2. Load IR and RGB images
3. Run the detection pipeline
4. Visualize and save results
"""

import cv2
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.oil_spill_detector import OilSpillDetector


def main():
    # ================= Configuration =================
    # Update these paths to your actual image paths
    IR_IMAGE_PATH = "data/images/000001_ir.jpg"
    RGB_IMAGE_PATH = "data/original/000001.jpg"
    OUTPUT_PATH = "results/detection_result.jpg"
    
    # VLM API Configuration
    # Set your API key here or use environment variable DASHSCOPE_API_KEY
    VLM_API_KEY = "sk-ef7db77064064747936dd65767cbd794"  # Qwen2.5-VL-72B API key
    VLM_MODEL = "qwen2.5-vl-72b-instruct"  # Qwen2.5-VL-72B-Instruct model
    
    # ================= Initialize Detector =================
    detector = OilSpillDetector(
        # Threshold parameters
        ir_threshold=200,           # IR hotspot threshold (0-255)
        rgb_dark_threshold=80,       # RGB dark color threshold (0-255)
        
        # Area filters
        min_area=500,               # Minimum contour area
        max_area=100000,            # Maximum contour area
        
        # VLM configuration
        vlm_api_key=VLM_API_KEY,
        vlm_model=VLM_MODEL,
        
        # IR enhancement
        clahe_clip_limit=3.0,       # CLAHE clip limit
        clahe_grid_size=(8, 8),      # CLAHE grid size
        
        # Morphological operations
        gaussian_blur_kernel=5,      # Gaussian blur kernel size
        morph_kernel_size=20,          # Morphological kernel size
        
        # Active observation
        min_crop_size=64,            # Minimum crop size for VLM
        padding_size=10,             # Padding around crops
    )
    
    # ================= Load Images =================
    print(f"Loading images...")
    print(f"  IR: {IR_IMAGE_PATH}")
    print(f"  RGB: {RGB_IMAGE_PATH}")
    
    ir_frame = cv2.imread(IR_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    rgb_frame = cv2.imread(RGB_IMAGE_PATH)
    
    if ir_frame is None:
        print(f"❌ Error: Could not load IR image from {IR_IMAGE_PATH}")
        return
    
    if rgb_frame is None:
        print(f"❌ Error: Could not load RGB image from {RGB_IMAGE_PATH}")
        return
    
    print(f"✅ Images loaded successfully")
    print(f"  IR shape: {ir_frame.shape}")
    print(f"  RGB shape: {rgb_frame.shape}")
    
    # ================= Run Detection =================
    proposals, verified_results = detector.detect(ir_frame, rgb_frame)
    
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
    
    result_img = detector.visualize_results(
        rgb_frame,
        proposals,
        verified_results,
        show=True,           # Display result window
        save_path=OUTPUT_PATH  # Save result image
    )
    
    print("\n✅ Detection complete!")


if __name__ == "__main__":
    main()
