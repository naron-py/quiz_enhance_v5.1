import cv2
import json
import os
import sys
from pathlib import Path
from ocr_processor import OCRProcessor

def analyze_regions(fullscreen_path, config_path):
    print(f"Analyzing {fullscreen_path}...")
    
    # Load image
    img = cv2.imread(fullscreen_path)
    if img is None:
        print("Failed to load image")
        return

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    regions = config['answer_regions']
    
    # Initialize OCR
    ocr = OCRProcessor()
    # Force settings just to be sure we match current state
    ocr.enable_binarization = False
    ocr.min_confidence = 0.3
    
    debug_dir = Path('debug_crops')
    debug_dir.mkdir(exist_ok=True)
    
    for label, r in regions.items():
        x, y, w, h = r['x'], r['y'], r['width'], r['height']
        
        # Crop
        crop = img[y:y+h, x:x+w]
        
        # Save crop
        crop_path = debug_dir / f"{label}_crop.png"
        cv2.imwrite(str(crop_path), crop)
        print(f"\n--- Region {label} ---")
        print(f"Saved crop to {crop_path}")
        print(f"Dimensions: {w}x{h}")
        
        # Run OCR
        # Use the standard preprocessing pipeline (Inversion, etc.)
        processed_crop = ocr.preprocess_image(crop)
        if processed_crop is None:
            print(f"  [Error] Preprocessing failed for {label}")
            continue
            
        # Run inference
        # Note: preprocess_image returns RGB numpy array suitable for model, but we need list check
        result = ocr.model([processed_crop])
        
        raw_text = ""
        print("Detailed OCR Output:")
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = ""
                    for word in line.words:
                        print(f"  Word: '{word.value}' (Conf: {word.confidence:.2f})")
                        line_text += word.value + " "
                    raw_text += line_text
        
        print(f"Full Extracted Text: '{raw_text.strip()}'")
        
        # Check if number is visible in simple check
        # (Visually inspecting the saved image is the real test)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_regions.py <fullscreen_image_path>")
        sys.exit(1)
    
    analyze_regions(sys.argv[1], 'config.json')
