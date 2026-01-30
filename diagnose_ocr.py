import sys
import os
import logging
import cv2
import numpy as np
from pathlib import Path
from ocr_processor import OCRProcessor
from PIL import Image

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def diagnose_ocr_pipeline(image_path, save_dir="ocr_diagnostic"):
    """
    Comprehensive diagnostic of the OCR pipeline.
    Saves intermediate images and logs detailed OCR results.
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    Path(save_dir).mkdir(exist_ok=True)
    
    # Setup file logging
    log_file = os.path.join(save_dir, "diagnostic_log.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        def log(msg):
            print(msg)
            f.write(msg + "\n")
            
        log("=" * 80)
        log("OCR PIPELINE DIAGNOSTIC")
        log("=" * 80)
    
    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not load {image_path}")
        return
    
    cv2.imwrite(f"{save_dir}/00_original.png", img)
    print(f"[OK] Original image: {img.shape}")
    
    # Test WITHOUT any preprocessing
    print("\n--- TEST 1: RAW IMAGE (no preprocessing) ---")
    processor_raw = OCRProcessor()
    processor_raw.enable_binarization = False
    processor_raw.min_confidence = 0.0  # Accept everything
    
    # Override model to test PARSeq
    from doctr.models import ocr_predictor
    import torch
    print("Loading PARSeq model for testing...")
    processor_raw.model = ocr_predictor(det_arch='db_mobilenet_v3_large', reco_arch='parseq', pretrained=True)
    if torch.cuda.is_available():
        processor_raw.model = processor_raw.model.cuda()
    processor_raw.model.eval()
    
    # Manually preprocess without binarization
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{save_dir}/01_raw_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    
    result_raw = processor_raw.model([rgb])
    print("Raw OCR Result:")
    for page in result_raw.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    print(f"  Word: '{word.value}' (confidence: {word.confidence:.3f})")
    
    # Test WITH CLAHE only (no binarization)
    print("\n--- TEST 2: WITH CLAHE (no binarization) ---")
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    cv2.imwrite(f"{save_dir}/02_clahe_only.png", cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
    
    result_clahe = processor_raw.model([enhanced])
    print("CLAHE OCR Result:")
    for page in result_clahe.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    print(f"  Word: '{word.value}' (confidence: {word.confidence:.3f})")
    
    # Test WITH BINARIZATION (current approach)
    print("\n--- TEST 3: WITH BINARIZATION (Otsu's) ---")
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(f"{save_dir}/03_binarized.png", binary)
    
    result_binary = processor_raw.model([binary_rgb])
    print("Binarized OCR Result:")
    for page in result_binary.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    print(f"  Word: '{word.value}' (confidence: {word.confidence:.3f})")
    
    # Test 4: INVERTED (Gray + Invert)
    print("\n--- TEST 4: INVERTED (Gray + Invert) ---")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    inverted_rgb = cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(f"{save_dir}/04_inverted.png", inverted)
    
    result_inverted = processor_raw.model([inverted_rgb])
    print("Inverted OCR Result:")
    for page in result_inverted.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    print(f"  Word: '{word.value}' (confidence: {word.confidence:.3f})")
    
    # Test 5: UPSCALE 2x (Bicubic)
    print("\n--- TEST 5: UPSCALE 2x (Bicubic) ---")
    height, width = rgb.shape[:2]
    upscaled = cv2.resize(rgb, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"{save_dir}/05_upscaled.png", cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR))
    
    result_upscaled = processor_raw.model([upscaled])
    print("Upscaled 2x OCR Result:")
    for page in result_upscaled.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    print(f"  Word: '{word.value}' (confidence: {word.confidence:.3f})")

    # Test confidence distribution
    print("\n--- CONFIDENCE ANALYSIS ---")
    all_confidences = []
    for page in result_clahe.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    all_confidences.append(word.confidence)
    
    if all_confidences:
        print(f"Total words detected: {len(all_confidences)}")
        print(f"Min confidence: {min(all_confidences):.3f}")
        print(f"Max confidence: {max(all_confidences):.3f}")
        print(f"Avg confidence: {sum(all_confidences)/len(all_confidences):.3f}")
        print(f"Words below 0.6: {sum(1 for c in all_confidences if c < 0.6)}/{len(all_confidences)}")
        print(f"Words below 0.4: {sum(1 for c in all_confidences if c < 0.4)}/{len(all_confidences)}")
    
    print("\n" + "=" * 80)
    print(f"Diagnostic images saved to: {save_dir}/")
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_ocr.py <image_path>")
        print("\nPlease provide the path to a captured answer choice image.")
        sys.exit(1)
    
    # Redirect stdout to a file to ensure we capture output
    output_file = "diagnostic_output_full.txt"
    original_stdout = sys.stdout
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        try:
            diagnose_ocr_pipeline(sys.argv[1])
        finally:
            sys.stdout = original_stdout
            
    print(f"Diagnostic completed. Output saved to {output_file}")
