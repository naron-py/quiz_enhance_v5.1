import cv2
import os
import torch
import numpy as np
from ocr_processor import OCRProcessor

def test_direct_reco():
    print("Initializing Processor...")
    ocr = OCRProcessor()
    
    crops = {}
    for name in ['A', 'B', 'C', 'D']:
        path = f"debug_crops/{name}_crop.png"
        if os.path.exists(path):
            img = cv2.imread(path)
            # OCRProcessor.preprocess_image does Inversion
            # We must apply it to match live pipeline
            crops[name] = ocr.preprocess_image(img)
        else:
            print(f"Missing {path}")

    if not crops:
        print("No crops found in debug_crops/")
        return

    print(f"\nScanning {len(crops)} crops DIRECTLY (No Detection step)...")
    
    img_list = list(crops.values())
    names = list(crops.keys())
    
    # Direct Inference
    try:
        with ocr.inference_context():
            reco_out = ocr.model.reco_predictor(img_list)
    except Exception as e:
        print(f"Inference Error: {e}")
        return

    print("\n--- Results ---")
    for name, (text, conf) in zip(names, reco_out):
        print(f"[{name}] Conf: {conf:.4f} | Text: '{text}'")
        
    # Check Result
    expected = "Karkaroff, 4"
    b_text = reco_out[names.index('B')][0] if 'B' in names else ""
    
    if expected in b_text:
        print(f"\n[SUCCESS] Found '{expected}'")
    else:
        print(f"\n[FAIL] Got '{b_text}', expected '{expected}'")

if __name__ == "__main__":
    test_direct_reco()
