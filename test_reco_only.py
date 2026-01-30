import time
import cv2
import torch
import numpy as np
from doctr.models import ocr_predictor
from config_manager import ConfigManager
import os

def test_reco_only():
    print("Loading OCR Model (PARSeq)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load full predictor but we will access reco_predictor directly
    model = ocr_predictor(det_arch='db_mobilenet_v3_large', reco_arch='parseq', pretrained=True)
    model = model.to(device)
    model.eval()
    
    reco_model = model.reco_predictor
    
    # Load a sample crop (Choice B from previous diagnostics)
    # We need a crop that exists. 
    # debug_crops/B_crop.png should exist if analyze_regions was run.
    image_path = "debug_crops/B_crop.png"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found. Run analyze_regions.py first.")
        return

    print(f"Testing on {image_path}")
    img = cv2.imread(image_path)
    
    # Preprocessing (Inversion)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    processed = cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)
    
    # Doctr expects list of numpy arrays
    batch = [processed]
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        model(batch)
        reco_model(batch)
        
    # Benchmark Full Pipeline
    print("\n--- Benchmarking Full Pipeline (Det + Reco) ---")
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            res = model(batch)
    avg_full = (time.time() - start) / 10
    print(f"Average Full Time: {avg_full:.4f}s")
    
    # Benchmark Reco Only
    print("\n--- Benchmarking Reco Only (Skip Det) ---")
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            res = reco_model(batch)
    avg_reco = (time.time() - start) / 10
    print(f"Average Reco Time: {avg_reco:.4f}s")
    
    # Verify Accuracy of Reco Only
    print("\n--- Accuracy Check (Reco Only) ---")
    with torch.no_grad():
        out = reco_model(batch)
    # out is a list of (text, conf) tuples? Or RecoResult?
    # reco_predictor returns a list of results.
    print(f"Raw Output Type: {type(out)}")
    # actually reco_predictor returns a list of (value, confidence) tuples
    print(f"Result: {out}")
    
    speedup = (avg_full - avg_reco) / avg_full * 100
    print(f"\nPotential Speedup: {speedup:.1f}%")

if __name__ == "__main__":
    test_reco_only()
