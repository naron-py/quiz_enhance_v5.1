import time
import cv2
import os
import sys
from ocr_processor import OCRProcessor
from config_manager import ConfigManager

def verify_speed():
    # Setup
    print("Initializing OCR Processor...")
    ocr = OCRProcessor()
    
    # Load debug crops to simulate a capture
    regions = {}
    for name in ['A', 'B', 'C', 'D']:
        path = f"debug_crops/{name}_crop.png"
        if os.path.exists(path):
            img = cv2.imread(path)
            # Simulate "capture" (which gives BGR)
            # OCRProcessor expects BGR input and preprocesses it.
            regions[name] = img
        else:
            print(f"Warning: {path} not found")
            
    if not regions:
        print("No regions to test!")
        return
        
    print(f"Testing on {len(regions)} regions...")
    
    # Warmup
    print("Warming up...")
    ocr.process_quiz_regions(regions)
    
    # Benchmark
    print("\n--- Benchmarking process_quiz_regions ---")
    start_time = time.time()
    iterations = 10
    for _ in range(iterations):
        results = ocr.process_quiz_regions(regions)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    
    # Check Result
    print("\n--- Accuracy Check ---")
    print(results)
    
    # Validations
    expected_B = "Karkaroff, 4"
    if expected_B in results.get('B', ''):
        print(f"[PASS] B contains '{expected_B}'")
    else:
        print(f"[FAIL] B is '{results.get('B')}' (Expected '{expected_B}')")

    print(f"\n>>> AVERAGE PROCESSING TIME (4 regions): {avg_time:.4f}s <<<")

if __name__ == "__main__":
    verify_speed()
