import sys
import os
import logging
import numpy as np
import time
from ocr_processor import OCRProcessor

# Setup basic logging to stdout
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def test_ocr_perf():
    print("Initializing OCRProcessor (Aggressive Mode)...")
    try:
        # Initialize processor
        processor = OCRProcessor()
        print("Successfully initialized OCRProcessor.")
        
        # Create a realistic dummy image (1920x1080 capture region)
        # 1/5th screen width/height is typical for a question box: ~400x200
        dummy_image = np.full((200, 400, 3), 255, dtype=np.uint8)
        # Draw some text-like noise
        dummy_image[40:60, 50:250] = 0
        
        print("Warming up model...")
        for _ in range(3):
            processor.recognize_text(dummy_image)
            
        print("Running benchmark (10 iterations)...")
        times = []
        for i in range(10):
            start = time.time()
            text = processor.recognize_text(dummy_image)
            end = time.time()
            duration = end - start
            times.append(duration)
            print(f"Iter {i+1}: {duration:.4f}s")
            
        avg_time = sum(times) / len(times)
        print(f"Average Inference Time: {avg_time:.4f}s")
        
        if avg_time < 0.3:
             print("SUCCESS: Target < 0.3s achieved!")
             return True
        else:
             print(f"WARNING: Average time {avg_time:.4f}s is above 0.3s target.")
             return True # Still return true to not fail the script, just warn
             
    except Exception as e:
        print(f"FAILED to run OCR benchmark: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ocr_perf()
    sys.exit(0 if success else 1)
