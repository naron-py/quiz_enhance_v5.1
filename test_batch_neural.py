import time
import cv2
import os
import torch
from ocr_processor import OCRProcessor

def test_batch_neural():
    print("Initializing Processor...")
    ocr = OCRProcessor()
    
    # Load and Preprocess Regions
    regions = []
    names = ['A', 'B', 'C', 'D']
    for name in names:
        path = f"debug_crops/{name}_crop.png"
        if os.path.exists(path):
            img = cv2.imread(path)
            processed = ocr.preprocess_image(img)
            regions.append(processed)
        else:
            print(f"Missing {path}")
            
    if not regions:
        print("No regions found.")
        return

    print(f"\nScanning {len(regions)} regions via BATCH NEURAL (Det + Reco)...")
    
    # Warmup
    ocr.model(regions)
    
    # Benchmark
    start_time = time.time()
    iterations = 10
    for _ in range(iterations):
        with ocr.inference_context():
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    result = ocr.model(regions)
            else:
                result = ocr.model(regions)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    print(f"\n>>> AVERAGE BATCH TIME: {avg_time:.4f}s <<<")
    
    # Check Accuracy (Last result)
    print("\n--- Results ---")
    for i, page in enumerate(result.pages):
        words = []
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    words.append(word.value)
        text = " ".join(words)
        print(f"[{names[i]}] {text}")
        
        if names[i] == 'B':
            if "Karkaroff" in text and "4" in text:
                print("  -> [PASS]")
            else:
                print(f"  -> [FAIL] Expected 'Karkaroff, 4'")

if __name__ == "__main__":
    test_batch_neural()
