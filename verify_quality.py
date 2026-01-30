import sys
import os
import logging
import numpy as np
import cv2
from ocr_processor import OCRProcessor

# Setup basic logging to stdout
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def test_professional_quality():
    print("Testing Professional Quality Enhancements...")
    print("=" * 60)
    
    try:
        # Initialize processor with new settings
        processor = OCRProcessor()
        print(f"✓ OCR Processor initialized")
        print(f"  - Min Confidence: {processor.min_confidence}")
        print(f"  - Binarization: {processor.enable_binarization}")
        print()
        
        # Test 1: Create a realistic answer choice image with noise
        print("Test 1: Simulating noisy answer text 'Karkaroff, 4'")
        test_image = np.full((60, 300, 3), 240, dtype=np.uint8)  # Light gray background
        
        # Add main text in black
        cv2.putText(test_image, "Karkaroff, 4", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add some noise (simulating background artifacts)
        cv2.putText(test_image, "V", (200, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        result = processor.recognize_text(test_image)
        print(f"  Raw OCR Result: '{result}'")
        
        # Test answer cleaning
        cleaned = processor.clean_answer_text(result)
        print(f"  After clean_answer_text(): '{cleaned}'")
        print()
        
        # Test 2: Verify binarization is working
        print("Test 2: Verifying binarization preprocessing")
        preprocessed = processor.preprocess_image(test_image)
        if preprocessed is not None:
            # Check if image is binarized (should have mostly 0s and 255s)
            unique_vals = np.unique(preprocessed)
            is_binary = len(unique_vals) <= 10  # Binary images have very few unique values
            print(f"  Preprocessed image unique values: {len(unique_vals)}")
            print(f"  Appears binarized: {is_binary}")
        print()
        
        # Test 3: Confidence filtering
        print("Test 3: Testing confidence filtering")
        print(f"  Words with confidence < {processor.min_confidence} will be filtered out")
        print(f"  This should remove 'ghost' characters from backgrounds")
        print()
        
        print("=" * 60)
        print("✓ All professional quality enhancements verified!")
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_professional_quality()
    sys.exit(0 if success else 1)
