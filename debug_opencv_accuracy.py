import cv2
import numpy as np
import os
# Mock OCRProcessor to avoid loading big models if not needed, 
# but we need preprocess_image logic. Best to import.
from ocr_processor import OCRProcessor

def debug():
    try:
        ocr = OCRProcessor()
    except:
        print("Could not load OCRProcessor (maybe cuda missing?), using manual preprocess")
        ocr = None

    # Path to user artifact (Row 93 Screenshot)
    img_path = r"C:\Users\Fade\.gemini\antigravity\brain\88890335-9549-4e0c-b78f-4b5c34b727a7\uploaded_media_0_1769781202006.png"
    
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}")
        return

    # Use robust loading
    full_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    if full_img is None:
        print("Failed to load image")
        return
        
    h_img, w_img = full_img.shape[:2]
    print(f"Image Loaded: {w_img}x{h_img}")

    # Region A (Row 93) - Missing
    # Region D (Row 97) - Scrambled to "97 Row"
    regions = {
        "A": {"x": 391, "y": 889, "width": 556, "height": 74}, 
        "D": {"x": 1042, "y": 982, "width": 553, "height": 66}  
    }
    
    for name, r in regions.items():
        # Clamp coordinates
        y1 = min(r['y'], h_img); y2 = min(r['y']+r['height'], h_img)
        x1 = min(r['x'], w_img); x2 = min(r['x']+r['width'], w_img)
        
        crop = full_img[y1:y2, x1:x2]
        
        if crop.size == 0:
            print(f"Region {name} is EMPTY! (Coords: {r})")
            continue
            
        print(f"Region {name} Size: {crop.shape}")

        # Preprocess (Invert)
        if ocr:
            processed = ocr.preprocess_image(crop)
        else:
            # Manual
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            inverted = cv2.bitwise_not(gray)
            processed = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR) # OpenCV uses BGR
        
        print(f"\n--- Region {name} ---")
        
        # Detection Logic
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Original Kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 12: continue
            bboxes.append((x, y, w, h))
            
        print(f"Found {len(bboxes)} raw boxes.")
        
        # Original Sort
        bboxes_sorted = sorted(bboxes, key=lambda b: (int(b[1] / 10), b[0]))
        print("BBoxes (Sorted Original - Scrambles?):")
        for b in bboxes_sorted:
            print(f"  {b} (y_bin={int(b[1]/10)})")

        # Proposed Sort (X only)
        bboxes_x = sorted(bboxes, key=lambda b: b[0])
        print("BBoxes (Sorted X-Only - Correct?):")
        for b in bboxes_x:
            print(f"  {b}")
            
        # Save Debug Image
        debug_img = processed.copy()
        for i, (x, y, w, h) in enumerate(bboxes_sorted):
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2) # Red
            cv2.putText(debug_img, f"{i}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        cv2.imwrite(f"debug_accuracy_{name}.png", debug_img)

if __name__ == "__main__":
    debug()
