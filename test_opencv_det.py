import cv2
import numpy as np
import torch
from doctr.models import ocr_predictor
import os

def test_opencv_strategy():
    image_path = "debug_crops/B_crop.png"
    if not os.path.exists(image_path):
        print("Image not found")
        return

    # 1. Preprocess (Invert so text is Black)
    # Actually, for contour detection, we usually want White Text on Black BG.
    # Original image: Light text on Dark BG.
    # So we want to keep it "Light on Dark"?
    # No, usually logical binary is White=Object, Black=Background.
    # Text is the object.
    # Text is Light. BG is Dark.
    # So converting to Grayscale is enough?
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Threshold
    # Since text is Light, we use simple threshold or Otsu
    # But wait, original problem was faint "4".
    # Inversion helped Reco.
    # Does Inversion help Detection?
    # If text is faint light, Inversion makes it faint dark.
    # Thresholding should find it if contrast exists.
    
    # Let's try Otsu on the grayscale (assuming Light Text)
    # Threshold: values > thresh are White (Text).
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    cv2.imwrite("debug_opencv_binary.png", binary)
    
    # 3. Morphological Dilation to connect letters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3)) # Wide kernel
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    cv2.imwrite("debug_opencv_dilated.png", dilated)
    
    # 4. Find Contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Filter and Extract Crops
    crops = []
    coords = []
    
    vis = img.copy()
    
    # Load Reco Model (for verification)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ocr_predictor(det_arch='db_mobilenet_v3_large', reco_arch='parseq', pretrained=True)
    reco_model = model.reco_predictor.to(device).eval()
    
    print(f"Found {len(contours)} contours")
    
    # Filter contours
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter noise
        if w < 10 or h < 10:
            continue
            
        # Optional: Aspect ratio filter?
        
        valid_contours.append((x, y, w, h))
        
    # Sort left to right
    valid_contours.sort(key=lambda c: c[0])
    
    for (x, y, w, h) in valid_contours:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Crop from ORIGINAL image (or Preprocessed?)
        # Reco needs "Black Text on White BG" (Inverted applied to original)
        # So we should crop from the INVERTED RGB buffer.
        
        # Let's prepare Inverted RGB
        inverted_gray = cv2.bitwise_not(gray)
        processed_rgb = cv2.cvtColor(inverted_gray, cv2.COLOR_GRAY2RGB)
        
        # Crop
        # Add padding? Text detectors usually give tight crops.
        # Let's add small padding.
        pad = 2
        H, W = processed_rgb.shape[:2]
        y1 = max(0, y - pad); y2 = min(H, y + h + pad)
        x1 = max(0, x - pad); x2 = min(W, x + w + pad)
        
        crop = processed_rgb[y1:y2, x1:x2]
        crops.append(crop)
        coords.append((x, y, w, h))
        
    cv2.imwrite("debug_opencv_vis.png", vis)
    
    if not crops:
        print("No crops found!")
        return
        
    # 6. Recognize
    print(f"Recognizing {len(crops)} words...")
    with torch.no_grad():
        results = reco_model(crops)
        
    full_text = []
    for (text, conf), (x,y,w,h) in zip(results, coords):
        print(f"Rect [{x},{y},{w},{h}] -> '{text}' ({conf:.2f})")
        full_text.append(text)
        
    print(f"Full Text: {' '.join(full_text)}")

if __name__ == "__main__":
    test_opencv_strategy()
