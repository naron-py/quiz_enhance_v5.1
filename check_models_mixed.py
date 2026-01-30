
from doctr.models import ocr_predictor
import sys

try:
    print("Testing Mixed MobileNetV3 (Large Det + Small Reco)...")
    # det_arch must be 'db_mobilenet_v3_large' as small failed
    model = ocr_predictor(det_arch='db_mobilenet_v3_large', reco_arch='crnn_mobilenet_v3_small', pretrained=True)
    print("SUCCESS: Mixed Architecture is valid.")
except Exception as e:
    print(f"FAILED: {e}")
