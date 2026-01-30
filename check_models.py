
from doctr.models import ocr_predictor
import sys

try:
    print("Testing MobileNetV3 Small...")
    model = ocr_predictor(det_arch='db_mobilenet_v3_small', reco_arch='crnn_mobilenet_v3_small', pretrained=True)
    print("SUCCESS: MobileNetV3 Small is valid.")
except Exception as e:
    print(f"FAILED: {e}")
