import cv2
import numpy as np
import torch
import re
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union
import time
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from concurrent.futures import ThreadPoolExecutor
from config_manager import ConfigManager

class OCRProcessor:
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the OCR processor with docTR"""
        self.logger = logging.getLogger(__name__)
        self.config = config if config is not None else ConfigManager().data
        self.scale_factor = self.config.get('image_scale_factor', 1.0)
        self.min_confidence = self.config.get('min_ocr_confidence', 0.6)
        self.enable_binarization = self.config.get('enable_binarization', True)
        require_cuda = self.config.get('require_cuda', False)
        self.model = ocr_predictor(det_arch='db_mobilenet_v3_large', reco_arch='parseq', pretrained=True)
        if require_cuda and not torch.cuda.is_available():
            self.logger.error("CUDA required but not available. OCR cannot start.")
            raise RuntimeError("CUDA required but not available. Install CUDA-enabled PyTorch.")
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.model.eval()
            self.logger.info("Using GPU for OCR")
        else:
            self.model = self.model.cpu()
            self.model.eval()
            self.logger.info("Using CPU for OCR")
        self.inference_context = (
            torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        )
        self.debug_dir = Path('debug_images')
        self.debug_dir.mkdir(exist_ok=True)
        
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> Optional[np.ndarray]:
        """
        Enhanced image preprocessing for better text recognition.
        Now uses Grayscale -> Invert pipeline which works best for light text on dark background.
        """
        try:
            if image is None:
                self.logger.warning("Received None image for preprocessing; skipping.")
                return None

            if isinstance(image, str) and not image:
                self.logger.warning("Received empty image path for preprocessing; skipping.")
                return None

            if isinstance(image, np.ndarray) and image.size == 0:
                self.logger.warning("Received empty numpy array for preprocessing; skipping.")
                return None

            if isinstance(image, Image.Image) and (image.size[0] == 0 or image.size[1] == 0):
                self.logger.warning("Received empty PIL image for preprocessing; skipping.")
                return None

            # Convert input to numpy array
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
                image_np = np.array(image)
            elif isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image

            if image_np.size == 0:
                self.logger.warning("Received image with no data after conversion; skipping.")
                return None

            # Convert to RGB if needed
            if len(image_np.shape) == 2:  # Grayscale
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            elif len(image_np.shape) == 3 and image_np.shape[2] == 3:  # Already RGB
                pass
            else:
                raise ValueError(f"Unexpected image shape: {image_np.shape}")

            # Resize if scale factor is not 1.0 (currently used as 1.0)
            if self.scale_factor and self.scale_factor != 1.0:
                new_size = (
                    int(image_np.shape[1] * self.scale_factor),
                    int(image_np.shape[0] * self.scale_factor)
                )
                interpolation = cv2.INTER_AREA if self.scale_factor < 1.0 else cv2.INTER_LINEAR
                image_np = cv2.resize(image_np, new_size, interpolation=interpolation)

            # --- NEW PIPELINE: INVERSION ---
            # 1. Convert to Grayscale
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # 2. Invert (Light text/Dark BG -> Dark text/Light BG)
            inverted = cv2.bitwise_not(gray)
            
            # 3. Convert back to RGB for docTR
            processed = cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            return None

    def clean_text(self, text: str) -> str:
        """Clean and normalize OCR text output"""
        if not text:
            return ""
            
        # Remove unwanted characters while preserving essential punctuation
        return self.clean_text_basic(text)

    def clean_text_basic(self, text: str) -> str:
        """Basic text cleaning"""
        text = re.sub(r'[^a-zA-Z0-9\s\-.,!?\'"]', '', text)
        text = re.sub(r'(?<![\'\s])([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*-\s*', '-', text)
        text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)
        return text.strip()

    def clean_answer_text(self, text: str) -> str:
        """Aggressive cleaning for quiz answers - removes trailing OCR noise"""
        if not text:
            return ""
        
        # Stage 1: Basic cleaning (remove special chars, fix spacing)
        cleaned = self.clean_text_basic(text)
        
        # Stage 1.5: Remove leading garbage (common border artifacts)
        # Removes "2---", "1.", "- ", etc.
        cleaned = re.sub(r'^[\d\W_]+-{2,}', '', cleaned) # Remove 2---
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', cleaned)  # Remove "1. " or "1) "
        cleaned = re.sub(r'^[\-\.,\|\/\\]+\s*', '', cleaned) # Remove leading punctuation
        
        # Stage 2: For "Name, Number" pattern, extract only the valid part
        # Handles "Karkaroff, 4" OR "Maxime 7" (missing comma)
        match = re.search(r'^([^,]+)[,\s]+(-?\d+)', cleaned)
        if match:
            name_part = match.group(1).strip()
            number_part = match.group(2).strip()
            # Ensure name part isn't empty
            if name_part:
                return f"{name_part}, {number_part}"
        
        # Stage 3: Remove trailing garbage (isolated chars/punctuation at end)
        # CAUTION: Do NOT remove digits (e.g., "7")
        # Only remove short LETTER sequences (e.g. "V", "a", "-")
        cleaned = re.sub(r'\s+[a-zA-Z\-_]{1,2}$', '', cleaned)
        cleaned = re.sub(r'[\s\-,\.!?]+$', '', cleaned)
        
        return cleaned

    def recognize_text(self, image: Union[str, np.ndarray, Image.Image], min_confidence: float = 0.5) -> str:
        """Perform OCR on the image and return recognized text"""
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image)

            if processed_image is None:
                self.logger.warning("Skipping OCR because preprocessing returned no data.")
                return ""

            # Perform OCR using docTR
            with self.inference_context():
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        result = self.model([processed_image])
                else:
                    result = self.model([processed_image])
            
            # Extract text from all blocks
            text_parts = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        line_words = []
                        for word in line.words:
                            # Confidence Check: Filter out uncertain words
                            if word.confidence >= self.min_confidence:
                                line_words.append(word.value)
                        
                        line_text = " ".join(line_words)
                        if line_text.strip():  # Only add non-empty lines
                            text_parts.append(line_text)
            
            # Join and clean the text
            full_text = ' '.join(text_parts)
            cleaned_text = self.clean_text(full_text)
            
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"Error in OCR process: {str(e)}")
            return ""

    def _detect_words_opencv(self, image_np: np.ndarray, filter_edges: bool = False) -> List[np.ndarray]:
        """
        Fast word detection using OpenCV Morphological operations.
        Assumes 'image_np' is RGB, but text is Dark on Light (after Inversion in preprocess).
        """
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # 2. Threshold (Otsu)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. Morphological Dilation to connect letters into words
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # 4. Find Contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        crops = []
        bounding_boxes = [] # (x, y, w, h)
        
        img_h, img_w = image_np.shape[:2]
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter Noise
            # Min width 10, Min height 12 (filters dashed lines, small noise)
            if w < 10 or h < 12:
                continue
                
            # Filter Edges (Relaxed to avoid deleting valid text)
            if filter_edges:
                # Only remove artifacts touching the VERY edges (e.g. 1px border)
                # Previous logic (x < 15) was too aggressive for left-aligned text
                if x < 2 or (x + w) > (img_w - 2):
                    continue
                
            bounding_boxes.append((x, y, w, h))
            
        # 5. Sort Contours (Top-to-Bottom, then Left-to-Right)
        # Tolerance for lines: 10px
        bounding_boxes.sort(key=lambda b: (int(b[1] / 10), b[0]))
        
        # 6. Crop
        pad = 2
        for (x, y, w, h) in bounding_boxes:
            y1 = max(0, y - pad); y2 = min(img_h, y + h + pad)
            x1 = max(0, x - pad); x2 = min(img_w, x + w + pad)
            crop = image_np[y1:y2, x1:x2]
            crops.append(crop)
            
        return crops

    def process_quiz_regions(self, regions: Dict[str, np.ndarray]) -> Dict[str, str]:
        """
        Hybrid Processing Strategy:
        - Question: Use Robust Neural Detection (High accuracy for multi-line text).
        - Answers: Use Fast OpenCV Detection (High speed for single-line text).
        """
        results = {name: "" for name in regions.keys()}
        
        # 1. Preprocess All Images
        def _preprocess(item):
            name, img = item
            return name, self.preprocess_image(img)

        with ThreadPoolExecutor() as executor:
            processed_pairs = list(executor.map(_preprocess, regions.items()))
            
        processed_map = {name: img for name, img in processed_pairs if img is not None}
        
        # 2. Process QUESTION (Neural Pipeline - Robust)
        if 'question' in processed_map:
            try:
                q_img = processed_map['question']
                # Run full model (Detection + Recognition) on just the question
                # This ensures complex layout is handled correctly
                with self.inference_context():
                    if torch.cuda.is_available():
                        with torch.amp.autocast('cuda'):
                            q_result = self.model([q_img])
                    else:
                        q_result = self.model([q_img])
                
                # Extract text
                text_parts = []
                for page in q_result.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            line_words = [w.value for w in line.words if w.confidence >= self.min_confidence]
                            if line_words:
                                text_parts.append(" ".join(line_words))
                
                results['question'] = self.clean_text(" ".join(text_parts))
                
            except Exception as e:
                self.logger.error(f"Error processing question: {e}")

        # 3. Process ANSWERS (Fast OpenCV Pipeline)
        answer_keys = [k for k in processed_map.keys() if k != 'question']
        all_crops = []
        region_meta = [] # (name, count)
        
        for name in answer_keys:
            img = processed_map[name]
            # Use filter_edges=True to remove Checkmarks/Borders
            crops = self._detect_words_opencv(img, filter_edges=True)
            
            if not crops:
                region_meta.append((name, 0))
                continue
            
            all_crops.extend(crops)
            region_meta.append((name, len(crops)))
            
        # Batch Recognition for Answers
        if all_crops:
            try:
                with self.inference_context():
                    if torch.cuda.is_available():
                        with torch.amp.autocast('cuda'):
                            reco_out = self.model.reco_predictor(all_crops)
                    else:
                        reco_out = self.model.reco_predictor(all_crops)
                
                current_idx = 0
                for name, count in region_meta:
                    if count == 0: 
                        continue
                        
                    region_results = reco_out[current_idx : current_idx + count]
                    current_idx += count
                    
                    valid_words = [text for text, conf in region_results if conf >= self.min_confidence]
                    
                    full_text = " ".join(valid_words)
                    cleaned = self.clean_text(full_text)
                    cleaned = self.clean_answer_text(cleaned) # Specific answer cleaning
                    
                    results[name] = cleaned
                    
            except Exception as e:
                 self.logger.error(f"Error processing answers: {e}")

        return results

    def export_detection_torchscript(self, export_path: str = "det_model.ts") -> Optional[str]:
        """Export the detection sub-model to TorchScript for investigation."""
        try:
            det_model = self.model.det_predictor.model
            det_model.eval()
            device = next(det_model.parameters()).device
            example = torch.randn(1, 3, 512, 512, device=device)
            scripted = torch.jit.trace(det_model, example)
            scripted.save(export_path)
            self.logger.info(f"Detection model exported to {export_path}")
            return export_path
        except Exception as e:
            self.logger.error(f"Failed to export detection model: {e}")
            return None

    def benchmark_detection(self, export_path: str = "det_model.ts", runs: int = 10) -> Dict[str, float]:
        """Benchmark inference speed between PyTorch and TorchScript detection models."""
        det_model = self.model.det_predictor.model
        device = next(det_model.parameters()).device
        dummy = torch.randn(1, 3, 512, 512, device=device)

        det_model.eval()
        for _ in range(3):
            det_model(dummy)
        start = time.time()
        for _ in range(runs):
            det_model(dummy)
        pytorch_time = (time.time() - start) / runs

        script = torch.jit.load(export_path, map_location=device)
        script.eval()
        for _ in range(3):
            script(dummy)
        start = time.time()
        for _ in range(runs):
            script(dummy)
        script_time = (time.time() - start) / runs

        self.logger.info(
            f"Detection benchmark - PyTorch: {pytorch_time:.4f}s, TorchScript: {script_time:.4f}s"
        )
        return {"pytorch": pytorch_time, "torchscript": script_time}
