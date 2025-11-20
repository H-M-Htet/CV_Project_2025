"""
Thai License Plate OCR using EasyOCR
Enhanced with crop saving
"""
import easyocr
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import log

class ThaiPlateOCR:
    """
    OCR for Thai license plates with crop saving
    """
    
    def __init__(self, use_gpu: bool = True, save_crops: bool = False, crops_dir: str = None):
        """
        Initialize EasyOCR reader
        
        Args:
            use_gpu: Use GPU acceleration if available
            save_crops: Whether to save cropped plates
            crops_dir: Directory to save crops
        """
        log.info("Initializing EasyOCR for Thai plates...")
        
        try:
            self.reader = easyocr.Reader(['th', 'en'], gpu=use_gpu)
            log.info("✓ EasyOCR initialized successfully")
        except Exception as e:
            log.error(f"Failed to initialize EasyOCR: {e}")
            log.info("Trying without GPU...")
            self.reader = easyocr.Reader(['th', 'en'], gpu=False)
        
        self.confidence_threshold = 0.3
        self.save_crops = save_crops
        self.crop_counter = 0
        
        # Setup crops directory
        if save_crops and crops_dir:
            self.crops_dir = Path(crops_dir)
            self.crops_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Plate crops will be saved to: {self.crops_dir}")
        else:
            self.crops_dir = None
    
    def preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Preprocess plate image for better OCR
        """
        img = plate_img.copy()
        
        # Convert to grayscale if needed
        if img.ndim == 3:
            # FIXED: Handle RGB input correctly
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Upscale if too small
        h, w = gray.shape[:2]
        min_h, min_w = 80, 200
        scale = max(1.0, min_h / float(h), min_w / float(w))
        if scale > 1.0:
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), 
                             interpolation=cv2.INTER_CUBIC)
        
        # Enhance contrast (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        return gray
    
    def extract_text(
        self,
        image: np.ndarray,
        bbox: List[float],
        preprocess: bool = True,
        frame_id: int = None,
        violation_id: int = None
    ) -> Optional[Dict]:
        """
        Extract text from license plate region
        
        Args:
            image: Full image (RGB)
            bbox: Plate bounding box [x1, y1, x2, y2]
            preprocess: Apply preprocessing
            frame_id: Frame number (for saving crops)
            violation_id: Violation ID (for saving crops)
        
        Returns:
            Dict with 'text', 'confidence', 'crop_path' or None
        """
        try:
            # Crop plate region
            x1, y1, x2, y2 = map(int, bbox)
            
            # Add padding
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            plate_crop = image[y1:y2, x1:x2].copy()
            
            if plate_crop.size == 0:
                log.warning("Empty plate crop")
                return None
            
            # Save original crop if requested
            crop_path = None
            if self.save_crops and self.crops_dir:
                self.crop_counter += 1
                if frame_id is not None:
                    filename = f"frame{frame_id:06d}_v{violation_id}_plate.jpg"
                else:
                    filename = f"plate_{self.crop_counter:06d}.jpg"
                
                crop_path = self.crops_dir / filename
                
                # Save as BGR for cv2.imwrite
                plate_bgr = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(crop_path), plate_bgr)
                log.debug(f"Saved plate crop: {crop_path}")
            
            # Preprocess for OCR
            if preprocess:
                plate_processed = self.preprocess_plate(plate_crop)
            else:
                plate_processed = plate_crop
            
            # Run OCR
            results = self.reader.readtext(plate_processed)
            
            if not results:
                log.debug("No text detected in plate")
                return {
                    'text': 'N/A',
                    'confidence': 0.0,
                    'crop_path': str(crop_path) if crop_path else None,
                    'raw_results': []
                }
            
            # Combine all detected text
            texts = []
            confidences = []
            
            for (bbox_ocr, text, conf) in results:
                if conf >= self.confidence_threshold:
                    texts.append(text)
                    confidences.append(conf)
            
            if not texts:
                return {
                    'text': 'Low_Confidence',
                    'confidence': 0.0,
                    'crop_path': str(crop_path) if crop_path else None,
                    'raw_results': results
                }
            
            # Combine text
            combined_text = ' '.join(texts)
            avg_confidence = np.mean(confidences)
            
            result = {
                'text': combined_text,
                'confidence': float(avg_confidence),
                'crop_path': str(crop_path) if crop_path else None,
                'raw_results': results,
                'num_detections': len(texts)
            }
            
            log.info(f"OCR: '{combined_text}' (conf: {avg_confidence:.2%})")
            
            return result
            
        except Exception as e:
            log.error(f"OCR extraction failed: {e}")
            return None
    
    def clean_plate_text(self, text: str) -> str:
        """Clean and format plate text"""
        text = ' '.join(text.split())
        text = text.upper()
        return text
