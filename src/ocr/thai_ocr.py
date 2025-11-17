"""
Thai License Plate OCR using EasyOCR
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
    OCR for Thai license plates
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize EasyOCR reader
        
        Args:
            use_gpu: Use GPU acceleration if available
        """
        log.info("Initializing EasyOCR for Thai plates...")
        
        try:
            # Initialize reader for Thai and English
            self.reader = easyocr.Reader(
                ['th', 'en'],  # Thai and English
                gpu=use_gpu
            )
            log.info("✓ EasyOCR initialized successfully")
            
        except Exception as e:
            log.error(f"Failed to initialize EasyOCR: {e}")
            log.info("Trying without GPU...")
            self.reader = easyocr.Reader(['th', 'en'], gpu=False)
        
        # Thai plate patterns (for validation)
        self.confidence_threshold = 0.3
    
    def preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Preprocess plate image for better OCR
        
        Args:
            plate_img: Cropped plate image (RGB)
        
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = plate_img
        
        # Resize if too small
        h, w = gray.shape
        if h < 50 or w < 100:
            scale = max(50/h, 100/w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        
        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def extract_text(
        self,
        image: np.ndarray,
        bbox: List[float],
        preprocess: bool = True
    ) -> Optional[Dict]:
        """
        Extract text from license plate region
        
        Args:
            image: Full image (RGB)
            bbox: Plate bounding box [x1, y1, x2, y2]
            preprocess: Apply preprocessing
        
        Returns:
            Dict with 'text', 'confidence', 'details' or None
        """
        try:
            # Crop plate region
            x1, y1, x2, y2 = map(int, bbox)
            
            # Add padding
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            plate_img = image[y1:y2, x1:x2].copy()
            
            if plate_img.size == 0:
                log.warning("Empty plate crop")
                return None
            
            # Preprocess if requested
            if preprocess:
                plate_img = self.preprocess_plate(plate_img)
            
            # Run OCR
            results = self.reader.readtext(plate_img)
            
            if not results:
                log.debug("No text detected in plate")
                return None
            
            # Combine all detected text
            texts = []
            confidences = []
            
            for (bbox, text, conf) in results:
                if conf >= self.confidence_threshold:
                    texts.append(text)
                    confidences.append(conf)
            
            if not texts:
                return None
            
            # Combine text
            combined_text = ' '.join(texts)
            avg_confidence = np.mean(confidences)
            
            result = {
                'text': combined_text,
                'confidence': float(avg_confidence),
                'raw_results': results,
                'num_detections': len(texts)
            }
            
            log.info(f"OCR Result: '{combined_text}' (conf: {avg_confidence:.2f})")
            
            return result
            
        except Exception as e:
            log.error(f"OCR extraction failed: {e}")
            return None
    
    def clean_plate_text(self, text: str) -> str:
        """
        Clean and format plate text
        
        Args:
            text: Raw OCR text
        
        Returns:
            Cleaned text
        """
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Convert to uppercase
        text = text.upper()
        
        # Remove special characters (keep Thai, English, numbers)
        # Thai characters: \u0E00-\u0E7F
        # You can add more cleaning rules here
        
        return text
    
    def validate_thai_plate(self, text: str) -> bool:
        """
        Basic validation of Thai plate format
        
        Args:
            text: Plate text
        
        Returns:
            True if looks like valid Thai plate
        """
        # Thai plates typically have:
        # - Thai characters (province)
        # - Numbers
        # - Mix of both
        
        if not text or len(text) < 3:
            return False
        
        # Check if contains both letters and numbers
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        
        return has_letter or has_number


# Test function
if __name__ == "__main__":
    print("Testing Thai Plate OCR...")
    
    # Initialize
    ocr = ThaiPlateOCR(use_gpu=True)
    
    # Test on a sample image
    test_images = list(Path('../../data/helmet_dataset/valid/images').glob('*'))
    
    if test_images:
        img = cv2.imread(str(test_images[0]))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Simulate plate bbox (you'd get this from detection)
        h, w = img.shape[:2]
        fake_bbox = [w//4, 3*h//4, 3*w//4, h-10]
        
        result = ocr.extract_text(img_rgb, fake_bbox)
        
        if result:
            print(f"\n✓ OCR Test Successful!")
            print(f"  Text: {result['text']}")
            print(f"  Confidence: {result['confidence']:.2f}")
        else:
            print("\n⚠️  No text detected (might be no plate in test region)")
    
    print("\n✓ OCR module ready!")
