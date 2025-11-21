"""
YOLO detector wrapper for multi-model inference
FIXED for correct 3-class mapping
"""
import torch
from ultralytics import YOLO
from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import log
from utils.config_loader import config

class YOLODetector:
    """
    Wrapper for YOLO detection with multiple models
    CORRECT 3-class mapping:
    Class 0: No_helmet
    Class 1: Wearing_helmet
    Class 2: Person_on_Bike
    """
    
    def __init__(
        self,
        motorcycle_model_path: str = 'yolov8n.pt',
        helmet_model_path: Optional[str] = '../models/yolo/best_S.pt',
        plate_model_path: Optional[str] = '../models/plate/best_P_0.1.pt',
        device: str = '0'
    ):
        """
        Initialize YOLO detector(s)
        
        Args:
            motorcycle_model_path: Path to motorcycle detection model (default: pre-trained)
            helmet_model_path: Path to 3-class helmet detection model
            plate_model_path: Path to plate detection model (optional)
            device: Device to use ('0' for GPU, 'cpu' for CPU)
        """
        self.device = device
        
        # Model 1: Motorcycle detection (pre-trained COCO)
        log.info(f"Loading motorcycle detector: {motorcycle_model_path}")
        self.motorcycle_model = YOLO(motorcycle_model_path)
        self.motorcycle_class_id = 3  # Motorcycle class in COCO
        
        # Model 2: Helmet detection (3-class model)
        self.helmet_model = None
        if helmet_model_path:
            # Handle wildcard paths
            helmet_path = Path(helmet_model_path)
            if '*' in str(helmet_path):
                search_dir = helmet_path.parent.parent
                matches = list(search_dir.glob(str(helmet_path.relative_to(search_dir))))
                if matches:
                    helmet_path = matches[0]
            
            if helmet_path.exists():
                log.info(f"Loading 3-class helmet detector: {helmet_path}")
                self.helmet_model = YOLO(str(helmet_path))
                # FIXED: Correct class mapping
                log.info("Classes: 0=No_helmet, 1=Wearing_helmet, 2=Person_on_Bike")
            else:
                log.warning(f"Helmet model not found: {helmet_model_path}")
        else:
            log.warning("Helmet model not provided")
        
        # Model 3: Plate detection (optional)
        self.plate_model = None
        if plate_model_path:
            plate_path = Path(plate_model_path)
            if '*' in str(plate_path):
                search_dir = plate_path.parent.parent
                matches = list(search_dir.glob(str(plate_path.relative_to(search_dir))))
                if matches:
                    plate_path = matches[0]
            
            if plate_path.exists():
                log.info(f"Loading plate detector: {plate_path}")
                self.plate_model = YOLO(str(plate_path))
            else:
                log.info("Plate model not found - will skip plate detection")
        else:
            log.info("Plate model not provided - will skip plate detection")
        
        # Confidence thresholds
        self.conf_thresholds = config.get('detection.conf_threshold', {
            'motorcycle': 0.5,
            'helmet': 0.4,
            'plate': 0.4
        })
        
        log.info("YOLODetector initialized successfully")
    
    def detect_motorcycles(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None
    ) -> List[Dict]:
        """Detect motorcycles in image"""
        conf = conf_threshold or self.conf_thresholds.get('motorcycle', 0.5)
        results = self.motorcycle_model(image, conf=conf, verbose=False)[0]
        
        motorcycles = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == self.motorcycle_class_id:
                motorcycles.append({
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                    'conf': float(box.conf[0])
                })
        
        log.debug(f"Detected {len(motorcycles)} motorcycles")
        return motorcycles
    
    def detect_riders(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Detect riders with 3-class helmet model
        
        Returns:
            List of detection dicts with 'bbox', 'class', 'conf'
            Classes (CORRECT):
                0 = No_helmet (violation)
                1 = Wearing_helmet (safe)
                2 = Person_on_Bike (full body - check for helmet)
        """
        if self.helmet_model is None:
            log.error("Helmet model not loaded!")
            return []
        
        conf = conf_threshold or self.conf_thresholds.get('helmet', 0.4)
        results = self.helmet_model(image, conf=conf, verbose=False)[0]
        
        riders = []
        for box in results.boxes:
            riders.append({
                'bbox': box.xyxy[0].cpu().numpy().tolist(),
                'class': int(box.cls[0]),  # 0, 1, or 2
                'conf': float(box.conf[0])
            })
        
        log.debug(f"Detected {len(riders)} rider detections (3 classes)")
        return riders
    
    def detect_plates(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
        motorcycle_bboxes: List[Dict] = None
    ) -> List[Dict]:
        """Detect license plates in image"""
        if self.plate_model is not None:
            conf = conf_threshold or self.conf_thresholds.get('plate', 0.4)
            results = self.plate_model(image, conf=conf, verbose=False)[0]
            
            plates = []
            for box in results.boxes:
                plates.append({
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                    'conf': float(box.conf[0])
                })
            
            log.debug(f"Detected {len(plates)} plates")
            return plates
        else:
            log.debug("No plate detection available")
            return []
    
    def detect_all(
        self,
        image: np.ndarray
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Run all detections on image
        
        Returns:
            Tuple of (motorcycles, riders, plates)
        """
        motorcycles = self.detect_motorcycles(image)
        riders = self.detect_riders(image)
        plates = self.detect_plates(image, motorcycle_bboxes=motorcycles)
        
        return motorcycles, riders, plates


# Test
if __name__ == "__main__":
    import cv2
    
    print("Testing YOLODetector (3-class)...")
    detector = YOLODetector()
    
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    motorcycles, riders, plates = detector.detect_all(test_img)
    
    print(f"✓ Detection test complete")
    print(f"  Motorcycles: {len(motorcycles)}")
    print(f"  Riders: {len(riders)}")
    print(f"  Plates: {len(plates)}")
    
    if riders:
        # FIXED: Correct class names order
        class_names = ['No_helmet', 'Wearing_helmet', 'Person_on_Bike']
        for i, r in enumerate(riders):
            print(f"    Rider {i+1}: {class_names[r['class']]} ({r['conf']:.2%})")
