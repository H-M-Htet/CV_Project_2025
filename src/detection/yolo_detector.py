"""
YOLO detector wrapper for multi-model inference
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
    """
    
    def __init__(
        self,
        motorcycle_model_path: str = 'yolov8n.pt',
        helmet_model_path: Optional[str] = None,
        plate_model_path: Optional[str] = '..models/plate/best.pt',
        device: str = '0'
    ):
        """
        Initialize YOLO detector(s)
        
        Args:
            motorcycle_model_path: Path to motorcycle detection model (default: pre-trained)
            helmet_model_path: Path to helmet detection model (your trained model)
            plate_model_path: Path to plate detection model (optional)
            device: Device to use ('0' for GPU, 'cpu' for CPU)
        """
        self.device = device
        
        # Model 1: Motorcycle detection (pre-trained COCO)
        log.info(f"Loading motorcycle detector: {motorcycle_model_path}")
        self.motorcycle_model = YOLO(motorcycle_model_path)
        self.motorcycle_class_id = 3  # Motorcycle class in COCO
        
        # Model 2: Helmet detection (your custom model)
        self.helmet_model = None
        if helmet_model_path and Path(helmet_model_path).exists():
            log.info(f"Loading helmet detector: {helmet_model_path}")
            self.helmet_model = YOLO(helmet_model_path)
        else:
            log.warning("Helmet model not provided or not found")
        
        # Model 3: Plate detection (optional)
        self.plate_model = None
        if plate_model_path and Path(plate_model_path).exists():
            log.info(f"Loading plate detector: {plate_model_path}")
            self.plate_model = YOLO(plate_model_path)
        else:
            log.info("Plate model not provided - will skip plate detection")
        
        # Confidence thresholds
        self.conf_thresholds = config.get('detection.conf_threshold', {
            'motorcycle': 0.5,
            'helmet': 0.6,
            'plate': 0.5
        })
        
        log.info("YOLODetector initialized successfully")
    
    def detect_motorcycles(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Detect motorcycles in image
        
        Args:
            image: Input image (RGB)
            conf_threshold: Confidence threshold (overrides config)
        
        Returns:
            List of detection dicts with 'bbox', 'conf'
        """
        conf = conf_threshold or self.conf_thresholds.get('motorcycle', 0.5)
        
        results = self.motorcycle_model(image, conf=conf, verbose=False)[0]
        
        motorcycles = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == self.motorcycle_class_id:  # Only motorcycles
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
        Detect riders and their helmet status
        
        Args:
            image: Input image (RGB)
            conf_threshold: Confidence threshold
        
        Returns:
            List of detection dicts with 'bbox', 'class', 'conf'
            class: 0=with_helmet, 1=without_helmet
        """
        if self.helmet_model is None:
            log.error("Helmet model not loaded!")
            return []
        
        conf = conf_threshold or self.conf_thresholds.get('helmet', 0.6)
        
        results = self.helmet_model(image, conf=conf, verbose=False)[0]
        
        riders = []
        for box in results.boxes:
            riders.append({
                'bbox': box.xyxy[0].cpu().numpy().tolist(),
                'class': int(box.cls[0]),  # 0 or 1
                'conf': float(box.conf[0])
            })
        
        log.debug(f"Detected {len(riders)} riders")
        return riders
    
    def detect_plates(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Detect license plates in image
        
        Args:
            image: Input image (RGB)
            conf_threshold: Confidence threshold
        
        Returns:
            List of detection dicts with 'bbox', 'conf'
        """
        if self.plate_model is None:
            log.debug("Plate model not available, skipping")
            return []
        
        conf = conf_threshold or self.conf_thresholds.get('plate', 0.5)
        
        results = self.plate_model(image, conf=conf, verbose=False)[0]
        
        plates = []
        for box in results.boxes:
            plates.append({
                'bbox': box.xyxy[0].cpu().numpy().tolist(),
                'conf': float(box.conf[0])
            })
        
        log.debug(f"Detected {len(plates)} plates")
        return plates
    
    def detect_all(
        self,
        image: np.ndarray
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Run all detections on image
        
        Args:
            image: Input image (RGB)
        
        Returns:
            Tuple of (motorcycles, riders, plates)
        """
        motorcycles = self.detect_motorcycles(image)
        riders = self.detect_riders(image)
        plates = self.detect_plates(image)
        
        return motorcycles, riders, plates


class SingleModelDetector:
    """
    Alternative detector using single model for all classes
    Use this if your dataset has all 3 classes in one model
    """
    
    def __init__(self, model_path: str, device: str = '0'):
        """
        Initialize single-model detector
        
        Args:
            model_path: Path to YOLO model with all classes
            device: Device to use
        """
        log.info(f"Loading unified model: {model_path}")
        self.model = YOLO(model_path)
        self.device = device
        
        # Assuming class IDs (adjust based on your model)
        self.class_mapping = {
            'motorcycle': 0,
            'with_helmet': 1,
            'without_helmet': 2,
            'plate': 3  # If exists
        }
    
    def detect_all(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Detect all objects in one pass
        
        Returns:
            Tuple of (motorcycles, riders, plates)
        """
        results = self.model(image, conf=conf_threshold, verbose=False)[0]
        
        motorcycles = []
        riders = []
        plates = []
        
        for box in results.boxes:
            cls = int(box.cls[0])
            detection = {
                'bbox': box.xyxy[0].cpu().numpy().tolist(),
                'conf': float(box.conf[0])
            }
            
            if cls == self.class_mapping['motorcycle']:
                motorcycles.append(detection)
            elif cls == self.class_mapping['with_helmet']:
                detection['class'] = 0
                riders.append(detection)
            elif cls == self.class_mapping['without_helmet']:
                detection['class'] = 1
                riders.append(detection)
            elif cls == self.class_mapping.get('plate'):
                plates.append(detection)
        
        return motorcycles, riders, plates


# Test the detector
if __name__ == "__main__":
    import cv2
    
    # Test multi-model detector
    print("Testing YOLODetector...")
    detector = YOLODetector()
    
    # Create dummy image
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    motorcycles, riders, plates = detector.detect_all(test_img)
    
    print(f"✓ Detection test complete")
    print(f"  Motorcycles: {len(motorcycles)}")
    print(f"  Riders: {len(riders)}")
    print(f"  Plates: {len(plates)}")
