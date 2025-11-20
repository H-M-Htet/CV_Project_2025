"""
Visualization utilities for detection results
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.common import draw_bbox
from utils.config_loader import config
from utils.logger import log

class DetectionVisualizer:
    """
    Visualize detection results on images
    Updated for 3-class helmet model
    """
    
    def __init__(self):
        """Initialize visualizer with color scheme for 3 classes"""
        # CORRECT 3-class color scheme
        self.rider_colors = {
            0: [255, 0, 0],      # No_helmet → RED
            1: [0, 255, 0],      # Wearing_helmet → GREEN
            2: [255, 255, 0],    # Person_on_Bike → YELLOW
        }
        
        self.motorcycle_color = [100, 100, 100]  # Gray
        self.plate_color = [0, 0, 255]           # Blue
        
        # Box thickness
        self.thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        
        log.info("DetectionVisualizer initialized (3-class)")
    
    def draw_box(self, image, bbox, color, label=None, conf=None):
        """
        Draw single bounding box
        
        Args:
            image: Image array (RGB)
            bbox: [x1, y1, x2, y2]
            color: RGB color [r, g, b]
            label: Optional label text
            conf: Optional confidence score
        """
        # Clamp coordinates
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), w-1))
        y1 = max(0, min(int(y1), h-1))
        x2 = max(0, min(int(x2), w-1))
        y2 = max(0, min(int(y2), h-1))
        
        color = tuple(int(c) for c in color)
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.thickness)
        
        # Draw label if provided
        if label:
            text = f"{label}"
            if conf is not None:
                text += f" {conf:.2f}"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(text, self.font, self.font_scale, 1)
            y_text = max(text_h + 6, y1 - 6)
            
            cv2.rectangle(image, (x1, y_text - text_h - 4), 
                         (x1 + text_w, y_text), color, -1)
            cv2.putText(image, text, (x1, y_text - 2), self.font, 
                       self.font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    def draw_detections(
        self,
        image,
        motorcycles=None,
        riders=None,
        plates=None,
        draw_motorcycles=False
    ):
        """
        Draw all detections (3-class riders)
        
        Args:
            image: Image array (RGB)
            motorcycles: List of motorcycle detections (optional)
            riders: List of 3-class rider detections
            plates: List of plate detections
            draw_motorcycles: Whether to draw motorcycle boxes (default: False)
        
        Returns:
            Image with drawn boxes
        """
        result = image.copy()
        
        # Draw motorcycles (optional, gray color for context only)
        if motorcycles and draw_motorcycles:
            for moto in motorcycles:
                self.draw_box(result, moto['bbox'], self.motorcycle_color, 
                            'Motorcycle', moto['conf'])
        
        # Draw riders (3 classes with CORRECT names)
        if riders:
            # CORRECT ORDER matching your data.yaml!
            class_names = ['No_helmet', 'Wearing_helmet', 'Person_on_Bike']
            for rider in riders:
                color = self.rider_colors[rider['class']]
                label = class_names[rider['class']]
                self.draw_box(result, rider['bbox'], color, label, rider['conf'])
        
        # Draw plates
        if plates:
            for plate in plates:
                self.draw_box(result, plate['bbox'], self.plate_color, 
                            'Plate', plate['conf'])
        
        return result
    
    def draw_violations(
        self,
        image,
        violations,
        draw_motorcycles=False
    ):
        """
        Draw violations with emphasis
        
        Args:
            image: Image array (RGB)
            violations: List of violation dicts from associator
            draw_motorcycles: Whether to show motorcycle boxes
        
        Returns:
            Image with violations highlighted
        """
        result = image.copy()
        
        # CORRECT class names
        class_names = ['No_helmet', 'Wearing_helmet', 'Person_on_Bike']
        
        for i, violation in enumerate(violations):
            # Draw rider (RED for violations)
            rider_bbox = violation['rider_bbox']
            rider_class = violation.get('rider_class', 0)
            
            label = f"VIOLATION: {class_names[rider_class]}"
            
            self.draw_box(result, rider_bbox, [255, 0, 0], label, 
                         violation['rider_conf'])
            
            # Draw motorcycle (optional, gray)
            if draw_motorcycles and 'motorcycle_bbox' in violation:
                self.draw_box(result, violation['motorcycle_bbox'], 
                            self.motorcycle_color, 'Motorcycle', 
                            violation.get('motorcycle_conf'))
            
            # Draw plate if detected
            if 'plate_bbox' in violation:
                self.draw_box(result, violation['plate_bbox'], 
                            self.plate_color, 'Plate', 
                            violation.get('plate_conf'))
                
                # Draw plate text if available
                if 'plate_text' in violation:
                    plate_bbox = violation['plate_bbox']
                    x1, y1 = int(plate_bbox[0]), int(plate_bbox[3])
                    
                    text = f"Plate: {violation['plate_text']}"
                    cv2.putText(result, text, (x1, y1 + 20), self.font, 
                               0.7, [0, 255, 255], 2, cv2.LINE_AA)
        
        return result
