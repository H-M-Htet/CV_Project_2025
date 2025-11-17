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

class DetectionVisualizer:
    """
    Visualize detection results on images
    """
    
    def __init__(self):
        """Initialize visualizer with color scheme"""
        colors = config.get('visualization.colors', {
            'motorcycle': [0, 255, 0],
            'with_helmet': [0, 255, 255],
            'without_helmet': [255, 0, 0],
            'plate': [0, 0, 255]
        })
        
        self.colors = {
            'motorcycle': tuple(colors['motorcycle']),
            'with_helmet': tuple(colors['with_helmet']),
            'without_helmet': tuple(colors['without_helmet']),
            'plate': tuple(colors['plate'])
        }
        
        self.line_thickness = config.get('visualization.line_thickness', 2)
    
    def draw_detections(
        self,
        image: np.ndarray,
        motorcycles: List[Dict] = None,
        riders: List[Dict] = None,
        plates: List[Dict] = None,
        show_conf: bool = True
    ) -> np.ndarray:
        """
        Draw all detections on image
        
        Args:
            image: Input image (RGB)
            motorcycles: List of motorcycle detections
            riders: List of rider detections
            plates: List of plate detections
            show_conf: Whether to show confidence scores
        
        Returns:
            Image with drawn detections
        """
        img = image.copy()
        
        # Draw motorcycles
        if motorcycles:
            for moto in motorcycles:
                img = draw_bbox(
                    img,
                    moto['bbox'],
                    'Motorcycle',
                    self.colors['motorcycle'],
                    moto['conf'] if show_conf else None
                )
        
        # Draw riders
        if riders:
            for rider in riders:
                label = 'No Helmet' if rider['class'] == 0 else 'WITH HELMET'
                color = self.colors['without_helmet'] if rider['class'] == 0 else self.colors['with_helmet']
                
                img = draw_bbox(
                    img,
                    rider['bbox'],
                    label,
                    color,
                    rider['conf'] if show_conf else None
                )
        
        # Draw plates
        if plates:
            for plate in plates:
                img = draw_bbox(
                    img,
                    plate['bbox'],
                    'Plate',
                    self.colors['plate'],
                    plate['conf'] if show_conf else None
                )
        
        return img
    
    def draw_violations(
        self,
        image: np.ndarray,
        violations: List[Dict],
        show_association: bool = True
    ) -> np.ndarray:
        """
        Draw violation detections with emphasis
        
        Args:
            image: Input image (RGB)
            violations: List of violation dicts from associator
            show_association: Draw lines connecting associated objects
        
        Returns:
            Image with drawn violations
        """
        img = image.copy()
        
        for i, v in enumerate(violations):
            # Draw motorcycle (green)
            moto_bbox = [int(x) for x in v['motorcycle_bbox']]
            cv2.rectangle(
                img,
                (moto_bbox[0], moto_bbox[1]),
                (moto_bbox[2], moto_bbox[3]),
                self.colors['motorcycle'],
                self.line_thickness
            )
            
            # Draw rider (red - violation!)
            rider_bbox = [int(x) for x in v['rider_bbox']]
            cv2.rectangle(
                img,
                (rider_bbox[0], rider_bbox[1]),
                (rider_bbox[2], rider_bbox[3]),
                self.colors['without_helmet'],
                self.line_thickness + 1  # Thicker for emphasis
            )
            
            # Add violation label
            label = f"VIOLATION #{i+1}"
            cv2.putText(
                img,
                label,
                (rider_bbox[0], rider_bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.colors['without_helmet'],
                2
            )
            
            # Draw plate if available
            if v.get('plate_bbox') is not None:
                plate_bbox = [int(x) for x in v['plate_bbox']]
                cv2.rectangle(
                    img,
                    (plate_bbox[0], plate_bbox[1]),
                    (plate_bbox[2], plate_bbox[3]),
                    self.colors['plate'],
                    self.line_thickness
                )
                
                # Draw line connecting rider to plate
                if show_association:
                    rider_center = (
                        (rider_bbox[0] + rider_bbox[2]) // 2,
                        (rider_bbox[1] + rider_bbox[3]) // 2
                    )
                    plate_center = (
                        (plate_bbox[0] + plate_bbox[2]) // 2,
                        (plate_bbox[1] + plate_bbox[3]) // 2
                    )
                    cv2.line(
                        img,
                        rider_center,
                        plate_center,
                        self.colors['without_helmet'],
                        1,
                        cv2.LINE_AA
                    )
        
        # Add summary
        summary_text = f"Total Violations: {len(violations)}"
        cv2.putText(
            img,
            summary_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )
        
        return img
    
    def create_comparison_view(
        self,
        original: np.ndarray,
        detections: np.ndarray,
        violations: np.ndarray
    ) -> np.ndarray:
        """
        Create side-by-side comparison view
        
        Args:
            original: Original image
            detections: Image with all detections
            violations: Image with violations highlighted
        
        Returns:
            Horizontal concatenation of images
        """
        # Resize if needed to same height
        h = min(original.shape[0], detections.shape[0], violations.shape[0])
        
        orig_resized = cv2.resize(original, (int(original.shape[1] * h / original.shape[0]), h))
        det_resized = cv2.resize(detections, (int(detections.shape[1] * h / detections.shape[0]), h))
        viol_resized = cv2.resize(violations, (int(violations.shape[1] * h / violations.shape[0]), h))
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(orig_resized, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(det_resized, "Detections", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(viol_resized, "Violations", (10, 30), font, 1, (255, 255, 255), 2)
        
        # Concatenate horizontally
        comparison = np.hstack([orig_resized, det_resized, viol_resized])
        
        return comparison

