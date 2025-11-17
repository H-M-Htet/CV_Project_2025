"""
Association logic for connecting objects spatially
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.common import calculate_iou, get_bbox_center, point_in_bbox, euclidean_distance
from utils.logger import log

class ObjectAssociator:
    """
    Associates riders with motorcycles using spatial reasoning
    """
    
    def __init__(self, iou_threshold: float = 0.2, max_distance: float = 100):
        """
        Initialize associator
        
        Args:
            iou_threshold: Minimum IoU for association
            max_distance: Maximum distance (pixels) for association
        """
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        
        log.info(f"ObjectAssociator initialized: IoU={iou_threshold}, MaxDist={max_distance}")
    
    def associate_rider_with_motorcycle(
        self, 
        rider_bbox: List[float], 
        motorcycles: List[Dict]
    ) -> Optional[int]:
        """
        Find which motorcycle (if any) a rider is on
        
        Args:
            rider_bbox: [x1, y1, x2, y2] of rider
            motorcycles: List of dicts with 'bbox' and 'conf'
        
        Returns:
            Index of associated motorcycle, or None
        """
        if not motorcycles:
            return None
        
        rider_center = get_bbox_center(rider_bbox)
        
        # Method 1: Check if rider center is inside any motorcycle
        for i, moto in enumerate(motorcycles):
            if point_in_bbox(rider_center, moto['bbox']):
                log.debug(f"Rider center inside motorcycle {i} (point-in-box)")
                return i
        
        # Method 2: Check IoU overlap
        best_iou = 0
        best_idx = None
        
        for i, moto in enumerate(motorcycles):
            iou = calculate_iou(rider_bbox, moto['bbox'])
            if iou > self.iou_threshold and iou > best_iou:
                best_iou = iou
                best_idx = i
        
        if best_idx is not None:
            log.debug(f"Rider associated with motorcycle {best_idx} (IoU={best_iou:.3f})")
        
        return best_idx
    
    def find_nearest_plate(
        self,
        reference_bbox: List[float],
        plates: List[Dict],
        max_distance: Optional[float] = None
    ) -> Optional[int]:
        """
        Find nearest license plate to a reference bounding box
        
        Args:
            reference_bbox: Reference box (rider or motorcycle)
            plates: List of dicts with 'bbox' and 'conf'
            max_distance: Maximum distance to consider (uses self.max_distance if None)
        
        Returns:
            Index of nearest plate, or None
        """
        if not plates:
            return None
        
        max_dist = max_distance if max_distance is not None else self.max_distance
        
        ref_center = get_bbox_center(reference_bbox)
        
        nearest_idx = None
        min_distance = float('inf')
        
        for i, plate in enumerate(plates):
            plate_center = get_bbox_center(plate['bbox'])
            distance = euclidean_distance(ref_center, plate_center)
            
            if distance < min_distance and distance < max_dist:
                min_distance = distance
                nearest_idx = i
        
        if nearest_idx is not None:
            log.debug(f"Found plate {nearest_idx} at distance {min_distance:.1f}px")
        
        return nearest_idx
    
    def process_frame(
        self,
        motorcycles: List[Dict],
        riders: List[Dict],
        plates: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Process a frame to find violations using association
        
        Args:
            motorcycles: List of dicts with 'bbox', 'conf'
            riders: List of dicts with 'bbox', 'class', 'conf'
                   class: 0=WITHOUT_helmet, 1=WITH_helmet (YOUR MODEL'S MAPPING!)
            plates: Optional list of plate detections
        
        Returns:
            List of violation dictionaries with associated objects
        """
        violations = []
        
        log.debug(f"Processing: {len(motorcycles)} motorcycles, {len(riders)} riders")
        
        for rider_idx, rider in enumerate(riders):
            # Step 1: Associate rider with motorcycle
            moto_idx = self.associate_rider_with_motorcycle(
                rider['bbox'],
                motorcycles
            )
            
            # Step 2: Check if violation (on motorcycle + no helmet)
            if moto_idx is not None:  # Rider is ON a motorcycle
                # YOUR MODEL: 0=without_helmet, 1=with_helmet
                if rider['class'] == 0:  # without_helmet - VIOLATION!
                    violation = {
                        'rider_idx': rider_idx,
                        'rider_bbox': rider['bbox'],
                        'rider_conf': rider['conf'],
                        'motorcycle_idx': moto_idx,
                        'motorcycle_bbox': motorcycles[moto_idx]['bbox'],
                        'motorcycle_conf': motorcycles[moto_idx]['conf'],
                        'plate_idx': None,
                        'plate_bbox': None,
                        'plate_conf': None
                    }
                    
                    # Step 3: Find associated license plate (if available)
                    if plates:
                        # Look near the motorcycle
                        plate_idx = self.find_nearest_plate(
                            motorcycles[moto_idx]['bbox'],
                            plates
                        )
                        
                        if plate_idx is not None:
                            violation['plate_idx'] = plate_idx
                            violation['plate_bbox'] = plates[plate_idx]['bbox']
                            violation['plate_conf'] = plates[plate_idx]['conf']
                    
                    violations.append(violation)
                    log.info(f"VIOLATION detected: Rider {rider_idx} on Motorcycle {moto_idx}")
        
        log.info(f"Found {len(violations)} violations in frame")
        return violations
