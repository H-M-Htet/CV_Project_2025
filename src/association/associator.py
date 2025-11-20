"""
3-Class Association - CORRECT MAPPING
Class 0: No_helmet
Class 1: Wearing_helmet
Class 2: Person_on_Bike
"""
import numpy as np
from typing import List, Dict
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import log

class ObjectAssociator:
    def __init__(self, iou_threshold: float = 0.3, max_distance: int = 300):
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        log.info(f"ObjectAssociator: IoU={iou_threshold}, MaxDist={max_distance}")
    
    def process_frame(self, motorcycles, riders, plates):
        """
        CORRECT MAPPING:
        0 = No_helmet → violation
        1 = Wearing_helmet → safe
        2 = Person_on_Bike → check for helmet
        """
        violations = []
        
        # Separate by CORRECT class IDs
        no_helmet = [r for r in riders if r['class'] == 0]
        wearing_helmet = [r for r in riders if r['class'] == 1]
        person_on_bike = [r for r in riders if r['class'] == 2]
        
        log.debug(f"No_helmet(0)={len(no_helmet)}, Wearing_helmet(1)={len(wearing_helmet)}, Person_on_Bike(2)={len(person_on_bike)}")
        
        # Rule 1: No_helmet (0) = violation
        for person in no_helmet:
            violation = {
                'rider_bbox': person['bbox'],
                'rider_conf': person['conf'],
                'rider_class': 0,
                'motorcycle_bbox': person['bbox'],
                'motorcycle_conf': person['conf']
            }
            violations.append(violation)
        
        # Rule 2: Person_on_Bike (2) without Wearing_helmet (1) = violation
        for person in person_on_bike:
            has_helmet = False
            for helmet in wearing_helmet:
                iou = self._calculate_iou(person['bbox'], helmet['bbox'])
                if iou > self.iou_threshold:
                    has_helmet = True
                    log.debug(f"Person_on_Bike has helmet (IoU={iou:.2f})")
                    break
            
            if not has_helmet:
                violation = {
                    'rider_bbox': person['bbox'],
                    'rider_conf': person['conf'],
                    'rider_class': 2,
                    'motorcycle_bbox': person['bbox'],
                    'motorcycle_conf': person['conf']
                }
                violations.append(violation)
                log.debug(f"VIOLATION: Person_on_Bike without helmet")
        
        # Add plates
        for v in violations:
            plate = self._find_nearest_plate(v['rider_bbox'], plates)
            if plate:
                v['plate_bbox'] = plate['bbox']
                v['plate_conf'] = plate['conf']
        
        log.info(f"Found {len(violations)} violations")
        return violations
    
    def _find_nearest_plate(self, bbox, plates):
        if not plates:
            return None
        center = self._get_center(bbox)
        best = None
        min_dist = float('inf')
        
        for plate in plates:
            pc = self._get_center(plate['bbox'])
            dist = np.sqrt((center[0]-pc[0])**2 + (center[1]-pc[1])**2)
            if dist < min_dist and dist < self.max_distance:
                min_dist = dist
                best = plate
        return best
    
    def _get_center(self, bbox):
        return ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
    
    def _calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2-x1) * (y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = area1 + area2 - intersection
        
        return intersection/union if union > 0 else 0.0