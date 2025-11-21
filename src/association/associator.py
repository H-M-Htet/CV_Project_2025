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
        0 = No_helmet → violation (only if on motorcycle)
        1 = Wearing_helmet → safe
        2 = Person_on_Bike → check for helmet
        """
        violations = []
        
        # Separate by CORRECT class IDs
        no_helmet = [r for r in riders if r['class'] == 0]
        wearing_helmet = [r for r in riders if r['class'] == 1]
        person_on_bike = [r for r in riders if r['class'] == 2]
        
        log.debug(f"No_helmet(0)={len(no_helmet)}, Wearing_helmet(1)={len(wearing_helmet)}, Person_on_Bike(2)={len(person_on_bike)}")
        
        # Rule 1: No_helmet (0) = violation ONLY if near motorcycle
        for person in no_helmet:
            # Find nearest motorcycle
            moto = self._find_nearest_motorcycle(person['bbox'], motorcycles)
            
            if moto:
                # Check distance - must be close (not pedestrian)
                dist = self._calculate_distance(person['bbox'], moto['bbox'])
                if dist < 200:  # Within 200px = on motorcycle
                    violation = {
                        'rider_bbox': person['bbox'],
                        'rider_conf': person['conf'],
                        'rider_class': 0,
                        'motorcycle_bbox': moto['bbox'],
                        'motorcycle_conf': moto['conf']
                    }
                    violations.append(violation)
                else:
                    log.debug(f"Skipped pedestrian (dist={dist:.0f}px)")
        
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
    
    def _find_nearest_motorcycle(self, bbox, motorcycles):
        """Find nearest motorcycle to bbox"""
        if not motorcycles:
            return None
        
        center = self._get_center(bbox)
        best_moto = None
        min_dist = float('inf')
        
        for moto in motorcycles:
            moto_center = self._get_center(moto['bbox'])
            dist = np.sqrt((center[0]-moto_center[0])**2 + (center[1]-moto_center[1])**2)
            
            if dist < min_dist:
                min_dist = dist
                best_moto = moto
        
        return best_moto
    
    def _calculate_distance(self, bbox1, bbox2):
        """Calculate center-to-center distance"""
        c1 = self._get_center(bbox1)
        c2 = self._get_center(bbox2)
        return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
    
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