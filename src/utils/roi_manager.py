"""
Simplified ROI Manager
"""
import cv2
import numpy as np
from typing import List, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import log

class ROIManager:
    """Simple ROI for detection filtering"""
    
    def __init__(self, roi_points: List[Tuple[int, int]] = None):
        """
        Args:
            roi_points: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        if roi_points and len(roi_points) >= 3:
            self.roi_polygon = np.array(roi_points, dtype=np.int32)
            self.has_roi = True
            log.info(f"ROI: {len(roi_points)} points")
        else:
            self.roi_polygon = None
            self.has_roi = False
            log.info("No ROI - counting entire frame")
    
    def is_in_roi(self, bbox: List[float]) -> bool:
        """Check if bbox center is in ROI"""
        if not self.has_roi:
            return True
        
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        
        result = cv2.pointPolygonTest(
            self.roi_polygon,
            (float(cx), float(cy)),
            False
        )
        
        return result >= 0
    
    def draw_roi(self, image: np.ndarray) -> np.ndarray:
        """Draw ROI polygon"""
        if not self.has_roi:
            return image
        
        result = image.copy()
        
        # Draw polygon (cyan)
        cv2.polylines(result, [self.roi_polygon], True, (0, 255, 255), 3)
        
        # Label
        cv2.putText(result, "ROI", tuple(self.roi_polygon[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # # Semi-transparent fill
        # overlay = result.copy()
        # cv2.fillPoly(overlay, [self.roi_polygon], (0, 255, 255))
        # result = cv2.addWeighted(result, 0.92, overlay, 0.08, 0)
        
        return result
