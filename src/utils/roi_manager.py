"""
Region of Interest (ROI) Manager
Define and manage detection zones
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import log

class ROIManager:
    """
    Manage Region of Interest for counting and filtering detections
    """
    
    def __init__(self, roi_points: List[Tuple[int, int]] = None):
        """
        Initialize ROI Manager
        
        Args:
            roi_points: List of (x, y) tuples defining polygon
                       e.g., [(100, 200), (500, 200), (500, 600), (100, 600)]
        """
        if roi_points and len(roi_points) >= 3:
            self.roi_polygon = np.array(roi_points, dtype=np.int32)
            self.has_roi = True
            log.info(f"ROI defined with {len(roi_points)} points")
        else:
            self.roi_polygon = None
            self.has_roi = False
            log.info("No ROI defined - will count entire frame")
        
        # For counting line crossing (optional)
        self.counting_line = None
    
    def set_roi_interactive(self, first_frame: np.ndarray):
        """
        Interactively define ROI by clicking on frame
        
        Args:
            first_frame: First frame of video to draw on
        
        Returns:
            List of points selected
        """
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
                if len(points) > 1:
                    cv2.line(frame_copy, points[-2], points[-1], (0, 255, 0), 2)
                cv2.imshow('Define ROI', frame_copy)
        
        frame_copy = first_frame.copy()
        cv2.imshow('Define ROI', frame_copy)
        cv2.setMouseCallback('Define ROI', mouse_callback)
        
        log.info("Click to define ROI polygon. Press 'ENTER' when done, 'ESC' to cancel")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                break
            elif key == 27:  # Esc
                points = []
                break
        
        cv2.destroyAllWindows()
        
        if len(points) >= 3:
            self.roi_polygon = np.array(points, dtype=np.int32)
            self.has_roi = True
            log.info(f"ROI defined with {len(points)} points")
        
        return points
    
    def is_in_roi(self, bbox: List[float]) -> bool:
        """
        Check if bounding box center is inside ROI
        
        Args:
            bbox: [x1, y1, x2, y2]
        
        Returns:
            True if inside ROI (or no ROI defined)
        """
        if not self.has_roi:
            return True  # No ROI = count everything
        
        # Get center point of bbox
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Check if point is inside polygon
        result = cv2.pointPolygonTest(
            self.roi_polygon,
            (float(center_x), float(center_y)),
            False
        )
        
        return result >= 0
    
    def draw_roi(self, image: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255), 
                 thickness: int = 3) -> np.ndarray:
        """
        Draw ROI polygon on image
        
        Args:
            image: Input image
            color: ROI line color (B, G, R)
            thickness: Line thickness
        
        Returns:
            Image with ROI drawn
        """
        if not self.has_roi:
            return image
        
        result = image.copy()
        
        # Draw polygon
        cv2.polylines(result, [self.roi_polygon], True, color, thickness)
        
        # Draw label
        label_pos = tuple(self.roi_polygon[0])
        cv2.putText(result, "ROI", label_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        
        # Semi-transparent overlay
        overlay = result.copy()
        cv2.fillPoly(overlay, [self.roi_polygon], color)
        result = cv2.addWeighted(result, 0.9, overlay, 0.1, 0)
        
        return result
    
    def set_counting_line(self, point1: Tuple[int, int], point2: Tuple[int, int]):
        """
        Define a line for counting crossings
        
        Args:
            point1: (x, y) start point
            point2: (x, y) end point
        """
        self.counting_line = (point1, point2)
        log.info(f"Counting line set: {point1} -> {point2}")
    
    def draw_counting_line(self, image: np.ndarray) -> np.ndarray:
        """Draw counting line on image"""
        if self.counting_line is None:
            return image
        
        result = image.copy()
        cv2.line(result, self.counting_line[0], self.counting_line[1], 
                (255, 0, 255), 3)
        cv2.putText(result, "COUNT LINE", self.counting_line[0],
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        return result
    
    def get_roi_stats(self) -> Dict:
        """Get ROI statistics"""
        if not self.has_roi:
            return {'has_roi': False}
        
        # Calculate ROI area
        area = cv2.contourArea(self.roi_polygon)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(self.roi_polygon)
        
        return {
            'has_roi': True,
            'points': len(self.roi_polygon),
            'area': int(area),
            'bbox': [x, y, w, h]
        }


# Test function
if __name__ == "__main__":
    print("Testing ROI Manager...")
    
    # Test 1: Basic ROI
    roi_points = [(100, 200), (500, 200), (500, 600), (100, 600)]
    roi = ROIManager(roi_points)
    
    # Test bbox
    bbox_inside = [250, 300, 350, 400]
    bbox_outside = [50, 50, 80, 80]
    
    print(f"BBox {bbox_inside} in ROI: {roi.is_in_roi(bbox_inside)}")
    print(f"BBox {bbox_outside} in ROI: {roi.is_in_roi(bbox_outside)}")
    
    # Test stats
    print(f"ROI Stats: {roi.get_roi_stats()}")
    
    print("✓ ROI Manager test complete")
