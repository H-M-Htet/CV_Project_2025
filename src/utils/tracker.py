"""
Motorcycle Tracker
Track unique motorcycles across frames and count crossings
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from pathlib import Path
import sys
import cv2

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import log

class MotorcycleTracker:
    """
    Track motorcycles across frames to count unique vehicles
    """
    
    def __init__(self, max_distance: float = 100, max_frames_missing: int = 30):
        """
        Initialize tracker
        
        Args:
            max_distance: Maximum distance (pixels) to match detections
            max_frames_missing: Max frames before removing lost track
        """
        self.tracks = {}  # {track_id: track_info}
        self.next_id = 0
        self.max_distance = max_distance
        self.max_frames_missing = max_frames_missing
        
        # Counting
        self.entered_roi = set()
        self.exited_roi = set()
        self.crossed_line = set()
        
        self.frame_count = 0
        
        log.info(f"Tracker initialized: MaxDist={max_distance}, MaxMissing={max_frames_missing}")
    
    def _calculate_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate distance between two bboxes (center-to-center)"""
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance
    
    def _get_bbox_area(self, bbox: List[float]) -> float:
        """Calculate bbox area"""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    def update(self, detections: List[Dict], frame_number: int = None):
        """
        Update tracks with new detections
        
        Args:
            detections: List of detection dicts with 'bbox', 'conf'
            frame_number: Current frame number
        
        Returns:
            List of tracked objects with 'track_id' added
        """
        if frame_number is not None:
            self.frame_count = frame_number
        else:
            self.frame_count += 1
        
        matched_tracks = set()
        tracked_detections = []
        
        # Match detections to existing tracks
        for detection in detections:
            best_match = None
            best_distance = float('inf')
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                
                distance = self._calculate_distance(detection['bbox'], track['bbox'])
                
                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_match = track_id
            
            # Update existing track or create new
            if best_match is not None:
                # Update existing track
                track_id = best_match
                self.tracks[track_id]['bbox'] = detection['bbox']
                self.tracks[track_id]['conf'] = detection['conf']
                self.tracks[track_id]['last_seen'] = self.frame_count
                self.tracks[track_id]['frames_tracked'] += 1
                matched_tracks.add(track_id)
            else:
                # Create new track
                track_id = self.next_id
                self.next_id += 1
                
                self.tracks[track_id] = {
                    'track_id': track_id,
                    'bbox': detection['bbox'],
                    'conf': detection['conf'],
                    'first_seen': self.frame_count,
                    'last_seen': self.frame_count,
                    'frames_tracked': 1,
                    'in_roi': False,
                    'crossed_line': False
                }
                
                log.debug(f"New track created: ID={track_id}")
            
            # Add track_id to detection
            detection['track_id'] = track_id
            tracked_detections.append(detection)
        
        # Remove old tracks (not seen for max_frames_missing)
        to_remove = []
        for track_id, track in self.tracks.items():
            if self.frame_count - track['last_seen'] > self.max_frames_missing:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            log.debug(f"Removing lost track: ID={track_id}")
            del self.tracks[track_id]
        
        return tracked_detections
    
    def update_roi_status(self, track_id: int, is_in_roi: bool):
        """
        Update whether track is in ROI
        
        Args:
            track_id: Track ID
            is_in_roi: Whether motorcycle is currently in ROI
        """
        if track_id not in self.tracks:
            return
        
        was_in_roi = self.tracks[track_id].get('in_roi', False)
        
        # Entering ROI
        if not was_in_roi and is_in_roi:
            self.entered_roi.add(track_id)
            log.debug(f"Track {track_id} entered ROI")
        
        # Exiting ROI
        elif was_in_roi and not is_in_roi:
            self.exited_roi.add(track_id)
            log.debug(f"Track {track_id} exited ROI")
        
        self.tracks[track_id]['in_roi'] = is_in_roi
    
    def check_line_crossing(self, track_id: int, line_start: Tuple[int, int], 
                           line_end: Tuple[int, int]) -> bool:
        """
        Check if track crossed counting line
        
        Args:
            track_id: Track ID
            line_start: Line start point (x, y)
            line_end: Line end point (x, y)
        
        Returns:
            True if crossed in this frame
        """
        if track_id not in self.tracks:
            return False
        
        track = self.tracks[track_id]
        
        # Get current center
        bbox = track['bbox']
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        # Simple line crossing detection
        # (This is basic - could be improved with direction detection)
        
        # Check if center is near the line
        # Calculate distance from point to line
        x1, y1 = line_start
        x2, y2 = line_end
        px, py = center
        
        # Line equation: distance from point to line
        num = abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1)
        den = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        
        if den == 0:
            return False
        
        distance = num / den
        
        # If within 20 pixels of line and not yet marked as crossed
        if distance < 20 and track_id not in self.crossed_line:
            self.crossed_line.add(track_id)
            track['crossed_line'] = True
            log.info(f"Track {track_id} crossed counting line")
            return True
        
        return False
    
    def get_statistics(self) -> Dict:
        """
        Get tracking statistics
        
        Returns:
            Dict with statistics
        """
        active_tracks = len(self.tracks)
        total_tracks = self.next_id
        
        return {
            'active_tracks': active_tracks,
            'total_unique_motorcycles': total_tracks,
            'entered_roi': len(self.entered_roi),
            'exited_roi': len(self.exited_roi),
            'crossed_line': len(self.crossed_line),
            'current_frame': self.frame_count
        }
    
    def draw_tracks(self, image: np.ndarray) -> np.ndarray:
        """
        Draw track IDs on image
        
        Args:
            image: Input image
        
        Returns:
            Image with tracks drawn
        """
        result = image.copy()
        
        for track_id, track in self.tracks.items():
            bbox = track['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw track ID
            label = f"ID:{track_id}"
            cv2.putText(result, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Draw small circle at center
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(result, center, 5, (255, 255, 0), -1)
        
        return result


# Test function
if __name__ == "__main__":
    print("Testing Motorcycle Tracker...")
    
    tracker = MotorcycleTracker()
    
    # Simulate detections across frames
    frame1_detections = [
        {'bbox': [100, 100, 200, 200], 'conf': 0.9},
        {'bbox': [300, 150, 400, 250], 'conf': 0.85}
    ]
    
    frame2_detections = [
        {'bbox': [110, 105, 210, 205], 'conf': 0.88},  # Same as first
        {'bbox': [305, 155, 405, 255], 'conf': 0.87},  # Same as second
        {'bbox': [500, 300, 600, 400], 'conf': 0.92}   # New!
    ]
    
    tracked1 = tracker.update(frame1_detections, frame_number=1)
    print(f"Frame 1: {len(tracked1)} tracked")
    
    tracked2 = tracker.update(frame2_detections, frame_number=2)
    print(f"Frame 2: {len(tracked2)} tracked")
    
    stats = tracker.get_statistics()
    print(f"Stats: {stats}")
    
    print("✓ Tracker test complete")
