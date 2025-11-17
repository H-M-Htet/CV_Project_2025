"""
Common utility functions
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import torch

def load_image(image_path: str) -> np.ndarray:
    """Load image from path"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(image: np.ndarray, save_path: str):
    """Save image to path"""
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img_bgr)

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union between two boxes
    
    Args:
        box1, box2: [x1, y1, x2, y2] format
    
    Returns:
        IoU value (0 to 1)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0

def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Get center point of bounding box"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def point_in_bbox(point: Tuple[float, float], bbox: List[float]) -> bool:
    """Check if point is inside bounding box"""
    px, py = point
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def check_gpu():
    """Check GPU availability and print info"""
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return True
    else:
        print("⚠ No GPU detected - using CPU (will be slow)")
        return False

def draw_bbox(image: np.ndarray, bbox: List[int], label: str, 
              color: Tuple[int, int, int], confidence: float = None) -> np.ndarray:
    """
    Draw bounding box with label on image
    
    Args:
        image: Input image (RGB)
        bbox: [x1, y1, x2, y2]
        label: Text label
        color: RGB color tuple
        confidence: Optional confidence score
    
    Returns:
        Image with drawn box
    """
    img = image.copy()
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Prepare label text
    text = f"{label}"
    if confidence:
        text += f": {confidence:.2f}"
    
    # Calculate text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
    )
    
    # Draw background for text
    cv2.rectangle(
        img,
        (x1, y1 - text_height - baseline - 5),
        (x1 + text_width, y1),
        color,
        -1
    )
    
    # Draw text
    cv2.putText(
        img, text, (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 2
    )
    
    return img

class VideoProcessor:
    """Utility class for processing videos"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read next frame"""
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def release(self):
        """Release video capture"""
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

