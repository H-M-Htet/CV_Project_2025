"""
Video violation detection script
"""
import cv2
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime
from tqdm import tqdm
import argparse

sys.path.append(str(Path(__file__).parent))

from detection.yolo_detector import YOLODetector
from association.associator import ObjectAssociator
from detection.visualizer import DetectionVisualizer

class VideoViolationDetector:
    """Process videos for helmet violations"""
    
    def __init__(self, helmet_model_path: str):
        print("Initializing detector...")
        self.detector = YOLODetector(
            motorcycle_model_path='yolov8n.pt',
            helmet_model_path=helmet_model_path
        )
        self.associator = ObjectAssociator(iou_threshold=0.2, max_distance=100)
        self.visualizer = DetectionVisualizer()
        
    def process_video(
        self,
        video_path: str,
        output_path: str = None,
        save_frames: bool = False,
        frame_skip: int = 1
    ):
        """Process video file"""
        print(f"\nProcessing: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Prepare output video
        out = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Results
        violations_list = []
        frame_idx = 0
        
        # Process frames
        pbar = tqdm(total=total_frames, desc="Processing")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                pbar.update(1)
                continue
            
            # Process frame
            motorcycles, riders, _ = self.detector.detect_all(frame)
            violations = self.associator.process_frame(motorcycles, riders)
            
            # Visualize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if len(violations) > 0:
                result_rgb = self.visualizer.draw_violations(frame_rgb, violations)
            else:
                result_rgb = self.visualizer.draw_detections(frame_rgb, motorcycles, riders)
            
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            
            # Save violation info
            if len(violations) > 0:
                violations_list.append({
                    'frame': frame_idx,
                    'timestamp': frame_idx / fps,
                    'count': len(violations)
                })
                
                # Save frame if requested
                if save_frames:
                    frame_dir = Path('../results/violation_frames')
                    frame_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(frame_dir / f'violation_frame_{frame_idx:06d}.jpg'),
                        result_bgr
                    )
            
            # Write to output video
            if out:
                out.write(result_bgr)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        if out:
            out.release()
        
        # Summary
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Total frames processed: {frame_idx}")
        print(f"Frames with violations: {len(violations_list)}")
        print(f"Total violations: {sum(v['count'] for v in violations_list)}")
        if output_path:
            print(f"Output video: {output_path}")
        print(f"{'='*60}")
        
        # Save JSON summary
        summary = {
            'video': video_path,
            'frames_processed': frame_idx,
            'violation_frames': len(violations_list),
            'violations': violations_list
        }
        
        summary_path = Path('../results/violation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Summary saved: {summary_path}")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Helmet Violation Detection on Video')
    parser.add_argument('--model', required=True, help='Path to helmet detection model')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--output', help='Path to output video (optional)')
    parser.add_argument('--save-frames', action='store_true', help='Save violation frames')
    parser.add_argument('--frame-skip', type=int, default=1, help='Process every N frames')
    
    args = parser.parse_args()
    
    # Initialize
    detector = VideoViolationDetector(args.model)
    
    # Process
    detector.process_video(
        video_path=args.video,
        output_path=args.output,
        save_frames=args.save_frames,
        frame_skip=args.frame_skip
    )

if __name__ == "__main__":
    main()
