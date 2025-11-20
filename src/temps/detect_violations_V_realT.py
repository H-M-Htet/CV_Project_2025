"""
Video violation detection script with REAL-TIME PREVIEW
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
        frame_skip: int = 1,
        show_realtime: bool = True  # NEW!
    ):
        """Process video file with real-time preview"""
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
        
        if show_realtime:
            print("\n▶️  REAL-TIME PREVIEW ENABLED")
            print("   Press 'q' to quit")
            print("   Press 'p' to pause/resume")
            print("   Press 's' to save current frame\n")
        
        # Prepare output video
        out = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Results
        violations_list = []
        frame_idx = 0
        paused = False
        
        # Create window for real-time display
        if show_realtime:
            cv2.namedWindow('Helmet Violation Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Helmet Violation Detection', 1280, 720)
        
        # Process frames
        print("Processing...\n")
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if needed
                if frame_idx % frame_skip != 0:
                    frame_idx += 1
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
                
                # Add info overlay
                info_text = f"Frame: {frame_idx}/{total_frames} | Violations: {len(violations)}"
                cv2.putText(result_bgr, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
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
            
            # Show real-time preview
            if show_realtime:
                cv2.imshow('Helmet Violation Detection', result_bgr)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n⏹️  Stopped by user")
                    break
                elif key == ord('p'):
                    paused = not paused
                    if paused:
                        print("⏸️  Paused (press 'p' to resume)")
                    else:
                        print("▶️  Resumed")
                elif key == ord('s'):
                    # Save current frame
                    save_path = Path('../results/manual_saves')
                    save_path.mkdir(parents=True, exist_ok=True)
                    filename = f'frame_{frame_idx}_{datetime.now().strftime("%H%M%S")}.jpg'
                    cv2.imwrite(str(save_path / filename), result_bgr)
                    print(f"💾 Saved: {filename}")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        if show_realtime:
            cv2.destroyAllWindows()
        
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
    parser.add_argument('--no-display', action='store_true', help='Disable real-time preview')
    
    args = parser.parse_args()
    
    # Initialize
    detector = VideoViolationDetector(args.model)
    
    # Process with real-time display
    detector.process_video(
        video_path=args.video,
        output_path=args.output,
        save_frames=args.save_frames,
        frame_skip=args.frame_skip,
        show_realtime=not args.no_display  # Show by default
    )

if __name__ == "__main__":
    main()