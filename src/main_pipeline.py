"""
COMPLETE VIOLATION DETECTION PIPELINE
Features:
- ROI-based counting
- Motorcycle tracking (unique IDs)
- Violation detection
- OCR on plates
- CSV logging
- Statistics
"""
import cv2
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple

from detection.yolo_detector import YOLODetector
from association.associator import ObjectAssociator
from detection.visualizer import DetectionVisualizer
from ocr.thai_ocr import ThaiPlateOCR
from violation_logger import ViolationLogger
from utils.roi_manager import ROIManager
from utils.tracker import MotorcycleTracker
from utils.logger import log

class ViolationDetectionSystem:
    """
    Complete violation detection system with ROI and tracking
    """
    
    def __init__(
        self,
        helmet_model: str,
        plate_model: str,
        output_dir: str,
        use_gpu: bool = True,
        roi_points: List[Tuple] = None
    ):
        """Initialize system"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        log.info("="*70)
        log.info("Initializing Violation Detection System...")
        log.info("="*70)
        
        # Detection models
        self.detector = YOLODetector(
            motorcycle_model_path='yolov8n.pt',
            helmet_model_path=helmet_model,
            plate_model_path=plate_model
        )
        
        self.associator = ObjectAssociator()
        self.visualizer = DetectionVisualizer()
        
        # OCR with crop saving
        crops_dir = self.output_dir / 'plate_crops'
        self.ocr = ThaiPlateOCR(use_gpu=use_gpu, save_crops=True, crops_dir=str(crops_dir))
        
        # CSV logger
        csv_path = self.output_dir / 'violations.csv'
        self.logger = ViolationLogger(str(csv_path))
        
        # ROI Manager
        self.roi_manager = ROIManager(roi_points)
        
        # Motorcycle Tracker
        self.tracker = MotorcycleTracker(max_distance=100, max_frames_missing=30)
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_motorcycles_detected': 0,
            'unique_motorcycles': 0,
            'motorcycles_in_roi': 0,
            'total_violations': 0,
            'violations_in_roi': 0,
            'safe_riders': 0,
            'plates_detected': 0,
            'plates_with_text': 0,
            'entered_roi': 0,
            'exited_roi': 0
        }
        
        log.info("✓ All components initialized")
        log.info(f"✓ ROI defined: {self.roi_manager.has_roi}")
        log.info("="*70)
    
    def process_frame(self, frame, frame_number, timestamp):
        """
        Process single frame with tracking and ROI
        
        Args:
            frame: Video frame (BGR)
            frame_number: Frame number
            timestamp: Timestamp in seconds
        
        Returns:
            Annotated frame (BGR)
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Step 1: Detection
        motorcycles, riders, plates = self.detector.detect_all(img_rgb)
        self.stats['total_motorcycles_detected'] += len(motorcycles)
        
        # Step 2: Track motorcycles (assign unique IDs)
        tracked_motorcycles = self.tracker.update(motorcycles, frame_number)
        
        # Step 3: Update ROI status for each track
        motorcycles_in_roi_count = 0
        for moto in tracked_motorcycles:
            is_in_roi = self.roi_manager.is_in_roi(moto['bbox'])
            self.tracker.update_roi_status(moto['track_id'], is_in_roi)
            
            if is_in_roi:
                motorcycles_in_roi_count += 1
        
        self.stats['motorcycles_in_roi'] += motorcycles_in_roi_count
        
        # Step 4: Find violations
        violations = self.associator.process_frame(motorcycles, riders, plates)
        self.stats['total_violations'] += len(violations)
        
        # Step 5: Filter violations by ROI
        violations_in_roi = []
        for v in violations:
            if self.roi_manager.is_in_roi(v['rider_bbox']):
                violations_in_roi.append(v)
        
        self.stats['violations_in_roi'] += len(violations_in_roi)
        
        # Step 6: Count safe riders
        safe_riders = [r for r in riders if r['class'] == 1]  # Wearing_helmet
        self.stats['safe_riders'] += len(safe_riders)
        
        # Step 7: OCR on violations in ROI
        for i, v in enumerate(violations_in_roi):
            if 'plate_bbox' in v:
                self.stats['plates_detected'] += 1
                
                result = self.ocr.extract_text(
                    img_rgb,
                    v['plate_bbox'],
                    preprocess=True,
                    frame_id=frame_number,
                    violation_id=i
                )
                
                if result:
                    v['plate_text'] = result['text']
                    v['plate_ocr_conf'] = result.get('confidence', 0)
                    v['plate_crop_path'] = result.get('crop_path', '')
                    
                    if result['text'] not in ['N/A', 'Low_Confidence']:
                        self.stats['plates_with_text'] += 1
                else:
                    v['plate_text'] = 'N/A'
                    v['plate_ocr_conf'] = 0.0
            
            # Log violation to CSV
            self.logger.log_violation(frame_number, timestamp, v)
        
        # Step 8: Visualize
        result_img = self.visualizer.draw_violations(img_rgb, violations, draw_motorcycles=False)
        
        # Step 9: Draw ROI
        result_img = self.roi_manager.draw_roi(result_img)
        
        # Step 10: Draw track IDs
        result_img = self.tracker.draw_tracks(result_img)
        
        # Step 11: Draw statistics on frame
        tracker_stats = self.tracker.get_statistics()
        
        stats_text = [
            f"Frame: {frame_number}",
            f"Tracked: {tracker_stats['active_tracks']} | Total: {tracker_stats['total_unique_motorcycles']}",
            f"In ROI: {motorcycles_in_roi_count}",
            f"Violations: {len(violations_in_roi)}",
            f"Safe: {len(safe_riders)}"
        ]
        
        y_offset = 30
        for text in stats_text:
            # Background
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(result_img, (5, y_offset - 25), (15 + text_w, y_offset + 5), 
                         (0, 0, 0), -1)
            # Text
            cv2.putText(result_img, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 35
        
        return cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    
    def process_video(self, video_path: str, output_video: str = None):
        """
        Process entire video
        
        Args:
            video_path: Path to input video
            output_video: Path to output video (optional)
        """
        log.info("="*70)
        log.info(f"Processing video: {video_path}")
        log.info("="*70)
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            log.error(f"Failed to open video: {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.stats['total_frames'] = total_frames
        
        log.info(f"Video info: {w}x{h} @ {fps}fps, {total_frames} frames")
        
        # Output video
        if output_video is None:
            output_video = str(self.output_dir / 'output_video.mp4')
        
        writer = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (w, h)
        )
        
        frame_count = 0
        
        log.info("Starting processing...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps
            
            # Process frame
            result = self.process_frame(frame, frame_count, timestamp)
            writer.write(result)
            cv2.imshow('Detection', result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            # Progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                log.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        writer.release()
        
        # Get final tracker stats
        tracker_stats = self.tracker.get_statistics()
        self.stats['unique_motorcycles'] = tracker_stats['total_unique_motorcycles']
        self.stats['entered_roi'] = tracker_stats['entered_roi']
        self.stats['exited_roi'] = tracker_stats['exited_roi']
        
        # Save summary
        self.save_summary()
        
        log.info("="*70)
        log.info("✓ Processing complete!")
        log.info("="*70)
        log.info(f"Output video: {output_video}")
        log.info(f"Results directory: {self.output_dir}")
        log.info("="*70)
    
    def save_summary(self):
        """Save summary statistics"""
        summary_path = self.output_dir / 'summary.txt'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("VIOLATION DETECTION SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write("=== VIDEO INFO ===\n")
            f.write(f"Total Frames: {self.stats['total_frames']}\n\n")
            
            f.write("=== MOTORCYCLE TRACKING ===\n")
            f.write(f"Total Detections: {self.stats['total_motorcycles_detected']}\n")
            f.write(f"Unique Motorcycles (Tracked): {self.stats['unique_motorcycles']}\n")
            f.write(f"Motorcycles in ROI: {self.stats['motorcycles_in_roi']}\n")
            f.write(f"Entered ROI: {self.stats['entered_roi']}\n")
            f.write(f"Exited ROI: {self.stats['exited_roi']}\n\n")
            
            f.write("=== VIOLATIONS ===\n")
            f.write(f"Total Violations: {self.stats['total_violations']}\n")
            f.write(f"Violations in ROI: {self.stats['violations_in_roi']}\n")
            f.write(f"Safe Riders: {self.stats['safe_riders']}\n\n")
            
            f.write("=== LICENSE PLATES ===\n")
            f.write(f"Plates Detected: {self.stats['plates_detected']}\n")
            f.write(f"Plates with OCR Text: {self.stats['plates_with_text']}\n\n")
            
            f.write("=== RATES ===\n")
            if self.stats['motorcycles_in_roi'] > 0:
                violation_rate = (self.stats['violations_in_roi'] / 
                                self.stats['motorcycles_in_roi'] * 100)
                f.write(f"Violation Rate (ROI): {violation_rate:.1f}%\n")
            
            if self.stats['plates_detected'] > 0:
                ocr_success_rate = (self.stats['plates_with_text'] / 
                                   self.stats['plates_detected'] * 100)
                f.write(f"OCR Success Rate: {ocr_success_rate:.1f}%\n")
            
            f.write("\n" + "="*70 + "\n")
        
        log.info(f"Summary saved: {summary_path}")
        
        # Print to console
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Unique Motorcycles: {self.stats['unique_motorcycles']}")
        print(f"Motorcycles in ROI: {self.stats['motorcycles_in_roi']}")
        print(f"Violations in ROI: {self.stats['violations_in_roi']}")
        print(f"Safe Riders: {self.stats['safe_riders']}")
        print(f"Plates with Text: {self.stats['plates_with_text']}/{self.stats['plates_detected']}")
        print("="*70 + "\n")


def main():
    """Main entry point with command-line interface"""
    parser = argparse.ArgumentParser(
        description='Helmet Violation Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (no ROI)
  python main_pipeline.py --video test.mp4 --helmet-model model.pt --plate-model plate.pt
  
  # With ROI (rectangle)
  python main_pipeline.py --video test.mp4 --helmet-model model.pt --plate-model plate.pt \\
    --roi 100 200 500 200 500 600 100 600
  
  # With custom output
  python main_pipeline.py --video test.mp4 --helmet-model model.pt --plate-model plate.pt \\
    --output-dir ./results/run1 --output-video ./results/run1/output.mp4
        """
    )
    
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--helmet-model', required=True, help='Helmet detection model path')
    parser.add_argument('--plate-model', required=True, help='Plate detection model path')
    parser.add_argument('--output-dir', default='../results', help='Output directory')
    parser.add_argument('--output-video', default=None, help='Output video path')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')
    parser.add_argument('--roi', nargs='+', type=int, 
                       help='ROI polygon points: x1 y1 x2 y2 x3 y3 ... (min 6 values)')
    
    args = parser.parse_args()
    
    # Parse ROI
    roi_points = None
    if args.roi:
        if len(args.roi) < 6 or len(args.roi) % 2 != 0:
            print("Error: ROI must have at least 3 points (6 values: x1 y1 x2 y2 x3 y3)")
            return
        
        roi_points = [(args.roi[i], args.roi[i+1]) for i in range(0, len(args.roi), 2)]
        print(f"ROI defined with {len(roi_points)} points")
    
    # Initialize system
    system = ViolationDetectionSystem(
        helmet_model=args.helmet_model,
        plate_model=args.plate_model,
        output_dir=args.output_dir,
        use_gpu=not args.no_gpu,
        roi_points=roi_points
    )
    
    # Process video
    system.process_video(args.video, args.output_video)


if __name__ == "__main__":
    main()