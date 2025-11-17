"""
Video violation detection with OCR
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
from ocr.thai_ocr import ThaiPlateOCR  # NEW!

class VideoViolationDetectorWithOCR:
    """Process videos with OCR"""
    
    def __init__(self, helmet_model_path: str, use_ocr: bool = True):
        print("Initializing detector with OCR...")
        
        self.detector = YOLODetector(
            motorcycle_model_path='yolov8n.pt',
            helmet_model_path=helmet_model_path
        )
        self.associator = ObjectAssociator(iou_threshold=0.2, max_distance=100)
        self.visualizer = DetectionVisualizer()
        
        # Initialize OCR
        self.use_ocr = use_ocr
        self.ocr = None
        if use_ocr:
            print("Loading OCR module...")
            self.ocr = ThaiPlateOCR(use_gpu=True)
            print("✓ OCR ready")
        
    def process_video(
        self,
        video_path: str,
        output_path: str = None,
        save_frames: bool = False,
        frame_skip: int = 1,
        show_realtime: bool = True
    ):
        """Process video with OCR"""
        print(f"\nProcessing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        if self.use_ocr:
            print("OCR: ENABLED ✓")
        
        # Output video
        out = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Results
        violations_list = []
        frame_idx = 0
        paused = False
        
        # Display window
        if show_realtime:
            cv2.namedWindow('Violation Detection + OCR', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Violation Detection + OCR', 1280, 720)
            print("\n▶️  Controls: 'q'=quit, 'p'=pause, 's'=save frame\n")
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue
                
                # Detect
                motorcycles, riders, plates = self.detector.detect_all(frame)
                violations = self.associator.process_frame(motorcycles, riders, plates)
                
                # OCR on violations with plates
                for violation in violations:
                    if violation.get('plate_bbox') and self.use_ocr and self.ocr:
                        # Convert BGR to RGB for OCR
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Extract text
                        ocr_result = self.ocr.extract_text(
                            frame_rgb,
                            violation['plate_bbox'],
                            preprocess=True
                        )
                        
                        if ocr_result:
                            violation['plate_text'] = ocr_result['text']
                            violation['plate_ocr_conf'] = ocr_result['confidence']
                        else:
                            violation['plate_text'] = "N/A"
                            violation['plate_ocr_conf'] = 0.0
                
                # Visualize
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if len(violations) > 0:
                    result_rgb = self.visualizer.draw_violations(frame_rgb, violations)
                    
                    # Add plate text overlay
                    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
                    for v in violations:
                        if v.get('plate_text') and v['plate_text'] != "N/A":
                            # Draw text near plate
                            if v.get('plate_bbox'):
                                x1, y1, x2, y2 = map(int, v['plate_bbox'])
                                text = f"Plate: {v['plate_text']}"
                                cv2.putText(result_bgr, text, (x1, y2 + 20),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    result_rgb = self.visualizer.draw_detections(frame_rgb, motorcycles, riders, plates)
                    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
                
                # Info overlay
                info = f"Frame: {frame_idx}/{total_frames} | Violations: {len(violations)}"
                if self.use_ocr:
                    ocr_count = sum(1 for v in violations if v.get('plate_text'))
                    info += f" | OCR: {ocr_count}"
                cv2.putText(result_bgr, info, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Save violations
                if len(violations) > 0:
                    violation_record = {
                        'frame': frame_idx,
                        'timestamp': frame_idx / fps,
                        'count': len(violations),
                        'details': []
                    }
                    
                    for v in violations:
                        detail = {
                            'rider_conf': float(v['rider_conf']),
                            'motorcycle_conf': float(v['motorcycle_conf'])
                        }
                        if v.get('plate_text'):
                            detail['plate_text'] = v['plate_text']
                            detail['plate_conf'] = float(v.get('plate_ocr_conf', 0))
                        
                        violation_record['details'].append(detail)
                    
                    violations_list.append(violation_record)
                    
                    if save_frames:
                        frame_dir = Path('../results/violation_frames')
                        frame_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(
                            str(frame_dir / f'violation_{frame_idx:06d}.jpg'),
                            result_bgr
                        )
                
                if out:
                    out.write(result_bgr)
                
                frame_idx += 1
            
            # Display
            if show_realtime:
                cv2.imshow('Violation Detection + OCR', result_bgr)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️  Stopped")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("⏸️  Paused" if paused else "▶️  Resumed")
                elif key == ord('s'):
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
        print("RESULTS WITH OCR")
        print(f"{'='*60}")
        print(f"Frames processed: {frame_idx}")
        print(f"Violations found: {len(violations_list)}")
        
        plates_read = sum(
            1 for v in violations_list 
            for d in v.get('details', []) 
            if d.get('plate_text')
        )
        print(f"Plates read: {plates_read}")
        
        if output_path:
            print(f"Output: {output_path}")
        print(f"{'='*60}")
        
        # Save JSON
        summary = {
            'video': video_path,
            'frames_processed': frame_idx,
            'violations': violations_list,
            'ocr_enabled': self.use_ocr,
            'plates_read': plates_read
        }
        
        summary_path = Path('../results/violation_summary_ocr.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Summary: {summary_path}")
        return summary


def main():
    parser = argparse.ArgumentParser(description='Violation Detection with OCR')
    parser.add_argument('--model', required=True, help='Helmet model path')
    parser.add_argument('--video', required=True, help='Input video')
    parser.add_argument('--output', help='Output video')
    parser.add_argument('--save-frames', action='store_true')
    parser.add_argument('--frame-skip', type=int, default=1)
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--no-ocr', action='store_true', help='Disable OCR')
    
    args = parser.parse_args()
    
    detector = VideoViolationDetectorWithOCR(
        helmet_model_path=args.model,
        use_ocr=not args.no_ocr
    )
    
    detector.process_video(
        video_path=args.video,
        output_path=args.output,
        save_frames=args.save_frames,
        frame_skip=args.frame_skip,
        show_realtime=not args.no_display
    )


if __name__ == "__main__":
    main()
