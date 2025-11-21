"""
Video violation detection with OCR + ROI
FOR PRESENTATION DEMO
"""
import cv2
import numpy as np
from pathlib import Path
import sys
import json
import csv
from datetime import datetime
import argparse

sys.path.append(str(Path(__file__).parent))

from detection.yolo_detector import YOLODetector
from association.associator import ObjectAssociator
from detection.visualizer import DetectionVisualizer
from ocr.thai_ocr import ThaiPlateOCR
from utils.tracker import MotorcycleTracker

class VideoViolationDetectorWithOCR:
    """Process videos with OCR + ROI"""
    
    def __init__(self, helmet_model_path: str, plate_model_path: str = None, use_ocr: bool = True):
        print("Initializing detector with OCR + ROI...")
        
        self.detector = YOLODetector(
            motorcycle_model_path='yolov8n.pt',
            helmet_model_path=helmet_model_path,
            plate_model_path=plate_model_path
        )
        self.associator = ObjectAssociator(iou_threshold=0.2, max_distance=100)
        self.visualizer = DetectionVisualizer()
        self.tracker = MotorcycleTracker()
        
        # ROI (will be created from video size)
        self.roi_polygon = None
        
        # OCR
        self.use_ocr = use_ocr
        self.ocr = None
        if use_ocr:
            print("Loading OCR...")
            self.ocr = ThaiPlateOCR(use_gpu=True, save_crops=False)
            print("✓ OCR ready")
        
        # Stats
        self.stats = {
            'total_motorcycles': 0,
            'motorcycles_in_roi': 0,
            'total_violations': 0,
            'violations_in_roi': 0,
            'wearing_helmet': 0
        }
    
    def create_roi(self, width, height):
        """Create ROI covering 80% of center"""
        x1 = int(width * 0.1)
        x2 = int(width * 0.9)
        y1 = int(height * 0.1)
        y2 = int(height * 0.9)
        
        self.roi_polygon = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ], dtype=np.int32)
        
        print(f"ROI: {x2-x1}x{y2-y1} (80% of frame)")
    
    def is_in_roi(self, bbox):
        """Check if bbox center is in ROI"""
        if self.roi_polygon is None:
            return True
        
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        
        result = cv2.pointPolygonTest(self.roi_polygon, (cx, cy), False)
        return result >= 0
    
    def draw_roi(self, image):
        """Draw ROI on image"""
        if self.roi_polygon is None:
            return image
        
        result = image.copy()
        
        # Draw polygon (cyan)
        cv2.polylines(result, [self.roi_polygon], True, (0, 255, 255), 3)
        cv2.putText(result, "ROI", tuple(self.roi_polygon[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # Semi-transparent fill
        # overlay = result.copy()
        # cv2.fillPoly(overlay, [self.roi_polygon], (0, 255, 255))
        # result = cv2.addWeighted(result, 0.92, overlay, 0.08, 0)
        
        return result
    
    def process_video(
        self,
        video_path: str,
        output_path: str = None,
        show_realtime: bool = True
    ):
        """Process video with ROI"""
        print(f"\nProcessing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open: {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {width}x{height}, {fps}fps, {total_frames} frames")
        
        # Create ROI
        self.create_roi(width, height)
        
        # Output video
        if output_path is None:
            output_path = '../results/presentation_demo.mp4'
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        # CSV log
        csv_path = Path(output_path).parent / 'violations_log.csv'
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Frame', 'Time_Sec', 'Class', 'Confidence', 'Plate_Text', 'In_ROI'])
        
        frame_idx = 0
        
        if show_realtime:
            cv2.namedWindow('Presentation Demo', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Presentation Demo', 1280, 720)
            print("\n▶️  Press 'q' to quit\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect
            motorcycles, riders, plates = self.detector.detect_all(frame)
            self.stats['total_motorcycles'] += len(motorcycles)
            
            # Track motorcycles
            tracked = self.tracker.update(motorcycles, frame_idx)
            for moto in tracked:
                if self.is_in_roi(moto['bbox']):
                    self.tracker.update_roi_status(moto['track_id'], True)
            
            # Count safe riders (wearing helmet)
            wearing_helmet = [r for r in riders if r['class'] == 1]
            self.stats['wearing_helmet'] += len(wearing_helmet)
            
            # Find violations
            violations = self.associator.process_frame(motorcycles, riders, plates)
            self.stats['total_violations'] += len(violations)
            
            # Filter violations in ROI
            violations_in_roi = [v for v in violations if self.is_in_roi(v['rider_bbox'])]
            self.stats['violations_in_roi'] += len(violations_in_roi)
            
            # OCR on violations
            class_names = ['No_helmet', 'Wearing_helmet', 'Person_on_Bike']
            for v in violations_in_roi:
                plate_text = 'N/A'
                
                if v.get('plate_bbox') and self.use_ocr and self.ocr:
                    ocr_result = self.ocr.extract_text(img_rgb, v['plate_bbox'], preprocess=True)
                    if ocr_result:
                        plate_text = ocr_result['text']
                        v['plate_text'] = plate_text
                
                # Log to CSV
                csv_writer.writerow([
                    frame_idx,
                    f"{frame_idx/fps:.2f}",
                    class_names[v['rider_class']],
                    f"{v['rider_conf']:.2f}",
                    plate_text,
                    'Yes'
                ])
            
            # Visualize
            result = self.visualizer.draw_violations(frame, violations_in_roi, draw_motorcycles=False)
            result = self.draw_roi(result)
            
            # Stats overlay (TOP LEFT)
            tracker_stats = self.tracker.get_statistics()
            stats_text = [
                f"Frame: {frame_idx}/{total_frames}",
                f"Motorcycles Entered ROI: {tracker_stats['entered_roi']}",
                f"Wearing Helmet: {len(wearing_helmet)}",
                f"Violations in ROI: {len(violations_in_roi)}"
            ]
            
            y = 40
            for text in stats_text:
                # Background
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(result, (5, y-30), (tw+15, y+5), (0, 0, 0), -1)
                # Text
                cv2.putText(result, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y += 40
            
            # Convert and save
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            out.write(result_bgr)
            
            if show_realtime:
                cv2.imshow('Presentation Demo', result_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if frame_idx % 30 == 0:
                print(f"Progress: {frame_idx}/{total_frames}")
        
        cap.release()
        out.release()
        csv_file.close()
        
        if show_realtime:
            cv2.destroyAllWindows()
        
        # Get final stats
        tracker_stats = self.tracker.get_statistics()
        self.stats['motorcycles_in_roi'] = tracker_stats['entered_roi']
        
        # Save summary
        self.save_summary(output_path, csv_path)
        
        return self.stats
    
    def save_summary(self, video_path, csv_path):
        """Save presentation summary"""
        summary_path = Path(video_path).parent / 'PRESENTATION_SUMMARY.txt'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("HELMET VIOLATION DETECTION - DEMO RESULTS\n")
            f.write("="*70 + "\n\n")
            
            f.write("📊 SUMMARY:\n")
            f.write(f"   • Total Motorcycles Entered ROI: {self.stats['motorcycles_in_roi']}\n")
            f.write(f"   • Riders Wearing Helmet: {self.stats['wearing_helmet']}\n")
            f.write(f"   • Violations Detected: {self.stats['violations_in_roi']}\n\n")
            
            if self.stats['motorcycles_in_roi'] > 0:
                violation_rate = (self.stats['violations_in_roi'] / self.stats['motorcycles_in_roi']) * 100
                f.write(f"   • Violation Rate: {violation_rate:.1f}%\n\n")
            
            f.write("📁 OUTPUT FILES:\n")
            f.write(f"   • Demo Video: {video_path}\n")
            f.write(f"   • Violations Log: {csv_path}\n")
            f.write(f"   • Summary: {summary_path}\n\n")
            
            f.write("="*70 + "\n")
        
        # Print to console
        print("\n" + "="*70)
        print("PRESENTATION DEMO - RESULTS")
        print("="*70)
        print(f"Motorcycles Entered ROI: {self.stats['motorcycles_in_roi']}")
        print(f"Wearing Helmet: {self.stats['wearing_helmet']}")
        print(f"Violations: {self.stats['violations_in_roi']}")
        print(f"\nVideo: {video_path}")
        print(f"CSV: {csv_path}")
        print(f"Summary: {summary_path}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Presentation Demo')
    parser.add_argument('--video', required=True, help='Input video')
    parser.add_argument('--helmet-model', required=True, help='Helmet model')
    parser.add_argument('--plate-model', help='Plate model')
    parser.add_argument('--output', help='Output video path')
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--no-ocr', action='store_true')
    
    args = parser.parse_args()
    
    detector = VideoViolationDetectorWithOCR(
        helmet_model_path=args.helmet_model,
        plate_model_path=args.plate_model,
        use_ocr=not args.no_ocr
    )
    
    detector.process_video(
        video_path=args.video,
        output_path=args.output,
        show_realtime=not args.no_display
    )


if __name__ == "__main__":
    main()
