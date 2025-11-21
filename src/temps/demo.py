"""
FINAL DEMO - Clean & Working
"""
import cv2
import csv
from pathlib import Path
from detection.yolo_detector import YOLODetector
from association.associator import ObjectAssociator
from detection.visualizer import DetectionVisualizer
from utils.roi_manager import ROIManager
from utils.tracker import MotorcycleTracker
from utils.logger import log

class FinalDemo:
    def __init__(self, helmet_model, plate_model, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info("Initializing...")
        
        self.detector = YOLODetector(
            motorcycle_model_path='yolov8n.pt',
            helmet_model_path=helmet_model,
            plate_model_path=plate_model
        )
        self.associator = ObjectAssociator()
        self.visualizer = DetectionVisualizer()
        self.tracker = MotorcycleTracker()
        
        self.roi_manager = None  # Created from video size
        
        # CSV
        csv_path = self.output_dir / 'violations.csv'
        self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Frame', 'Time', 'Class', 'Conf', 'Plate'])
        
        log.info("Ready!")
    
    def create_roi(self, w, h):
        """Create ROI covering 80% of frame"""
        x1 = int(w * 0.1)
        x2 = int(w * 0.9)
        y1 = int(h * 0.1)
        y2 = int(h * 0.9)
        
        roi_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        self.roi_manager = ROIManager(roi_points)
    
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.error("Cannot open video")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        log.info(f"Video: {w}x{h} @ {fps}fps, {total} frames")
        
        # Create ROI
        self.create_roi(w, h)
        
        # Output
        out_path = str(self.output_dir / 'output.mp4')
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        frame_num = 0
        violation_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect
            motorcycles, riders, plates = self.detector.detect_all(img_rgb)
            
            # Track
            self.tracker.update(motorcycles, frame_num)
            
            # Find violations
            violations = self.associator.process_frame(motorcycles, riders, plates)
            
            # Filter by ROI
            violations_roi = [v for v in violations if self.roi_manager.is_in_roi(v['rider_bbox'])]
            violation_count += len(violations_roi)
            
            # Log to CSV
            class_names = ['No_helmet', 'Wearing_helmet', 'Person_on_Bike']
            for v in violations_roi:
                self.csv_writer.writerow([
                    frame_num,
                    f"{frame_num/fps:.2f}s",
                    class_names[v['rider_class']],
                    f"{v['rider_conf']:.2f}",
                    v.get('plate_text', 'N/A')
                ])
            
            # Draw
            result = self.visualizer.draw_violations(img_rgb, violations_roi, draw_motorcycles=False)
            result = self.roi_manager.draw_roi(result)
            
            # Stats
            tracker_stats = self.tracker.get_statistics()
            cv2.putText(result, f"Frame: {frame_num}/{total}", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(result, f"In ROI: {tracker_stats['entered_roi']}", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(result, f"Violations: {len(violations_roi)}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            writer.write(result_bgr)
            
            cv2.imshow('Demo', result_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if frame_num % 30 == 0:
                log.info(f"{frame_num}/{total}")
        
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        self.csv_file.close()
        
        # Summary
        tracker_stats = self.tracker.get_statistics()
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Motorcycles in ROI: {tracker_stats['entered_roi']}")
        print(f"Total Violations: {violation_count}")
        print(f"Video: {out_path}")
        print(f"CSV: {self.output_dir / 'violations.csv'}")
        print("="*60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python demo_final.py <video> <helmet_model> <plate_model>")
        sys.exit(1)
    
    demo = FinalDemo(sys.argv[2], sys.argv[3], '../results/final_demo')
    demo.process_video(sys.argv[1])
