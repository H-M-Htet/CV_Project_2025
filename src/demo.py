"""
DEMO with existing ROI Manager & Tracker
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

class Demo:
    def __init__(self, helmet_model, plate_model, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Models
        self.detector = YOLODetector(
            motorcycle_model_path='yolov8n.pt',
            helmet_model_path=helmet_model,
            plate_model_path=plate_model
        )
        self.associator = ObjectAssociator()
        self.visualizer = DetectionVisualizer()
        
        # Will be initialized with video size
        self.roi_manager = None
        self.tracker = MotorcycleTracker()
        
        # Stats
        self.stats = {'violations': 0}
        
        # CSV
        csv_path = self.output_dir / 'violations.csv'
        self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'Frame', 'Timestamp', 'Class', 'Confidence', 'Plate', 'In_ROI'
        ])
        
        log.info("Demo initialized")
    
    def create_center_roi(self, width, height):
        """Create ROI at center"""
        x1, x2 = int(width * 0.1), int(width * 0.9)
        y1, y2 = int(height * 0.1), int(height * 0.9)
        
        roi_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        self.roi_manager = ROIManager(roi_points)
        log.info(f"ROI: {x2-x1}x{y2-y1} at center")
    
    def process_video(self, video_path, output_video=None):
        log.info(f"Processing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.error("Cannot open video")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create ROI
        self.create_center_roi(w, h)
        
        # Output
        if output_video is None:
            output_video = str(self.output_dir / 'demo_output.mp4')
        
        writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect
            motorcycles, riders, plates = self.detector.detect_all(img_rgb)
            
            # Track
            tracked = self.tracker.update(motorcycles, frame_count)
            for moto in tracked:
                is_in = self.roi_manager.is_in_roi(moto['bbox'])
                self.tracker.update_roi_status(moto['track_id'], is_in)
            
            # Violations
            violations = self.associator.process_frame(motorcycles, riders, plates)
            violations_in_roi = [v for v in violations if self.roi_manager.is_in_roi(v['rider_bbox'])]
            self.stats['violations'] += len(violations_in_roi)
            
            # Log CSV
            class_names = ['No_helmet', 'Wearing_helmet', 'Person_on_Bike']
            for v in violations_in_roi:
                self.csv_writer.writerow([
                    frame_count, f"{timestamp:.2f}",
                    class_names[v['rider_class']], f"{v['rider_conf']:.2f}",
                    v.get('plate_text', 'N/A'), 'Yes'
                ])
            
            # Visualize
            result = self.visualizer.draw_violations(img_rgb, violations_in_roi, draw_motorcycles=False)
            result = self.roi_manager.draw_roi(result)
            result = self.tracker.draw_tracks(result)
            
            # Stats overlay
            tracker_stats = self.tracker.get_statistics()
            stats_text = [
                f"Frame: {frame_count}/{total}",
                f"In ROI: {tracker_stats['entered_roi']}",
                f"Violations: {len(violations_in_roi)}"
            ]
            
            y = 40
            for text in stats_text:
                cv2.putText(result, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                y += 35
            
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            writer.write(result_bgr)
            
            cv2.imshow('Demo', result_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if frame_count % 30 == 0:
                log.info(f"{frame_count}/{total}")
        
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        self.csv_file.close()
        
        # Summary
        tracker_stats = self.tracker.get_statistics()
        print("\n" + "="*60)
        print("DEMO RESULTS")
        print("="*60)
        print(f"Motorcycles entered ROI: {tracker_stats['entered_roi']}")
        print(f"Violations: {self.stats['violations']}")
        print(f"Video: {output_video}")
        print(f"CSV: {self.output_dir / 'violations.csv'}")
        print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python demo_with_utils.py <video> <helmet_model> <plate_model>")
        sys.exit(1)
    
    demo = Demo(sys.argv[2], sys.argv[3], '../results/demo')
    demo.process_video(sys.argv[1])