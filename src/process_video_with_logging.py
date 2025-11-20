"""
Process video with violation logging and plate crops
"""
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from detection.yolo_detector import YOLODetector
from association.associator import ObjectAssociator
from detection.visualizer import DetectionVisualizer
from ocr.thai_ocr import ThaiPlateOCR
from violation_logger import ViolationLogger
from utils.logger import log

def process_video_with_logging(
    video_path: str,
    output_video: str,
    output_csv: str,
    crops_dir: str,
    helmet_model: str,
    plate_model: str
):
    """
    Process video with full logging
    """
    log.info("="*70)
    log.info("PROCESSING VIDEO WITH LOGGING")
    log.info("="*70)
    
    # Initialize models
    detector = YOLODetector(
        motorcycle_model_path='yolov8n.pt',
        helmet_model_path=helmet_model,
        plate_model_path=plate_model
    )
    associator = ObjectAssociator()
    visualizer = DetectionVisualizer()
    ocr = ThaiPlateOCR(use_gpu=True, save_crops=True, crops_dir=crops_dir)
    logger = ViolationLogger(output_csv)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video writer
    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h)
    )
    
    # Counters
    frame_count = 0
    total_violations = 0
    total_motorcycles = 0
    safe_count = 0
    
    log.info(f"Video: {video_path}")
    log.info(f"Total frames: {total_frames}, FPS: {fps}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect
        motorcycles, riders, plates = detector.detect_all(img_rgb)
        total_motorcycles += len(motorcycles)
        
        # Find violations
        violations = associator.process_frame(motorcycles, riders, plates)
        total_violations += len(violations)
        
        # Count safe riders
        safe_riders = [r for r in riders if r['class'] == 1]  # Wearing_helmet
        safe_count += len(safe_riders)
        
        # OCR on violations
        for i, v in enumerate(violations):
            if 'plate_bbox' in v:
                result = ocr.extract_text(
                    img_rgb,
                    v['plate_bbox'],
                    preprocess=True,
                    frame_id=frame_count,
                    violation_id=i
                )
                
                if result:
                    v['plate_text'] = result['text']
                    v['plate_ocr_conf'] = result['confidence']
                    v['plate_crop_path'] = result['crop_path']
            
            # Log to CSV
            logger.log_violation(frame_count, timestamp, v)
        
        # Visualize
        result_img = visualizer.draw_violations(img_rgb, violations, draw_motorcycles=False)
        
        # Add frame info
        info_text = f"Frame: {frame_count}/{total_frames} | Violations: {len(violations)}"
        cv2.putText(result_img, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Write frame
        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        writer.write(result_bgr)
        cv2.imshow('Detection', result_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        # Progress
        if frame_count % 30 == 0:
            log.info(f"Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    writer.release()
    
    # Save summary
    logger.log_summary(total_frames, total_motorcycles, total_violations, safe_count)
    
    log.info("="*70)
    log.info("PROCESSING COMPLETE!")
    log.info("="*70)
    log.info(f"Output video: {output_video}")
    log.info(f"CSV log: {output_csv}")
    log.info(f"Plate crops: {crops_dir}")
    log.info(f"Total violations: {total_violations}")

if __name__ == "__main__":
    process_video_with_logging(
        video_path='../data/test_videos/TU_test1.mp4',
        output_video='../results/output_with_logging.mp4',
        output_csv='../results/violations.csv',
        crops_dir='../results/plate_crops',
        helmet_model='../models/yolo/best_H_0.2.pt',
        plate_model='../models/plate/best_P_0.1.pt'
    )
