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
import traceback
from pathlib import Path
from typing import List, Tuple

from detection.yolo_detector import YOLODetector
from association.associator import ObjectAssociator
from detection.visualizer import DetectionVisualizer
from ocr.thai_ocr import ThaiPlateOCR
from src.temps.violation_logger import ViolationLogger
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
        log.info("=" * 70)
        log.info("Initializing Violation Detection System.")
        log.info("=" * 70)

        # Detection models
        # NOTE: pass only model paths here; YOLODetector handles internals.
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
        log.info("=" * 70)

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
        # Keep a BGR copy for detector (YOLO expects BGR in this project).
        bgr_frame = frame

        # Also prepare RGB for OCR / visualizer which expect RGB arrays
        try:
            img_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        except Exception:
            # If conversion fails, continue but log
            img_rgb = None
            log.debug("Could not convert frame to RGB for visualization/OCR.")

        # Step 1: Detection (use BGR to match working script behavior)
        try:
            motorcycles, riders, plates = self.detector.detect_all(bgr_frame)
        except Exception as e:
            log.error("Exception during detection: " + str(e))
            log.debug(traceback.format_exc())
            # Ensure we have empty lists and continue pipeline gracefully
            motorcycles, riders, plates = [], [], []

        # Debug logging: counts and a sample detection
        try:
            self.stats['total_motorcycles_detected'] += len(motorcycles)
            log.info(f"DEBUG: detections -> motorcycles={len(motorcycles)}, riders={len(riders)}, plates={len(plates)}")
            if len(motorcycles) > 0:
                sample = motorcycles[0]
                # sample likely has keys 'bbox' and 'conf'
                log.debug(f"DEBUG: sample motorbike: bbox={sample.get('bbox') if isinstance(sample, dict) else sample}, conf={sample.get('conf') if isinstance(sample, dict) else 'n/a'}")
        except Exception:
            log.debug("DEBUG: failed to log detection samples", exc_info=True)

        # Step 2: Track motorcycles (assign unique IDs)
        tracked_motorcycles = self.tracker.update(motorcycles, frame_number)

        # Step 3: Update ROI status for each track
        motorcycles_in_roi_count = 0
        for moto in tracked_motorcycles:
            try:
                is_in_roi = self.roi_manager.is_in_roi(moto['bbox'])
            except Exception:
                # If bbox format unexpected, treat as not in ROI
                is_in_roi = False
            self.tracker.update_roi_status(moto['track_id'], is_in_roi)

            if is_in_roi:
                motorcycles_in_roi_count += 1

        self.stats['motorcycles_in_roi'] += motorcycles_in_roi_count

        # Step 4: Find violations (associator expects raw detections lists)
        try:
            violations = self.associator.process_frame(motorcycles, riders, plates)
        except Exception as e:
            log.error("Exception during association/violation processing: " + str(e))
            log.debug(traceback.format_exc())
            violations = []

        log.info(f"DEBUG: Total violations={len(violations)}, riders={len(riders)}")
        self.stats['total_violations'] += len(violations)

        # Step 5: Filter violations by ROI
        if self.roi_manager.has_roi:
            try:
                violations_in_roi = [v for v in violations if 'rider_bbox' in v and self.roi_manager.is_in_roi(v['rider_bbox'])]
            except Exception:
                # If key missing or check fails, fall back to all violations
                log.debug("ROI filtering failed on violations; falling back to all violations.")
                violations_in_roi = violations
        else:
            violations_in_roi = violations

        self.stats['violations_in_roi'] += len(violations_in_roi)
        log.info(f"DEBUG: violations_in_roi={len(violations_in_roi)}")

        # Step 6: Count safe riders (assumes riders have 'class' with 1 = wearing helmet)
        try:
            safe_riders = [r for r in riders if r.get('class') == 1]
        except Exception:
            safe_riders = []
        self.stats['safe_riders'] += len(safe_riders)

        # Step 7: OCR on violations in ROI (use RGB for OCR)
        for i, v in enumerate(violations_in_roi):
            if 'plate_bbox' in v and self.ocr is not None and img_rgb is not None:
                self.stats['plates_detected'] += 1
                try:
                    result = self.ocr.extract_text(
                        img_rgb,
                        v['plate_bbox'],
                        preprocess=True,
                        frame_id=frame_number,
                        violation_id=i
                    )
                except Exception as e:
                    log.error("OCR exception: " + str(e))
                    log.debug(traceback.format_exc())
                    result = None

                if result:
                    v['plate_text'] = result.get('text', 'N/A')
                    v['plate_ocr_conf'] = result.get('confidence', 0)
                    v['plate_crop_path'] = result.get('crop_path', '')

                    if v['plate_text'] not in ['N/A', 'Low_Confidence', '', None]:
                        self.stats['plates_with_text'] += 1
                else:
                    v['plate_text'] = 'N/A'
                    v['plate_ocr_conf'] = 0.0

            # Log violation to CSV (logger should handle formatting)
            try:
                self.logger.log_violation(frame_number, timestamp, v)
            except Exception:
                log.debug("Failed to log violation to CSV", exc_info=True)

        # Step 8: Visualize
        log.info(f"DEBUG: Drawing {len(violations_in_roi)} violations")
        # Visualizer expects RGB input for drawing
        vis_rgb = img_rgb if img_rgb is not None else cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        try:
            result_img = self.visualizer.draw_violations(vis_rgb, violations_in_roi, draw_motorcycles=False)
        except Exception as e:
            log.error("Visualizer draw_violations failed: " + str(e))
            log.debug(traceback.format_exc())
            # Fallback: draw detections if available
            try:
                result_img = self.visualizer.draw_detections(vis_rgb, motorcycles, riders, plates)
            except Exception:
                # As last resort convert original frame to RGB and continue
                result_img = vis_rgb

        log.info(f"DEBUG: Result image shape={result_img.shape if hasattr(result_img, 'shape') else 'unknown'}")

        # Step 9: Draw ROI (visualizer expects RGB)
        try:
            result_img = self.roi_manager.draw_roi(result_img)
        except Exception:
            log.debug("ROI draw failed; continuing without ROI overlay", exc_info=True)

        # Step 10: Draw track IDs (tracker.draw_tracks should accept RGB frames)
        try:
            result_img = self.tracker.draw_tracks(result_img)
        except Exception:
            log.debug("Tracker draw_tracks failed; continuing", exc_info=True)

        # Step 11: Draw statistics on frame (still in RGB space)
        try:
            tracker_stats = self.tracker.get_statistics()
            stats_text = [
                f"Frame: {frame_number}",
                f"Tracked: {tracker_stats.get('active_tracks', 0)} | Total: {tracker_stats.get('total_unique_motorcycles', 0)}",
                f"In ROI: {motorcycles_in_roi_count}",
                f"Violations: {len(violations_in_roi)}",
                f"Safe: {len(safe_riders)}"
            ]

            y_offset = 30
            for text in stats_text:
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # Draw in RGB space (but OpenCV text uses BGR color tuple — in RGB image we'll set color as (255,255,255))
                # We'll convert to BGR at the very end so draw with white text here.
                cv2.rectangle(result_img, (5, y_offset - 25), (15 + text_w, y_offset + 5), (0, 0, 0), -1)
                cv2.putText(result_img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 35
        except Exception:
            log.debug("Failed to draw stats text", exc_info=True)

        # Convert result back to BGR for display / writing
        try:
            return cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        except Exception:
            # If conversion fails, return original BGR frame
            log.debug("Failed to convert RGB result back to BGR; returning original frame.")
            return bgr_frame

    def process_video(self, video_path: str, output_video: str = None):
        """
        Process entire video

        Args:
            video_path: Path to input video
            output_video: Path to output video (optional)
        """
        log.info("=" * 70)
        log.info(f"Processing video: {video_path}")
        log.info("=" * 70)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            log.error(f"Failed to open video: {video_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
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

        log.info("Starting processing.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp = frame_count / fps

            # Process frame
            result = self.process_frame(frame, frame_count, timestamp)

            # Write and show
            if writer.isOpened():
                writer.write(result)

            cv2.imshow('Detection', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                log.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        # Get final tracker stats
        tracker_stats = self.tracker.get_statistics()
        self.stats['unique_motorcycles'] = tracker_stats.get('total_unique_motorcycles', 0)
        self.stats['entered_roi'] = list(tracker_stats.get('entered_roi', []))
        self.stats['exited_roi'] = list(tracker_stats.get('exited_roi', []))

        # Save summary
        try:
            self.save_summary()
        except Exception:
            log.debug("Failed to save summary", exc_info=True)

        log.info("=" * 70)
        log.info("✓ Processing complete!")
        log.info("=" * 70)
        log.info(f"Output video: {output_video}")
        log.info(f"Results directory: {self.output_dir}")
        log.info("=" * 70)

    def save_summary(self):
        """Save summary statistics"""
        summary_path = self.output_dir / 'summary.txt'

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("VIOLATION DETECTION SUMMARY\n")
            f.write("=" * 70 + "\n")
            f.write(f"Total frames: {self.stats.get('total_frames', 0)}\n")
            f.write(f"Total motorcycles detected: {self.stats.get('total_motorcycles_detected', 0)}\n")
            f.write(f"Unique motorcycles: {self.stats.get('unique_motorcycles', 0)}\n")
            f.write(f"Motorcycles in ROI: {self.stats.get('motorcycles_in_roi', 0)}\n")
            f.write(f"Violations in ROI: {self.stats.get('violations_in_roi', 0)}\n")
            f.write(f"Safe riders: {self.stats.get('safe_riders', 0)}\n")
            f.write(f"Plates detected: {self.stats.get('plates_detected', 0)}\n")
            f.write(f"Plates with text: {self.stats.get('plates_with_text', 0)}\n")
            f.write("\n" + "=" * 70 + "\n")

        log.info(f"Summary saved: {summary_path}")

        # Print to console
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Unique Motorcycles: {self.stats.get('unique_motorcycles', 0)}")
        print(f"Motorcycles in ROI: {self.stats.get('motorcycles_in_roi', 0)}")
        print(f"Violations in ROI: {self.stats.get('violations_in_roi', 0)}")
        print(f"Safe Riders: {self.stats.get('safe_riders', 0)}")
        print(f"Plates with Text: {self.stats.get('plates_with_text', 0)}/{self.stats.get('plates_detected', 0)}")
        print("=" * 70 + "\n")


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
    parser.add_argument('--output-dir', default='./results', help='Output directory')
    parser.add_argument('--output-video', default=None, help='Output video path')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')
    parser.add_argument('--roi', nargs='+', type=int,
                        help='ROI polygon points: x1 y1 x2 y2 x3 y3 . (min 6 values)')

    args = parser.parse_args()

    # Parse ROI
    roi_points = None
    if args.roi:
        if len(args.roi) < 6 or len(args.roi) % 2 != 0:
            print("Error: ROI must have at least 3 points (6 values: x1 y1 x2 y2 x3 y3)")
            return

        roi_points = [(args.roi[i], args.roi[i + 1]) for i in range(0, len(args.roi), 2)]
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
