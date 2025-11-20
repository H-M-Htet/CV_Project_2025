"""
Main violation detection pipeline
Combines all components for end-to-end detection
"""
import cv2
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import json
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from detection.yolo_detector import YOLODetector
from association.associator import ObjectAssociator
from detection.visualizer import DetectionVisualizer
from utils.logger import setup_logger, log
from utils.common import VideoProcessor, save_image
from utils.config_loader import config

class ViolationDetectionPipeline:
    """
    Complete pipeline for helmet violation detection
    """
    
    def __init__(
        self,
        helmet_model_path: str,
        motorcycle_model_path: str = 'yolov8n.pt',
        plate_model_path: str = None
    ):
        """
        Initialize detection pipeline
        
        Args:
            helmet_model_path: Path to trained helmet detection model
            motorcycle_model_path: Path to motorcycle model (default: pre-trained)
            plate_model_path: Path to plate model (optional)
        """
        log.info("Initializing Violation Detection Pipeline...")
        
        # Initialize detector
        self.detector = YOLODetector(
            motorcycle_model_path=motorcycle_model_path,
            helmet_model_path=helmet_model_path,
            plate_model_path=plate_model_path
        )
        
        # Initialize associator
        assoc_config = config.get('detection.association', {})
        self.associator = ObjectAssociator(
            iou_threshold=assoc_config.get('iou_threshold', 0.2),
            max_distance=assoc_config.get('max_distance', 100)
        )
        
        # Initialize visualizer
        self.visualizer = DetectionVisualizer()
        
        # Results storage
        self.results_dir = Path("../results/violations")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        log.info("✓ Pipeline initialized successfully")
    
    def process_image(
        self,
        image: np.ndarray,
        visualize: bool = True
    ) -> dict:
        """
        Process single image
        
        Args:
            image: Input image (RGB)
            visualize: Whether to create visualization
        
        Returns:
            Dict with detections and violations
        """
        # Stage 1: Detect all objects
        motorcycles, riders, plates = self.detector.detect_all(image)
        
        # Stage 2: Associate and find violations
        violations = self.associator.process_frame(motorcycles, riders, plates)
        
        result = {
            'motorcycles': motorcycles,
            'riders': riders,
            'plates': plates,
            'violations': violations,
            'num_violations': len(violations)
        }
        
        # Create visualization if requested
        if visualize:
            result['visualization'] = self.visualizer.draw_violations(
                image, violations
            )
        
        return result
    
    def process_video(
        self,
        video_path: str,
        output_path: str = None,
        save_frames: bool = False,
        frame_skip: int = 1
    ) -> dict:
        """
        Process video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            save_frames: Whether to save violation frames
            frame_skip: Process every N frames (1 = all frames)
        
        Returns:
            Dict with summary statistics
        """
        log.info(f"Processing video: {video_path}")
        
        # Open video
        with VideoProcessor(video_path) as video:
            log.info(f"Video info: {video.width}x{video.height}, {video.fps} FPS, {video.frame_count} frames")
            
            # Prepare output video if needed
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    video.fps,
                    (video.width, video.height)
                )
            
            # Process frames
            frame_idx = 0
            total_violations = 0
            violation_frames = []
            
            pbar = tqdm(total=video.frame_count, desc="Processing")
            
            while True:
                frame = video.read_frame()
                if frame is None:
                    break
                
                # Skip frames if needed
                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    pbar.update(1)
                    continue
                
                # Process frame
                result = self.process_image(frame, visualize=True)
                
                # Count violations
                if result['num_violations'] > 0:
                    total_violations += result['num_violations']
                    violation_frames.append({
                        'frame_idx': frame_idx,
                        'timestamp': frame_idx / video.fps,
                        'num_violations': result['num_violations'],
                        'violations': result['violations']
                    })
                    
                    # Save frame if requested
                    if save_frames:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = self.results_dir / f"violation_frame_{frame_idx}_{timestamp}.jpg"
                        save_image(result['visualization'], str(save_path))
                
                # Write to output video
                if output_path:
                    out_frame = cv2.cvtColor(result['visualization'], cv2.COLOR_RGB2BGR)
                    out.write(out_frame)
                
                frame_idx += 1
                pbar.update(1)
            
            pbar.close()
            
            if output_path:
                out.release()
        
        # Summary
        summary = {
            'video_path': video_path,
            'total_frames': frame_idx,
            'frames_processed': frame_idx // frame_skip,
            'total_violations': total_violations,
            'violation_frames': violation_frames,
            'output_video': output_path
        }
        
        log.info(f"✓ Video processing complete")
        log.info(f"  Total violations: {total_violations}")
        log.info(f"  Violation frames: {len(violation_frames)}")
        
        # Save summary
        summary_path = self.results_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        log.info(f"  Summary saved: {summary_path}")
        
        return summary
    
    def process_folder(
        self,
        folder_path: str,
        output_folder: str = None
    ):
        """
        Process all images in a folder
        
        Args:
            folder_path: Path to folder with images
            output_folder: Path to save results
        """
        folder = Path(folder_path)
        output = Path(output_folder) if output_folder else self.results_dir
        output.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        images = []
        for ext in image_extensions:
            images.extend(folder.glob(ext))
        
        log.info(f"Found {len(images)} images in {folder}")
        
        total_violations = 0
        
        for img_path in tqdm(images, desc="Processing images"):
            # Load and process
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            result = self.process_image(img, visualize=True)
            
            if result['num_violations'] > 0:
                total_violations += result['num_violations']
                
                # Save result
                output_path = output / f"{img_path.stem}_violations{img_path.suffix}"
                save_image(result['visualization'], str(output_path))
        
        log.info(f"✓ Batch processing complete")
        log.info(f"  Total violations: {total_violations}")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Helmet Violation Detection")
    parser.add_argument('--model', type=str, required=True, help='Path to helmet detection model')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--folder', type=str, help='Path to folder with images')
    parser.add_argument('--output', type=str, help='Path to output file/folder')
    parser.add_argument('--save-frames', action='store_true', help='Save violation frames')
    parser.add_argument('--frame-skip', type=int, default=1, help='Process every N frames')
    
    args = parser.parse_args()
    
    # Setup logger
    setup_logger(log_file="../results/logs/detection.log", level="INFO")
    
    # Initialize pipeline
    pipeline = ViolationDetectionPipeline(
        helmet_model_path=args.model
    )
    
    # Process based on input type
    if args.video:
        pipeline.process_video(
            video_path=args.video,
            output_path=args.output,
            save_frames=args.save_frames,
            frame_skip=args.frame_skip
        )
    elif args.image:
        img = cv2.imread(args.image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pipeline.process_image(img, visualize=True)
        
        if args.output:
            save_image(result['visualization'], args.output)
        
        log.info(f"Detected {result['num_violations']} violations")
    elif args.folder:
        pipeline.process_folder(
            folder_path=args.folder,
            output_folder=args.output
        )
    else:
        log.error("Please specify --video, --image, or --folder")


if __name__ == "__main__":
    main()
