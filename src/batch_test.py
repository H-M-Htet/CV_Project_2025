"""
Batch test on validation images
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from detection.yolo_detector import YOLODetector
from association.associator import ObjectAssociator
from detection.visualizer import DetectionVisualizer
import cv2
from tqdm import tqdm

print("Batch Testing Violation Detection")
print("="*60)

# Initialize
detector = YOLODetector(
    motorcycle_model_path='yolov8n.pt',
    helmet_model_path='../models/yolo/best0.1.pt'
)
associator = ObjectAssociator()
visualizer = DetectionVisualizer()

# Get test images
valid_images = list(Path('../data/helmet_dataset/test/images').glob('*.[jp][pn][g]'))[60:]  # Test first 20

print(f"Testing on {len(valid_images)} images...")

output_dir = Path('../results/batch_test')
output_dir.mkdir(parents=True, exist_ok=True)

stats = {
    'total_images': len(valid_images),
    'total_motorcycles': 0,
    'total_riders': 0,
    'with_helmet': 0,
    'without_helmet': 0,
    'violations': 0
}

for img_path in tqdm(valid_images, desc="Processing"):
    # Load image
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect
    motorcycles, riders, _ = detector.detect_all(image)
    
    # Update stats
    stats['total_motorcycles'] += len(motorcycles)
    stats['total_riders'] += len(riders)
    stats['with_helmet'] += sum(1 for r in riders if r['class'] == 0)
    stats['without_helmet'] += sum(1 for r in riders if r['class'] == 1)
    
    # Associate
    violations = associator.process_frame(motorcycles, riders)
    stats['violations'] += len(violations)
    
    # Save if violations found
    if len(violations) > 0:
        result_img = visualizer.draw_violations(image, violations)
        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f'violation_{img_path.name}'), result_bgr)

print("\n" + "="*60)
print("BATCH TEST RESULTS")
print("="*60)
print(f"Images tested: {stats['total_images']}")
print(f"Motorcycles detected: {stats['total_motorcycles']}")
print(f"Riders detected: {stats['total_riders']}")
print(f"  • With helmet: {stats['with_helmet']} ({stats['with_helmet']/max(stats['total_riders'],1)*100:.1f}%)")
print(f"  • Without helmet: {stats['without_helmet']} ({stats['without_helmet']/max(stats['total_riders'],1)*100:.1f}%)")
print(f"\nViolations found: {stats['violations']}")
print(f"\n✓ Results saved to: {output_dir}")
print("="*60)

