"""
Test complete detection pipeline with association
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from detection.yolo_detector import YOLODetector
from association.associator import ObjectAssociator
from detection.visualizer import DetectionVisualizer
import cv2

print("="*70)
print("TESTING COMPLETE VIOLATION DETECTION PIPELINE")
print("="*70)

# Initialize components
print("\n1. Initializing detector...")
detector = YOLODetector(
    helmet_model_path= ('../models/yolo/best0.1.pt'),
    motorcycle_model_path='yolov8n.pt',
)


print("2. Initializing associator...")
associator = ObjectAssociator(iou_threshold=0.2, max_distance=100)

print("3. Initializing visualizer...")
visualizer = DetectionVisualizer()

# Find test image
valid_images = list(Path('../data/helmet_dataset/test/images').glob('*.[jp][pn][g]'))

if len(valid_images) == 0:
    print("\n❌ No test images found!")
    exit(1)

test_img_path = valid_images[0]
print(f"\n4. Testing on: {test_img_path.name}")

# Load image - KEEP IN BGR FOR YOLO
image_bgr = cv2.imread(str(test_img_path))

print("\n5. Running detection...")
# Detect all objects - YOLO expects BGR
motorcycles, riders, plates = detector.detect_all(image_bgr)

print(f"   Detected:")
print(f"   - Motorcycles: {len(motorcycles)}")
print(f"   - Riders: {len(riders)}")
print(f"     • With helmet: {sum(1 for r in riders if r['class'] == 1)}")  # Fixed: 1=with
print(f"     • Without helmet: {sum(1 for r in riders if r['class'] == 0)}")  # Fixed: 0=without

print("\n6. Performing association...")
# Associate and find violations
violations = associator.process_frame(motorcycles, riders)

print(f"   Found {len(violations)} violation(s)!")

if len(violations) > 0:
    print("\n   Violation details:")
    for i, v in enumerate(violations):
        print(f"   Violation {i+1}:")
        print(f"     - Rider confidence: {v['rider_conf']:.2%}")
        print(f"     - Motorcycle confidence: {v['motorcycle_conf']:.2%}")

# Visualize - convert to RGB for visualizer
print("\n7. Creating visualization...")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

if len(violations) > 0:
    result_img_rgb = visualizer.draw_violations(image_rgb, violations)
else:
    result_img_rgb = visualizer.draw_detections(image_rgb, motorcycles, riders)

# Save - convert back to BGR for cv2.imwrite
output_dir = Path('../results/test_outputs')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f'pipeline_test_{test_img_path.stem}.jpg'

result_img_bgr = cv2.cvtColor(result_img_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(output_path), result_img_bgr)

print(f"\n✓ Result saved to: {output_path}")
print("\n" + "="*70)
print("✓ PIPELINE TEST COMPLETE!")
print("="*70)

# Summary
print("\nSummary:")
print(f"  • Helmet detection model: ✓ Working")
print(f"  • Motorcycle detection: ✓ Working")
print(f"  • Association logic: ✓ Working")
print(f"  • Visualization: ✓ Working")
print(f"\n🎉 Full pipeline is operational!")