"""
Test plate detection on single image
"""
from detection.yolo_detector import YOLODetector
from detection.visualizer import DetectionVisualizer
import cv2
from pathlib import Path

print("="*70)
print("TESTING PLATE DETECTION")
print("="*70)

# Load detector with plate model
print("\n1. Loading models...")
detector = YOLODetector(
    motorcycle_model_path= '',
    helmet_model_path= '',
    plate_model_path='../models/plate/best.pt'
)
print("✓ All models loaded!")

# Get test image
print("\n2. Loading test image...")
test_img_path = Path('../data/test_videos/Nohelmet_Plate.jpg')
print(f"   Image: {test_img_path.name}")

image = cv2.imread(str(test_img_path))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect everything
print("\n3. Running detection...")
motorcycles, riders, plates = detector.detect_all(image)

print(f"\n   Results:")
print(f"   - Motorcycles: {len(motorcycles)}")
print(f"   - Riders: {len(riders)}")
print(f"     • With helmet: {sum(1 for r in riders if r['class'] == 1)}")
print(f"     • Without helmet: {sum(1 for r in riders if r['class'] == 0)}")
print(f"   - Plates: {len(plates)}")

if len(plates) > 0:
    print(f"\n   Plate Details:")
    for i, plate in enumerate(plates):
        print(f"   Plate {i+1}:")
        print(f"     BBox: {plate['bbox']}")
        print(f"     Confidence: {plate['conf']:.2%}")

# Visualize
print("\n4. Creating visualization...")
visualizer = DetectionVisualizer()
result_img = visualizer.draw_detections(image_rgb, motorcycles, riders, plates)

# Save output
output_dir = Path('../results/plate_test')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'plate_detection_test.jpg'

result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(output_path), result_bgr)

print(f"\n✓ Output saved: {output_path}")

# Print summary
print("\n" + "="*70)
if len(plates) > 0:
    print("✅ PLATE DETECTION WORKING!")
    print(f"   Detected {len(plates)} plate(s)")
else:
    print("⚠️  NO PLATES DETECTED")
    print("   (This is normal if test image has no visible plates)")
print("="*70)

print(f"\n📷 View result: {output_path.absolute()}")