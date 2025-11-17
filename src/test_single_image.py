"""
Test helmet detection on a single image
"""
from ultralytics import YOLO
import cv2
from pathlib import Path

print("Testing helmet detection on sample image...")

# Load model
model = YOLO('../models/yolo/best0.1.pt')

# # Find a test image from your validation set
# valid_images = list(Path('../data/helmet_dataset/test/images/no-helmet.jpg').glob('*.jpg'))

# if len(valid_images) == 0:
#     print("❌ No validation images found!")
#     print("Please provide a test image path")
#     exit(1)

# Test on first image
test_img = Path('../data/helmet_dataset/test/images/With_helmet.jpg')

if not test_img.exists():
    print(f"❌ Test image not found at {test_img}")
    exit(1)

print(f"\nTesting on: {test_img}")

# Run detection
results = model(str(test_img), conf=0.5)

# Process results
result = results[0]
print(f"\n📊 Detection Results:")
print(f"Found {len(result.boxes)} object(s)")

for i, box in enumerate(result.boxes):
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    bbox = box.xyxy[0].cpu().numpy()
    
    class_name = model.names[cls]
    print(f"\n  Detection {i+1}:")
    print(f"    Class: {class_name}")
    print(f"    Confidence: {conf:.2%}")
    print(f"    BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")

# Save annotated result
output_dir = Path('../results/test_outputs')
output_dir.mkdir(parents=True, exist_ok=True)

annotated = result.plot()
output_path = output_dir / f'detection_{test_img.stem}.jpg'
cv2.imwrite(str(output_path), annotated)

print(f"\n✓ Annotated image saved to: {output_path}")
print(f"\n✓ Helmet detection test complete!")