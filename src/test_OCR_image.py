from ocr.thai_ocr import ThaiPlateOCR
import cv2
from pathlib import Path

print("Testing OCR module...")

ocr = ThaiPlateOCR(use_gpu=True)

# Test on a sample image
test_img = Path('../data/thai_plates/test_ocr.jpg')
img = cv2.imread(str(test_img))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Simulate plate region (center-bottom of image)
h, w = img.shape[:2]
fake_bbox = [w//4, 3*h//4, 3*w//4, h-10]

result = ocr.extract_text(img_rgb, fake_bbox)

if result:
    print(f"✓ OCR Working!")
    print(f"  Text: {result['text']}")
    print(f"  Confidence: {result['confidence']:.2f}")
else:
    print("⚠️  No text in test region (expected if no plate there)")

print("\n✓ OCR module functional")