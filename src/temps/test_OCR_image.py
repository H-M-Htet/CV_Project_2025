"""
Test Thai License Plate OCR
"""
import cv2
import easyocr
from pathlib import Path
import numpy as np

print("="*70)
print("THAI LICENSE PLATE OCR TEST")
print("="*70)

# Initialize EasyOCR
print("\n📚 Loading EasyOCR (Thai + English)...")
reader = easyocr.Reader(['th', 'en'], gpu=False)
print("✓ EasyOCR loaded\n")

# Load image
img_path = '../results/plate_crops/plate_1.jpg'
img = cv2.imread(img_path)

if img is None:
    print(f"❌ Cannot read image: {img_path}")
    print("\nPlease ensure the image is at: ./data/test_ocr_plate.jpg")
    exit()

print(f"📸 Image loaded: {img.shape[1]}x{img.shape[0]}")

# Test 1: Full image OCR
print("\n" + "="*70)
print("TEST 1: OCR ON FULL IMAGE")
print("="*70)

results_full = reader.readtext(img)

print(f"\nDetected {len(results_full)} text regions:\n")
for i, (bbox, text, conf) in enumerate(results_full, 1):
    print(f"[{i}] '{text}'")
    print(f"    Confidence: {conf:.2%}")
    print(f"    Position: ({int(bbox[0][0])}, {int(bbox[0][1])})\n")

# Test 2: Crop and enhance plate region
print("="*70)
print("TEST 2: OCR ON CROPPED & ENHANCED PLATE")
print("="*70)

# Manual crop (adjust these coordinates based on your image)
# For this image, plate is roughly in center-bottom
h, w = img.shape[:2]
plate_region = img[int(h*0.4):int(h*0.7), int(w*0.2):int(w*0.8)]

# Preprocessing
gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
upscaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
denoised = cv2.fastNlMeansDenoising(upscaled)
_, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# OCR on enhanced
results_enhanced = reader.readtext(thresh)

print(f"\nDetected {len(results_enhanced)} text regions:\n")
for i, (bbox, text, conf) in enumerate(results_enhanced, 1):
    print(f"[{i}] '{text}'")
    print(f"    Confidence: {conf:.2%}\n")

# Visualization
print("="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Vis 1: Full image with detections
vis_full = img.copy()
for bbox, text, conf in results_full:
    pts = np.array([[int(x), int(y)] for x, y in bbox], dtype=np.int32)
    cv2.polylines(vis_full, [pts], True, (0, 255, 0), 2)
    
    x, y = int(bbox[0][0]), int(bbox[0][1]) - 10
    label = f"{text} ({conf:.1%})"
    cv2.putText(vis_full, label, (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Save results
output_dir = Path('./results/ocr_tests')
output_dir.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(output_dir / 'full_image_ocr.jpg'), vis_full)
cv2.imwrite(str(output_dir / 'plate_cropped.jpg'), plate_region)
cv2.imwrite(str(output_dir / 'plate_preprocessed.jpg'), thresh)

print(f"\n✅ Results saved to: {output_dir}/")
print("   - full_image_ocr.jpg")
print("   - plate_cropped.jpg")
print("   - plate_preprocessed.jpg")

# Summary
print("\n" + "="*70)
print("EXPECTED TEXT:")
print("="*70)
print("Top line:    ๙กน 2816")
print("Bottom line: กรุงเทพมหานคร")
print("\nDid OCR detect correctly? Check results above! ✓")
print("="*70)
