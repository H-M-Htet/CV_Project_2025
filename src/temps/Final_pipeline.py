"""
FINAL WORKING PIPELINE - NO MOTORCYCLE DETECTOR NEEDED
"""
from detection.yolo_detector import YOLODetector
from detection.visualizer import DetectionVisualizer
from ocr.thai_ocr import ThaiPlateOCR
import cv2
import numpy as np
from pathlib import Path

print("="*70)
print("FINAL WORKING TEST")
print("="*70)

# Initialize
detector = YOLODetector(
    motorcycle_model_path='yolov8n.pt',
    helmet_model_path='../models/yolov12/bestv12.pt',
    plate_model_path='../models/plate/best_plate.pt'
)
ocr = ThaiPlateOCR(use_gpu=True)

# Load image
img_path = '../data/test_videos/Nohelmet_Plate.jpg'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(f"\n1. Image loaded: {img.shape}")

# Detect (IGNORE motorcycles - they're useless)
print("\n2. Running detection...")
_, riders, plates = detector.detect_all(img)

print(f"\n--- DETECTIONS ---")
print(f"Riders: {len(riders)}")
class_names = ['No_helmet', 'Person_on_Bike', 'Wearing_helmet']
for i, r in enumerate(riders):
    print(f"  {i+1}. {class_names[r['class']]} - conf:{r['conf']:.0%} - bbox:{r['bbox']}")

print(f"\nPlates: {len(plates)}")
for i, p in enumerate(plates):
    print(f"  {i+1}. Plate - conf:{p['conf']:.0%} - bbox:{p['bbox']}")

# Find violations (SIMPLE - NO MOTORCYCLE ASSOCIATION)
print("\n3. Finding violations...")
violations = []

no_helmet = [r for r in riders if r['class'] == 0]
person_on_bike = [r for r in riders if r['class'] == 1]
wearing_helmet = [r for r in riders if r['class'] == 2]

# Rule 1: No_helmet = violation
for r in no_helmet:
    violations.append({
        'rider_bbox': r['bbox'],
        'rider_conf': r['conf'],
        'rider_class': 0
    })

# Rule 2: Person_on_Bike without helmet = violation
for person in person_on_bike:
    # Check if helmet overlaps
    has_helmet = False
    for helmet in wearing_helmet:
        # Simple center check
        px, py = (person['bbox'][0]+person['bbox'][2])/2, (person['bbox'][1]+person['bbox'][3])/2
        hx1, hy1, hx2, hy2 = helmet['bbox']
        if hx1 <= px <= hx2 and hy1 <= py <= hy2:
            has_helmet = True
            break
    
    if not has_helmet:
        violations.append({
            'rider_bbox': person['bbox'],
            'rider_conf': person['conf'],
            'rider_class': 1
        })

print(f"Violations found: {len(violations)}")

# Associate plates with violations (nearest)
print("\n4. Associating plates...")
for v in violations:
    v_center = ((v['rider_bbox'][0]+v['rider_bbox'][2])/2,
                (v['rider_bbox'][1]+v['rider_bbox'][3])/2)
    
    best_plate = None
    min_dist = float('inf')
    
    for plate in plates:
        p_center = ((plate['bbox'][0]+plate['bbox'][2])/2,
                   (plate['bbox'][1]+plate['bbox'][3])/2)
        dist = np.sqrt((v_center[0]-p_center[0])**2 + (v_center[1]-p_center[1])**2)
        
        if dist < min_dist:
            min_dist = dist
            best_plate = plate
    
    if best_plate:
        v['plate_bbox'] = best_plate['bbox']
        v['plate_conf'] = best_plate['conf']
        print(f"  Plate associated (distance: {min_dist:.1f}px)")

# OCR on ALL plates
print("\n5. Running OCR...")
for i, plate in enumerate(plates):
    print(f"\n  Plate {i+1}:")
    
    # Crop with padding
    x1, y1, x2, y2 = map(int, plate['bbox'])
    pad = 5
    x1, y1 = max(0, x1-pad), max(0, y1-pad)
    x2, y2 = min(img.shape[1], x2+pad), min(img.shape[0], y2+pad)
    
    plate_crop = img_rgb[y1:y2, x1:x2]
    
    if plate_crop.size == 0:
        print("    ❌ Empty crop")
        plate['text'] = "N/A"
        continue
    
    print(f"    Crop size: {plate_crop.shape}")
    
    # Upscale if too small
    if plate_crop.shape[0] < 50 or plate_crop.shape[1] < 100:
        scale = max(50/plate_crop.shape[0], 100/plate_crop.shape[1], 3)
        new_h = int(plate_crop.shape[0] * scale)
        new_w = int(plate_crop.shape[1] * scale)
        plate_crop = cv2.resize(plate_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        print(f"    Upscaled to: {plate_crop.shape}")
    
    # Run OCR
    result = ocr.reader.readtext(plate_crop)
    
    if result:
        texts = [text for (bbox, text, conf) in result if conf > 0.3]
        if texts:
            plate['text'] = ' '.join(texts)
            print(f"    ✅ OCR: '{plate['text']}'")
        else:
            plate['text'] = "Low_Conf"
            print(f"    ⚠️  Low confidence")
    else:
        plate['text'] = "No_Text"
        print(f"    ❌ No text detected")

# Add plate text to violations
for v in violations:
    if 'plate_bbox' in v:
        for plate in plates:
            if plate['bbox'] == v['plate_bbox']:
                v['plate_text'] = plate.get('text', 'N/A')
                break

# CLEAN VISUALIZATION
print("\n6. Creating visualization...")
result_img = img_rgb.copy()

# Draw violations (BIG RED BOX)
for v in violations:
    x1, y1, x2, y2 = map(int, v['rider_bbox'])
    cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 4)
    cv2.putText(result_img, "VIOLATION", (x1, y1-10),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

# Draw safe riders (GREEN)
for helmet in wearing_helmet:
    x1, y1, x2, y2 = map(int, helmet['bbox'])
    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(result_img, "SAFE", (x1, y1-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Draw plates with text (BLUE + YELLOW TEXT)
for plate in plates:
    x1, y1, x2, y2 = map(int, plate['bbox'])
    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    text = plate.get('text', 'N/A')
    cv2.putText(result_img, f"PLATE: {text}", (x1, y2+25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# Save
output_path = Path('../results/FINAL_RESULT.jpg')
cv2.imwrite(str(output_path), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

print("\n" + "="*70)
print("✅ COMPLETE!")
print("="*70)
print(f"\nRESULTS:")
print(f"  Riders: {len(riders)}")
print(f"  Violations: {len(violations)}")
print(f"  Plates: {len(plates)}")
print(f"\n📸 Output: {output_path.absolute()}")
print("\nWhat you'll see:")
print("  🔴 RED boxes = Violations")
print("  🟢 GREEN boxes = Safe riders")
print("  🔵 BLUE boxes = Plates")
print("  🟡 YELLOW text = Plate text (or N/A)")
print("="*70)
