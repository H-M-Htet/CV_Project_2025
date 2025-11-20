from detection.yolo_detector import YOLODetector
from association.associator import ObjectAssociator
from detection.visualizer import DetectionVisualizer
from ocr.thai_ocr import ThaiPlateOCR
import cv2

print("FULL PIPELINE TEST\n")

detector = YOLODetector(
    motorcycle_model_path='yolov8n.pt',
    helmet_model_path='../models/yolo/best0.1.pt',
    plate_model_path='../models/plate/best.pt'
)
associator = ObjectAssociator(max_distance=300)
visualizer = DetectionVisualizer()
ocr = ThaiPlateOCR(use_gpu=True)

img = cv2.imread('../data/test_videos/Nohelmet_Plate.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect
motorcycles, riders, plates = detector.detect_all(img)
violations = associator.process_frame(motorcycles, riders, plates)

print(f"Violations: {len(violations)}\n")

# OCR
for i, v in enumerate(violations):
    print(f"Violation {i+1}:")
    print(f"  Rider confidence: {v['rider_conf']:.2%}")
    
    if v.get('plate_bbox'):
        print(f"  Plate detected: YES")
        result = ocr.extract_text(img_rgb, v['plate_bbox'])
        if result:
            print(f"  Plate text: '{result['text']}'")
            print(f"  OCR confidence: {result['confidence']:.2%}")
            v['plate_text'] = result['text']
        else:
            print(f"  Plate text: Could not read")
    else:
        print(f"  Plate detected: NO")

# Visualize
result = visualizer.draw_violations(img_rgb, violations)
cv2.imwrite('../results/final_test.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

print(f"\n✅ Saved: results/final_test.jpg")
print(f"✅ FULL PIPELINE WORKING!")
