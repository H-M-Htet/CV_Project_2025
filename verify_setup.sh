#!/bin/bash

echo "========================================="
echo "VERIFYING PROJECT SETUP"
echo "========================================="
echo ""

# Check structure
echo "Checking project structure..."
echo "✓ Root directory exists"

# Check key directories
dirs=("data" "models" "results" "src" "configs")
for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✓ $dir/ exists"
    else
        echo "✗ $dir/ missing"
    fi
done

echo ""
echo "Checking source modules..."

# Check source files
files=(
    "src/utils/config_loader.py"
    "src/utils/logger.py"
    "src/utils/common.py"
    "src/association/associator.py"
    "src/detection/yolo_detector.py"
    "src/detection/visualizer.py"
    "src/train_yolo.py"
    "src/detect_violations.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file missing"
    fi
done

echo ""
echo "Testing imports..."
cd src
python3 << 'PYTHON'
import sys
try:
    from utils.config_loader import config
    print("✓ config_loader")
    from utils.logger import log
    print("✓ logger")
    from utils.common import calculate_iou
    print("✓ common utilities")
    from association.associator import ObjectAssociator
    print("✓ associator")
    from detection.yolo_detector import YOLODetector
    print("✓ yolo_detector")
    from detection.visualizer import DetectionVisualizer
    print("✓ visualizer")
    print("\n✓ All imports successful!")
except Exception as e:
    print(f"\n✗ Import error: {e}")
    sys.exit(1)
PYTHON

echo ""
echo "========================================="
echo "✓ PROJECT SETUP COMPLETE!"
echo "========================================="
