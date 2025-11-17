# Helmet Detection System: YOLO vs DINO Comparison

## Project Overview
This project compares two approaches for detecting motorcycle helmet violations:
1. **YOLO (Supervised)**: End-to-end object detection
2. **DINO (Self-Supervised)**: Feature extraction + classifier

## Objectives
- Detect motorcycles and riders in traffic videos
- Classify helmet vs no-helmet using two different approaches
- Extract Thai license plates from violations
- Compare supervised vs self-supervised learning paradigms

## Project Structure
```
helmet_detection_project/
├── data/                          # Datasets
│   ├── helmet_dataset/           # Training data
│   ├── test_videos/              # Test traffic videos
│   └── thai_plates/              # License plate samples
├── models/                        # Saved models
│   ├── yolo/                     # YOLO checkpoints
│   └── dino/                     # DINO features & classifier
├── src/                          # Source code
│   ├── 1_data_preparation.py    # Download & prepare data
│   ├── 2_train_yolo.py          # Train YOLO model
│   ├── 3_extract_dino.py        # Extract DINO features
│   ├── 4_train_classifier.py    # Train DINO classifier
│   ├── 5_plate_detection.py     # License plate detection
│   ├── 6_pipeline.py            # Full inference pipeline
│   └── 7_evaluation.py          # Compare both methods
├── results/                      # Outputs
│   ├── visualizations/          # Plots, attention maps
│   ├── metrics/                 # Evaluation results
│   └── violations/              # Logged violations
├── notebooks/                    # Jupyter notebooks for exploration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
python src/1_data_preparation.py
```

### 3. Train Models
```bash
# Train YOLO
python src/2_train_yolo.py

# Extract DINO features & train classifier
python src/3_extract_dino.py
python src/4_train_classifier.py
```

### 4. Run Evaluation
```bash
python src/7_evaluation.py
```

### 5. Run Full Pipeline
```bash
python src/6_pipeline.py --video data/test_videos/traffic.mp4
```

## Hardware Requirements
- GPU: 4GB+ VRAM (works on shared cluster)
- RAM: 8GB+
- Storage: 5GB for datasets

## Timeline (7 Days)
- Day 1: Setup + Data collection
- Day 2: Train YOLO baseline
- Day 3: Extract DINO features
- Day 4: Train DINO classifier
- Day 5: License plate detection
- Day 6: Evaluation & comparison
- Day 7: Report & demo

## Key Concepts Explained

### YOLO (You Only Look Once)
- **Type**: Supervised learning
- **Architecture**: Single-stage detector (CNN-based)
- **Training**: Needs labeled bounding boxes + class labels
- **Advantage**: Fast, accurate, end-to-end
- **Disadvantage**: Needs lots of labeled data

### DINO (Self-Distillation with NO labels)
- **Type**: Self-supervised learning
- **Architecture**: Vision Transformer (ViT)
- **Training**: Pre-trained on ImageNet without labels
- **Advantage**: General features, less data needed
- **Disadvantage**: Two-stage (features + classifier)

### Why Compare?
This comparison explores:
- Supervised vs self-supervised learning
- Data efficiency
- Transfer learning effectiveness
- Speed vs accuracy tradeoffs

## Expected Results
| Metric | YOLO | DINO+Classifier |
|--------|------|-----------------|
| Accuracy | 90-95% | 75-85% |
| Speed | Real-time | Slower |
| Training Time | 1-2 hours | 30 min (classifier only) |
| Data Needed | 500+ images | 200+ images |

## License
MIT License - Educational Project

## Author
Master's CS Student - Computer Vision Course Project (30% Grade)

## Acknowledgments
- Ultralytics YOLOv8
- Facebook Research DINO
- EasyOCR for Thai text recognition
