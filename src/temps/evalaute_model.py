"""
Quick evaluation script - generates all metrics
"""
from ultralytics import YOLO
from pathlib import Path
import json
from datetime import datetime

print("="*70)
print("MODEL EVALUATION")
print("="*70)

# Load model
model_path = '../models/yolo/best0.1.pt'
model = YOLO(model_path)

print(f"\nModel: {model_path}")
print(f"Classes: {model.names}")

# Validate on test/validation set
print("\nRunning validation...")
data_yaml = '../data/helmet_dataset/data.yaml'
metrics = model.val(data=data_yaml, split='val')

# Extract metrics
results = {
    'timestamp': datetime.now().isoformat(),
    'model': model_path,
    'metrics': {
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'f1': float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr))
    },
    'per_class': {}
}

# Per-class metrics
class_names = ['without_helmet', 'with_helmet']
if hasattr(metrics.box, 'maps'):
    for i, name in enumerate(class_names):
        results['per_class'][name] = {
            'mAP50': float(metrics.box.maps[i]),
            'precision': float(metrics.box.p[i]) if hasattr(metrics.box, 'p') else None,
            'recall': float(metrics.box.r[i]) if hasattr(metrics.box, 'r') else None
        }

# Print results
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"mAP@0.5:     {results['metrics']['mAP50']:.3f}")
print(f"mAP@0.5:0.95: {results['metrics']['mAP50-95']:.3f}")
print(f"Precision:   {results['metrics']['precision']:.3f}")
print(f"Recall:      {results['metrics']['recall']:.3f}")
print(f"F1-Score:    {results['metrics']['f1']:.3f}")

print("\nPer-Class Performance:")
for class_name, metrics_dict in results['per_class'].items():
    print(f"\n  {class_name}:")
    print(f"    mAP@0.5: {metrics_dict['mAP50']:.3f}")

# Save results
output_dir = Path('../results/metrics')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'yolo_evaluation.json'

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved: {output_path}")
print("="*70)
