"""
Compare YOLO vs DINO
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt

print("Generating comparison plots...")

# Load results
yolo_results = json.load(open('../results/metrics/yolo_evaluation.json'))
dino_results = json.load(open('../results/metrics/dino_evaluation.json'))

# Extract metrics
methods = ['YOLO', 'DINO']
metrics = ['precision', 'recall', 'f1']

yolo_vals = [yolo_results['metrics'][m] for m in metrics]
dino_vals = [dino_results['metrics'][m] for m in metrics]

# Create comparison plot
fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(metrics))
width = 0.35

ax.bar([i - width/2 for i in x], yolo_vals, width, label='YOLO (Supervised)', color='#2E86AB')
ax.bar([i + width/2 for i in x], dino_vals, width, label='DINO (Self-Supervised)', color='#A23B72')

ax.set_ylabel('Score')
ax.set_title('YOLO vs DINO: Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels([m.capitalize() for m in metrics])
ax.legend()
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
output_path = Path('../results/metrics/yolo_vs_dino_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Comparison plot saved: {output_path}")

plt.show()
