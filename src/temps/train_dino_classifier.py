"""
Train classifier on DINO features
"""
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from pathlib import Path
import json
from datetime import datetime

print("="*70)
print("DINO CLASSIFIER TRAINING")
print("="*70)

# Load features
dino_dir = Path('../models/dino')

print("\nLoading DINO features...")
with open(dino_dir / 'train_features.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open(dino_dir / 'val_features.pkl', 'rb') as f:
    val_data = pickle.load(f)

X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = val_data['features'], val_data['labels']

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# Train classifier
print("\nTraining logistic regression classifier...")
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
print("Evaluating...")
train_pred = clf.predict(X_train)
val_pred = clf.predict(X_val)

train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)

val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
    y_val, val_pred, average='weighted'
)

# Results
results = {
    'timestamp': datetime.now().isoformat(),
    'model': 'DINO + Logistic Regression',
    'metrics': {
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'precision': float(val_precision),
        'recall': float(val_recall),
        'f1': float(val_f1)
    }
}

print("\n" + "="*70)
print("DINO RESULTS")
print("="*70)
print(f"Train Accuracy: {train_acc:.3f}")
print(f"Val Accuracy:   {val_acc:.3f}")
print(f"Precision:      {val_precision:.3f}")
print(f"Recall:         {val_recall:.3f}")
print(f"F1-Score:       {val_f1:.3f}")
print("="*70)

# Save
output_path = Path('../results/metrics/dino_evaluation.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

# Save model
with open(dino_dir / 'classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

print(f"\n✓ Results saved: {output_path}")
print(f"✓ Model saved: {dino_dir / 'classifier.pkl'}")
