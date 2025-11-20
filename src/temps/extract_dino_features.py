"""
Extract DINO features - with verbose logging
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle
import sys

print("="*70)
print("DINO FEATURE EXTRACTION")
print("="*70)

# Check GPU
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Check dataset exists
print("\nChecking dataset...")
train_dir = Path('../data/helmet_dataset/train/images')
if not train_dir.exists():
    print(f"❌ Train directory not found: {train_dir}")
    sys.exit(1)

train_images = list(train_dir.glob('*.[jp][pn][g]'))
print(f"✓ Found {len(train_images)} training images")

val_dir = Path('../data/helmet_dataset/valid/images')
val_images = list(val_dir.glob('*.[jp][pn][g]'))
print(f"✓ Found {len(val_images)} validation images")

# Load DINO model
print("\n" + "="*70)
print("Loading DINO model (this may take a few minutes)...")
print("="*70)

try:
    # Try to load DINO
    print("Downloading DINO ViT-S/16 from Facebook Research...")
    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', verbose=True)
    print("✓ DINO model loaded successfully!")
    
except Exception as e:
    print(f"\n❌ Error loading DINO: {e}")
    print("\nTrying alternative approach...")
    
    # Alternative: try loading with force_reload
    try:
        dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', 
                             force_reload=True, verbose=True)
        print("✓ DINO loaded with force_reload")
    except Exception as e2:
        print(f"❌ Still failed: {e2}")
        print("\nPlease check internet connection and try:")
        print("  pip install timm --break-system-packages")
        sys.exit(1)

dino.eval()

# Move to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nMoving model to device: {device}")
dino = dino.to(device)
print("✓ Model ready")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image_path):
    """Extract DINO features from image"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = dino(img_tensor)
        
        return features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Test on one image first
print("\n" + "="*70)
print("Testing on one image...")
print("="*70)
test_img = train_images[0]
print(f"Test image: {test_img.name}")
test_features = extract_features(test_img)
if test_features is not None:
    print(f"✓ Feature shape: {test_features.shape}")
    print(f"✓ Feature extraction working!")
else:
    print("❌ Feature extraction failed")
    sys.exit(1)

# Process training set (limit to 300 for speed)
print("\n" + "="*70)
print("Extracting features from training set...")
print("="*70)

num_train = min(300, len(train_images))  # Limit for speed
print(f"Processing {num_train} training images (of {len(train_images)} total)")

features = []
labels = []

for img_path in tqdm(train_images[:num_train], desc="Train"):
    # Get label
    label_path = img_path.parent.parent / 'labels' / f'{img_path.stem}.txt'
    
    if label_path.exists():
        with open(label_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                label = int(first_line.split()[0])
                
                feature = extract_features(img_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(label)

# Convert to arrays
features = np.array(features)
labels = np.array(labels)

print(f"\n✓ Extracted {len(features)} feature vectors")
print(f"  Feature shape: {features.shape}")
print(f"  Labels shape: {labels.shape}")
print(f"  Class 0 (without_helmet): {np.sum(labels == 0)}")
print(f"  Class 1 (with_helmet): {np.sum(labels == 1)}")

# Save training features
output_dir = Path('../models/dino')
output_dir.mkdir(parents=True, exist_ok=True)

train_output = output_dir / 'train_features.pkl'
with open(train_output, 'wb') as f:
    pickle.dump({'features': features, 'labels': labels}, f)

print(f"\n✓ Training features saved: {train_output}")

# Process validation set
print("\n" + "="*70)
print("Extracting features from validation set...")
print("="*70)
print(f"Processing {len(val_images)} validation images")

val_features = []
val_labels = []

for img_path in tqdm(val_images, desc="Valid"):
    label_path = img_path.parent.parent / 'labels' / f'{img_path.stem}.txt'
    
    if label_path.exists():
        with open(label_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                label = int(first_line.split()[0])
                
                feature = extract_features(img_path)
                if feature is not None:
                    val_features.append(feature)
                    val_labels.append(label)

val_features = np.array(val_features)
val_labels = np.array(val_labels)

print(f"\n✓ Extracted {len(val_features)} validation feature vectors")
print(f"  Feature shape: {val_features.shape}")
print(f"  Class 0: {np.sum(val_labels == 0)}")
print(f"  Class 1: {np.sum(val_labels == 1)}")

# Save validation features
val_output = output_dir / 'val_features.pkl'
with open(val_output, 'wb') as f:
    pickle.dump({'features': val_features, 'labels': val_labels}, f)

print(f"\n✓ Validation features saved: {val_output}")

print("\n" + "="*70)
print("✅ FEATURE EXTRACTION COMPLETE!")
print("="*70)
print(f"\nOutput files:")
print(f"  {train_output}")
print(f"  {val_output}")
print("\nNext: Run train_dino_classifier.py")