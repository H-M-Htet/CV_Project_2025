"""
Visualize DINO attention maps
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("Loading DINO for attention visualization...")

# Load DINO
dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
dino.eval()

# Get a test image
test_img_path = list(Path('../data/helmet_dataset/valid/images').glob('*'))[0]
print(f"Visualizing: {test_img_path.name}")

# Load and preprocess
img = Image.open(test_img_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.5])
])

img_tensor = transform(img).unsqueeze(0)

# Get attention weights
with torch.no_grad():
    # Forward pass through attention layers
    attentions = dino.get_last_selfattention(img_tensor)

# Reshape attention (remove CLS token, reshape to grid)
nh = attentions.shape[1]  # Number of attention heads
attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

# Get dimensions
w_featmap = img_tensor.shape[-2] // 16
h_featmap = img_tensor.shape[-1] // 16
attentions = attentions.reshape(nh, w_featmap, h_featmap)

# Plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('DINO Self-Attention Maps', fontsize=16)

# Original image
axes[0, 0].imshow(img)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

# Show first 5 attention heads
for idx in range(min(5, nh)):
    ax = axes[(idx+1)//3, (idx+1)%3]
    
    # Resize attention to image size
    attn = attentions[idx].numpy()
    attn = np.array(Image.fromarray(attn).resize(img.size, Image.BICUBIC))
    
    # Overlay on image
    ax.imshow(img, alpha=0.5)
    ax.imshow(attn, cmap='jet', alpha=0.5)
    ax.set_title(f'Attention Head {idx+1}')
    ax.axis('off')

plt.tight_layout()
output_path = Path('../results/visualizations/dino_attention.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Attention maps saved: {output_path}")
plt.show()
