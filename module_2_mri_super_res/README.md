# Module 2: MRI Super-Resolution

This module implements a **4× Super-Resolution** pipeline specifically designed for **Brain MRI** scans using the **MIRAM (Masked Image Reconstruction Across Multiple Scales)** architecture combined with **SRGAN**.

## 📋 Overview

The module focuses on enhancing low-resolution MRI scans to high-resolution outputs while preserving critical anatomical details. It uses mask-focused training to prioritize brain tissue regions over background.

### Key Components

| File | Description |
|------|-------------|
| `config.py` | All configuration settings and hyperparameters |
| `dataset.py` | Data loading, preprocessing, and LR/HR pair generation |
| `models.py` | Generator (MIRAM-SR) and Discriminator architectures |
| `losses.py` | Perceptual, Charbonnier, Edge, and Adversarial losses |
| `train.py` | Main training loop with warmup strategy |
| `enhance.py` | Single image enhancement script |
| `evaluate.py` | Evaluation and visualization |

## 🏗️ Architecture

### Generator (MIRAM-SR)

```
LR Input (32×32)
     ↓
Head Conv (1 → 64 channels)
     ↓
Residual Body:
├── ResidualBlock × 8
├── MIRAM Attention Block (every 3rd block)
│   ├── Channel Attention
│   └── Spatial Attention
     ↓
Upsampling:
├── PixelShuffle 2× (64 → 128)
└── PixelShuffle 2× (128 → 512)
     ↓
Tail Conv (64 → 1)
     ↓
HR Output (128×128) [4× upscaled]
```

### Loss Functions

| Loss | Weight | Purpose |
|------|--------|---------|
| Charbonnier | 1.0 | Robust pixel-wise loss |
| VGG Perceptual | 0.05 | Texture and perceptual quality |
| Adversarial | 0.005 | Realistic image generation |
| Edge | 0.1 | Sharp boundary preservation |

## 🛠️ Installation

```bash
cd module_2_mri_super_res

# Install dependencies
pip install torch torchvision tifffile pillow numpy tqdm tensorboard matplotlib
```

## 🚀 Usage

### Option 1: Local Training

```bash
# 1. Configure paths in config.py
# Update DRIVE_PATH and DATASET_PATH

# 2. Train the model
python train.py

# 3. Evaluate results
python evaluate.py

# 4. Enhance a single image
python enhance.py
```

### Option 2: Google Colab

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create project directory structure
import os
os.makedirs('/content/drive/My Drive/mri_project/Brain_MRI/HR', exist_ok=True)
os.makedirs('/content/drive/My Drive/mri_project/Brain_MRI/LR', exist_ok=True)
os.makedirs('/content/drive/My Drive/mri_project/Brain_MRI/MASK', exist_ok=True)

# Upload your MRI dataset to the HR folder
# The script will auto-generate LR images

# Run training
%cd /content/drive/My Drive/mri_project
!python train.py
```

## ⚙️ Configuration

Edit `config.py` to customize settings:

### Paths
```python
DRIVE_PATH = '/content/drive/My Drive/mri_project'
DATASET_PATH = os.path.join(DRIVE_PATH, 'Brain_MRI')
```

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `CROP_SIZE` | 128 | Input patch size |
| `SCALE_FACTOR` | 4 | Super-resolution scale (4×) |
| `BATCH_SIZE` | 8 | Training batch size |
| `N_EPOCHS` | 200 | Total training epochs |
| `WARMUP_PERCENTAGE` | 0.15 | Pixel-only training (15% of epochs) |
| `PATIENCE` | 20 | Early stopping patience |

### Loss Weights
```python
LAMBDA_PIXEL = 1.0      # Base structure
LAMBDA_VGG = 0.05       # Perceptual quality
LAMBDA_ADV = 0.005      # Adversarial realism
LAMBDA_EDGE = 0.1       # Edge sharpness
```

## 📁 Dataset Structure

```
Brain_MRI/
├── HR/                 # High-resolution images (ground truth)
│   ├── brain_001.tif
│   ├── brain_002.tif
│   └── ...
├── LR/                 # Low-resolution images (auto-generated)
│   ├── brain_001.tif
│   └── ...
└── MASK/               # Brain tissue masks
    ├── brain_001_mask.tif
    └── ...
```

### Auto-Organization

The `dataset.py` script automatically:
1. Scans your dataset folder for images
2. Sorts them into HR, LR, and MASK directories
3. Generates LR images if they don't exist (bicubic downsampling)

## 📊 Expected Outputs

### Training Outputs
```
mri_project/
├── best_generator.pth      # Best model weights
├── checkpoint.pth          # Training checkpoint
├── logs/                   # TensorBoard logs
└── results/               # Visualization outputs
```

### Monitor Training
```bash
# Start TensorBoard
tensorboard --logdir=/path/to/mri_project/logs
```

## 🔬 Technical Details

### Warmup Strategy

The training uses a warmup phase:
- **Epochs 0-30 (15%)**: Pixel loss only (Charbonnier)
  - Stabilizes generator before adversarial training
  - Prevents mode collapse
- **Epochs 31-200**: Full loss combination
  - Adds VGG perceptual, adversarial, and edge losses

### MIRAM Block

```python
class MIRAMBlock:
    # Channel Attention
    - Global Average Pooling
    - FC layers with ReLU
    - Sigmoid gating
    
    # Spatial Attention
    - Max/Mean pooling across channels
    - Conv layer with Sigmoid
    - Element-wise multiplication
```

### 16-bit Medical Image Support

The module preserves 16-bit precision for medical images:
```python
# Loading
if img.max() > 255: 
    img /= 65535.0  # 16-bit normalization
else: 
    img /= 255.0    # 8-bit normalization

# Saving
tifffile.imwrite(path, (img * 65535).astype(np.uint16))
```

## 📈 Training Progress

Example training output:
```
🚀 STARTING 4X SR TRAINING | Device: cuda
📁 Dataset Loaded: 2847 pairs.
Epoch 1/200: 100%|██████████| 356/356 [02:34<00:00] G:0.0234
   ✅ Val PSNR: 24.56 dB
   💾 Best Model Saved
Epoch 2/200: 100%|██████████| 356/356 [02:31<00:00] G:0.0198
   ✅ Val PSNR: 25.12 dB
   💾 Best Model Saved
...
```

## 🎯 Single Image Enhancement

```bash
python enhance.py
# Enter path when prompted: /path/to/your/image.tif
# Output: output.tif (4× upscaled)
```

Or programmatically:
```python
from enhance import enhance
enhance('/path/to/input.tif', '/path/to/output.tif')
```

## ⚠️ Troubleshooting

### "Dataset Loaded: 0 pairs"
- Ensure images are in the HR folder
- Check that corresponding masks exist
- Verify file extensions match supported formats

### GPU Out of Memory
```python
# In config.py, reduce:
BATCH_SIZE = 4
CROP_SIZE = 64
```

### Slow Training on Colab
- Enable GPU: Runtime → Change runtime type → GPU
- Use Colab Pro for better GPUs and longer sessions

### TensorBoard Not Showing Data
```python
# Ensure logs are being written
import os
print(os.listdir(LOG_DIR))  # Should show event files
```

## 📚 References

- [SRGAN: Photo-Realistic Single Image Super-Resolution](https://arxiv.org/abs/1609.04802)
- [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- [Deep Learning for Medical Image Super-Resolution](https://arxiv.org/abs/2002.03977)
