# Module 4: Enhanced SRGAN with Imbalance Handling

This module provides a complete **Super-Resolution GAN (SRGAN)** pipeline with advanced features for handling **imbalanced medical datasets**, **attention mechanisms**, and **medical imaging-specific augmentation**.

## 📋 Overview

This module addresses common challenges in medical image super-resolution:
- **Imbalanced datasets** (more normal samples than abnormal)
- **Diverse image quality** requirements
- **Preservation of diagnostic features**

### Key Features

| Feature | Description |
|---------|-------------|
| **Focal Loss** | Down-weights easy samples, focuses on hard examples |
| **Weighted Sampling** | Balances class distribution during training |
| **Medical Augmentation** | Rotation, translation, contrast adjustments |
| **Minority Oversampling** | Augments underrepresented classes |
| **Attention Mechanisms** | Channel and spatial attention in generator |
| **Hybrid Loss** | Charbonnier + SSIM + LPIPS combination |

## 🏗️ Architecture

### Enhanced Generator

```
LR Input
    ↓
Head Conv (1 → 64) + PReLU
    ↓
Attention Residual Blocks × 16
├── Conv3×3 → BN → PReLU → Conv3×3 → BN
├── Channel Attention (Squeeze-Excitation)
└── Spatial Attention (Max/Avg pooling)
    ↓
Post Residual Conv
    ↓
Global Residual Connection
    ↓
Upsample Blocks × 2 (4× total)
├── Conv → PixelShuffle(2×) → PReLU
└── Channel Attention
    ↓
Tail Conv (64 → 1)
    ↓
HR Output (4× resolution)
```

### Enhanced MIRAM Module

```
Input
    ↓
Head Conv
    ↓
Multi-Scale Feature Extraction
├── Conv 3×3 (local features)
├── Conv 5×5 (medium features)
└── Conv 7×7 (global features)
    ↓
Concatenate → Channel Attention → Spatial Attention
    ↓
Fusion Conv
    ↓
Residual + Tail Conv
    ↓
Output
```

## 🛠️ Installation

```bash
cd module_4_SR

# Install dependencies
pip install torch torchvision pillow numpy tqdm matplotlib seaborn
```

## 🚀 Usage

### Interactive Mode

```bash
python SR.py
```

You'll see a menu:
```
🧠 ENHANCED MIRAM + SRGAN Medical Image Enhancement System
============================================================

1️⃣  Train a new model
2️⃣  Enhance medical images

Enter your choice (1 or 2):
```

### Training

```bash
# Option 1: From zip file
python SR.py
> 1
> /path/to/dataset.zip
> 100    # epochs
> 8      # batch size
> 2.0    # oversample ratio
> y      # handle imbalance

# Option 2: From folder
python SR.py
> 1
> /path/to/train_folder
> ...
```

### Enhancement

```bash
python SR.py
> 2
> /path/to/input_folder
> /path/to/output_folder
> n      # or 'y' to customize settings
```

### Programmatic Usage

```python
from SR import train_miram_srgan, enhance_miram_srgan

# Train
train_miram_srgan(
    data_path="/path/to/train",
    max_epochs=100,
    batch_size=8,
    handle_imbalance=True,
    oversample_ratio=2.0
)

# Enhance
enhance_miram_srgan(
    input_folder="/path/to/input",
    output_folder="/path/to/output",
    sharpen_factor=8,
    contrast_factor=1.1,
    brightness_factor=1.05
)
```

## ⚙️ Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_epochs` | 100 | Maximum training epochs |
| `batch_size` | 8 | Training batch size |
| `patience_epochs` | 10 | Early stopping patience |
| `min_improvement` | 0.002 | Minimum metric improvement |
| `handle_imbalance` | True | Enable imbalance handling |
| `oversample_ratio` | 2.0 | Minority class oversampling factor |

### Enhancement Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sharpen_factor` | 8 | Sharpness enhancement (1.0 = no change) |
| `contrast_factor` | 1.1 | Contrast boost |
| `brightness_factor` | 1.05 | Brightness adjustment |

## 📁 Dataset Structure

### Supported Formats

```python
SUPPORTED_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", 
    ".tif", ".tiff", ".bmp", 
    ".gif", ".heif", ".heic", ".webp"
)
```

### Auto-Organization

The script automatically organizes datasets:

```
# Before
dataset.zip
└── images/
    ├── image001.png
    └── image002.png

# After extraction (80/20 split)
dataset/
├── train/
│   ├── image001.png
│   └── ...
└── test/
    ├── image101.png
    └── ...
```

### With Imbalance Handling

```
dataset/
├── train/           # Original training data
└── train_balanced/  # After oversampling
    ├── image001.png
    ├── image001_aug_0.png   # Augmented minority
    ├── image001_aug_1.png
    └── ...
```

## 📊 Metrics & Outputs

### Training Metrics

| Metric | Description |
|--------|-------------|
| **PSNR** | Peak Signal-to-Noise Ratio (dB) |
| **SSIM** | Structural Similarity Index (0-1) |
| **Accuracy** | Pixel-wise accuracy within threshold |
| **Loss_G** | Generator total loss |
| **Loss_D** | Discriminator loss |

### Output Files

```
training_results/
├── epoch_5.png       # Sample outputs every 5 epochs
├── epoch_10.png
└── ...

# Model weights
miram_best.pth        # Best MIRAM weights
srgan_best.pth        # Best Generator weights
disc_best.pth         # Best Discriminator weights
```

## 🔬 Technical Details

### Hybrid Loss Function

```python
Total_Loss = λ_char × L_Charbonnier + λ_ssim × L_SSIM + λ_lpips × L_LPIPS

where:
- L_Charbonnier = √(x² + ε²)  # Robust L1 variant
- L_SSIM = 1 - SSIM(pred, target)
- L_LPIPS = VGG feature distance (layers 4, 9, 16, 23)
```

### Focal Loss for Imbalance

```python
FL(pt) = -α(1-pt)^γ × log(pt)

where:
- α = 1.0 (class weight)
- γ = 2.0 (focusing parameter)
- pt = probability of correct class
```

### Medical Image Augmentation

| Augmentation | Range | Purpose |
|--------------|-------|---------|
| Rotation | ±10° | Orientation invariance |
| Translation | ±10px | Position invariance |
| Scale | 90-110% | Size invariance |
| Horizontal Flip | 50% | Symmetry |
| Contrast | 0.8-1.2× | Imaging variability |
| Brightness | 0.8-1.2× | Lighting conditions |

### Gradient Penalty (WGAN-GP Style)

```python
# Interpolate between real and fake
interpolates = α × real + (1-α) × fake

# Compute gradient norm
gradient_penalty = (||∇D(interpolates)||₂ - 1)²
```

## 📈 Training Progress

```
🔧 Using device: cuda

📊 Class Distribution:
   Class 0: 1234 samples (weight: 0.4052)
   Class 1: 456 samples (weight: 1.0965)

📊 Using Weighted Random Sampler for class imbalance...
🚀 Starting Enhanced Training with Imbalance Handling...

Epoch [1/100] PSNR: 22.45, SSIM: 0.7823, Acc: 87.23%
   Loss_G: 0.0456 | Loss_D: 0.3421
...
✅ Improvement detected — saved best models.
   Best PSNR: 28.34 | Best SSIM: 0.9123 | Best Acc: 94.56%
```

## ⚠️ Troubleshooting

### "No images found in the zip"
```python
# Check zip contents
import zipfile
with zipfile.ZipFile('dataset.zip', 'r') as z:
    print(z.namelist()[:10])
```

### GPU Memory Issues
```python
# Reduce batch size
train_miram_srgan(..., batch_size=4)

# Reduce image size (modify in augmentation)
transforms.Resize((128, 128))  # Instead of 256
```

### Discriminator Dominates Generator
- Reduce discriminator learning rate
- Increase gradient penalty weight
- Use label smoothing

### Oversampling Creates Artifacts
```python
# Reduce augmentation probability
MedicalImageAugmentation(p=0.3)  # Instead of 0.8
```

## 📚 References

- [Photo-Realistic SRGAN](https://arxiv.org/abs/1609.04802)
- [CBAM: Attention Module](https://arxiv.org/abs/1807.06521)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [The Perception-Distortion Tradeoff](https://arxiv.org/abs/1711.06077)
- [LPIPS: Perceptual Similarity](https://arxiv.org/abs/1801.03924)
