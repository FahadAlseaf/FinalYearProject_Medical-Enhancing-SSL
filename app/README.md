# App: Desktop GUI for Medical Image Enhancement

A user-friendly **Tkinter-based desktop application** for enhancing medical images using pre-trained MIRAM and SRGAN models.

## 📋 Overview

This application provides an interactive interface for:
- Loading and enhancing medical images (X-ray, MRI, CT)
- Switching between different enhancement models
- Adjusting enhancement parameters in real-time
- Batch processing multiple images
- Visualizing results with live preview

## 🖼️ Features

| Feature | Description |
|---------|-------------|
| **Model Selection** | Switch between MIRAM+SRGAN and MRI Enhancer models |
| **Live Preview** | Real-time preview of enhancement parameters |
| **Parameter Control** | Adjust sharpness, contrast, and brightness |
| **Batch Processing** | Enhance multiple images at once |
| **ViT Visualization** | Compare baseline vs SSL pre-trained results |

## 🛠️ Installation

```bash
cd app

# Install dependencies
pip install torch torchvision pillow numpy matplotlib timm tkinter
```

### Required Model Files

Place these pre-trained model files in the `app/` directory:

| File | Description |
|------|-------------|
| `SR_best.pth` | MIRAM restoration model weights |
| `srgan_best.pth` | SRGAN generator weights |
| `mri_enhancer_best.pth` | MRI-specific enhancer weights |
| `best_generator.pth` | SSL pre-trained ViT model |
| `best_generator_baseline.pth` | Baseline ViT model |

## 🚀 Usage

### Launch the Application

```bash
python app.py
```

### Application Interface

```
┌────────────────────────────────────────────────────┐
│     AI Image Enhancement Tool                      │
├────────────────────────────────────────────────────┤
│  [M1: MIRAM+SRGAN]  [M2: MRI Enhancer]            │
│                                                    │
│  Model: EnhancedMIRAM + SRGAN (M1)                │
├────────────────────────────────────────────────────┤
│  [Select Images]    [Select Output Folder]        │
│                                                    │
│  Output: /path/to/output                          │
│                                                    │
│  ┌──────────────────────────────────────────┐     │
│  │ image1.png                                │     │
│  │ image2.png                                │     │
│  └──────────────────────────────────────────┘     │
│                                                    │
│  [Select Preview Example]                         │
│                                                    │
│  Sharpness:    [═══════════●═══]  8.0            │
│  Contrast:     [═══●═══════════]  1.1            │
│  Brightness:   [══●════════════]  1.05           │
│  Preview Size: [═══════●═══════]  1.0×           │
│                                                    │
│  [Reset to Default]                               │
│                                                    │
│  [        Start Enhancement        ]              │
│                                                    │
│  [      Visualize ViT Results      ]              │
└────────────────────────────────────────────────────┘
```

## 📖 Step-by-Step Guide

### 1. Select a Model

Click one of the model buttons:
- **M1: MIRAM+SRGAN** - Two-stage enhancement (restoration + super-resolution)
- **M2: MRI Enhancer** - 4× super-resolution optimized for brain MRI

### 2. Select Images

1. Click **"Select Images"**
2. Choose one or more medical images
3. Images appear in the list box

### 3. Select Output Folder

1. Click **"Select Output Folder"**
2. Choose where to save enhanced images

### 4. Preview Enhancement (Optional)

1. Click **"Select Preview Example"**
2. Choose a sample image
3. A preview window opens showing real-time enhancement
4. Adjust sliders to fine-tune parameters:
   - **Sharpness**: 0.0 - 50.0 (default: 1.0)
   - **Contrast**: 0.0 - 4.0 (default: 1.0)
   - **Brightness**: 0.0 - 4.0 (default: 1.0)
   - **Preview Size**: 0.2× - 2.0× (default: 1.0×)

### 5. Start Enhancement

1. Click **"Start Enhancement"**
2. Wait for processing to complete
3. Enhanced images are saved with prefix (M1_ or M2_)

### 6. Visualize ViT Results (Optional)

1. Select a preview image first
2. Click **"Visualize ViT Results"**
3. A matplotlib window shows:
   - Low-res input (56×56)
   - Baseline output (from scratch)
   - SSL pre-trained output
   - Ground truth (224×224)

## ⚙️ Model Details

### M1: MIRAM + SRGAN (Two-Stage)

```
Input Image
    ↓
EnhancedMIRAM (Restoration)
├── Multi-scale feature extraction
├── Channel attention
└── Spatial attention
    ↓
EnhancedGenerator (Super-Resolution)
├── Attention residual blocks
├── PixelShuffle upsampling
└── 4× resolution increase
    ↓
Enhanced Output
```

### M2: MRI Enhancer

```
Input Image
    ↓
GeneratorMIRAMSR
├── Residual blocks with MIRAM attention
├── 4× upsampling
└── Grayscale optimization
    ↓
4× Super-Resolved Output
```

## 📁 File Structure

```
app/
├── app.py              # Main application
├── SR.py               # Model definitions (EnhancedMIRAM, EnhancedGenerator)
├── mri.py              # MRI-specific model (GeneratorMIRAMSR)
├── SR_best.pth         # MIRAM weights
├── srgan_best.pth      # SRGAN weights
├── mri_enhancer_best.pth    # MRI enhancer weights
├── best_generator.pth       # SSL ViT weights
└── best_generator_baseline.pth  # Baseline ViT weights
```

## 📊 Output Naming

Enhanced images are saved with model prefix:

| Model | Prefix | Example |
|-------|--------|---------|
| M1 (MIRAM+SRGAN) | `M1_enh_` | `M1_enh_brain_scan.png` |
| M2 (MRI Enhancer) | `M2_enh_` | `M2_enh_brain_scan.png` |

## ⚠️ Troubleshooting

### "Model file not found"

Ensure all `.pth` files are in the `app/` directory:
```bash
ls app/*.pth
# Should show: SR_best.pth, srgan_best.pth, etc.
```

### Preview Window Not Appearing

The preview opens in a separate window. Check:
- Taskbar for hidden windows
- Multiple monitor setups

### Slow Preview Updates

Preview uses a smaller inference size (max 512px) for speed. For full-quality results, use the batch enhancement.

### CUDA Out of Memory

The app automatically uses GPU if available. For large images:
```python
# The app limits preview size to 512px
PREVIEW_MAX = 512
```

### Tkinter Not Found (Linux)

```bash
sudo apt-get install python3-tk
```

### Tkinter Not Found (macOS)

```bash
brew install python-tk
```

## 🎨 Enhancement Tips

### For X-ray Images
- **Sharpness**: 6-10 (enhance bone edges)
- **Contrast**: 1.2-1.4 (improve tissue distinction)
- **Brightness**: 1.0-1.1 (slight boost)

### For MRI Scans
- **Sharpness**: 2-4 (avoid over-sharpening soft tissue)
- **Contrast**: 1.1-1.2 (preserve gray matter detail)
- **Brightness**: 0.95-1.05 (maintain original exposure)

### For CT Scans
- **Sharpness**: 4-8 (enhance structural details)
- **Contrast**: 1.1-1.3 (improve organ boundaries)
- **Brightness**: 1.0 (usually well-calibrated)

## 📚 Dependencies

```python
# Core
torch>=2.0.0
torchvision>=0.15.0
pillow>=10.0.0
numpy>=1.24.0

# GUI
tkinter  # Usually included with Python

# Visualization
matplotlib>=3.7.0

# Model
timm>=0.9.0
```

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10 / Ubuntu 20.04 / macOS 11 | Latest versions |
| **Python** | 3.8 | 3.10+ |
| **RAM** | 8 GB | 16 GB+ |
| **GPU** | CUDA-capable (optional) | NVIDIA RTX 3060+ |
| **Display** | 1280×720 | 1920×1080+ |
