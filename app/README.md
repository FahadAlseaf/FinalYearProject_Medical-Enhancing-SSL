# GUI Application: AI Medical Image Enhancement Tool

This folder contains a **Tkinter-based GUI application** that provides a unified interface for all enhancement models in the repository.

## 🌟 Features

- **Model Selection**: Switch between different enhancement pipelines
  - **M1 (MIRAM+SRGAN)**: Two-stage enhancement with 4x super-resolution
  - **M2 (MRI Enhancer)**: Specialized brain MRI enhancement with 4x SR
- **Live Preview**: Real-time preview of enhancement results
- **Parameter Adjustment**: Interactive sliders for Sharpness, Contrast, Brightness
- **Batch Processing**: Enhance multiple images at once
- **ViT Visualization**: Compare SSL-pretrained vs baseline results

## 📂 File Structure

```
app/
├── app.py          # Main GUI application (Tkinter)
└── README.md       # This file
```

## 🔧 Model Architecture

### Model 1: EnhancedMIRAM + EnhancedGenerator (Two-Stage)

```
Input Image
    │
    ▼
┌─────────────────────────────────────┐
│ Stage 1: EnhancedMIRAM              │
│   Multi-scale convolutions (3×3,    │
│   5×5, 7×7) + Channel/Spatial       │
│   Attention + Residual              │
│   Output: Same size, enhanced       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Stage 2: EnhancedGenerator          │
│   16 Attention Residual Blocks      │
│   + PixelShuffle 4× Upsampling      │
│   Output: 4× larger resolution      │
└─────────────────────────────────────┘
    │
    ▼
Enhanced Output (4× resolution)
```

### Model 2: GeneratorMIRAMSR (from Module 2)

Single-stage 4× super-resolution optimized for brain MRI images.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision pillow numpy timm matplotlib
```

### 2. Download Model Weights

Place the following model files in the `app/` directory (or set `MODEL_DIR` environment variable):

| File | Description | Source |
|------|-------------|--------|
| `SR_best.pth` | EnhancedMIRAM weights | Train with SR.py |
| `srgan_best.pth` | EnhancedGenerator weights | Train with SR.py |
| `mri_enhancer_best.pth` | MRI Generator weights | module_2 training |

**Optional (for ViT visualization):**
| File | Description |
|------|-------------|
| `best_generator.pth` | SSL-pretrained ViT model |
| `best_generator_baseline.pth` | Baseline ViT model |

### 3. Run the Application

```bash
cd app
python app.py
```

Or from the repository root:
```bash
python app/app.py
```

## 📖 Usage Guide

### Basic Enhancement

1. Click **"Select Images"** to choose input images
2. Click **"Select Output Folder"** for saving results
3. Select a model: **M1** (MIRAM+SRGAN) or **M2** (MRI Enhancer)
4. Adjust sliders if desired (Sharpness, Contrast, Brightness)
5. Click **"Start Enhancement"** to process all selected images

### Live Preview

1. Click **"Select Preview Example"** to load a single image
2. A preview window will open showing real-time enhancement
3. Adjust sliders to see changes instantly
4. Use the **Preview Size (×)** slider to zoom in/out

### ViT Comparison (SSL vs Baseline)

1. Load a preview image first
2. Ensure `best_generator.pth` and `best_generator_baseline.pth` exist
3. Click **"Visualize ViT Results"**
4. A matplotlib figure will show:
   - Low Resolution Input
   - Baseline (trained from scratch)
   - SSL Pre-trained result
   - Ground Truth

## ⚙️ Configuration

### Environment Variables

```bash
# Set custom model directory
export MODEL_DIR="/path/to/model/weights"

# Example
export MODEL_DIR="./trained_models"
python app/app.py
```

### Default Parameters

| Parameter | Default | Range |
|-----------|---------|-------|
| Sharpness | 1.0 | 0.0 - 50.0 |
| Contrast | 1.0 | 0.0 - 4.0 |
| Brightness | 1.0 | 0.0 - 4.0 |
| Preview Scale | 1.0 | 0.2 - 2.0 |

## 📊 Output Naming

Enhanced images are saved with model prefixes:
- `M1_enh_filename.png` - Enhanced with MIRAM+SRGAN
- `M2_enh_filename.png` - Enhanced with MRI Enhancer

## 🔧 Troubleshooting

**"No model loaded" warning:**
- Ensure model weight files (.pth) are in the correct location
- Check `MODEL_DIR` environment variable
- Verify file names match expected names

**CUDA out of memory:**
- Preview uses a 512px size limit for speed
- Final enhancement uses full resolution
- Reduce input image size if needed

**Preview not updating:**
- Select an example image first
- Check console for error messages
- Try reducing preview scale

## 🏗️ Model Training

To train the EnhancedMIRAM+SRGAN models:

```bash
# The SR.py script in uploads contains training code
# Extract and run:
python SR.py
# Choose option 1 to train
```

Training produces:
- `miram_best.pth` → rename to `SR_best.pth`
- `srgan_best.pth` → use as-is

## 📜 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
pillow>=10.0.0
numpy>=1.24.0
timm>=0.9.0
matplotlib>=3.7.0
tkinter (usually included with Python)
```
