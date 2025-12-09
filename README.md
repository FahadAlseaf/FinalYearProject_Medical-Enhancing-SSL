# Improving Medical Imaging With SSL Image Translation

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)
![Status](https://img.shields.io/badge/Status-Completed-success)

This repository contains the official implementation for the Final Year Project: **Improving Medical Imaging With SSL Image Translation**.

The project introduces the **MIRAM (Masked Image Reconstruction Across Multiple Scales)** framework, utilizing Self-Supervised Learning (SSL) to enhance medical imaging quality without relying on massive labeled datasets.

## 📄 Abstract

This study investigates a self-supervised image-to-image translation approach for medical imaging. The method utilizes a Generative Adversarial Network (GAN) trained with reconstruction, adversarial, and cycle-consistency losses. We introduce the **MIRAM** architecture combined with Vision Transformers (ViT) and Masked Autoencoders (MAE) to learn robust anatomical features from unlabeled data.

Experimental results on X-ray, CT, and MRI datasets show significant gains, achieving a **4.2 dB increase in PSNR** and a **12% improvement in SSIM** compared to baseline methods.

## 👥 Authors (Qassim University)

- **Ahmed Alhabib**
- **Fahad Alseaf**
- **Meshal Albaradi**

**Supervisor:** Dr. Mohmmad Ali A. Hammoudeh

---

## 📊 Results

| Method | PSNR (dB) | SSIM |
|--------|-----------|------|
| Bicubic Interpolation | 28.4 | 0.82 |
| Baseline (No SSL) | 31.2 | 0.88 |
| **Ours (MIRAM + SSL)** | **35.4** | **0.94** |

---

## 📂 Repository Structure

```
medical-ssl-sr/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── app/                         # 🖥️ GUI Application
│   ├── README.md                # App documentation
│   ├── app.py                   # Tkinter GUI interface
│   └── SR_models.py             # EnhancedMIRAM + SRGAN models
│
├── module_1_general_ssl/        # X-Ray/CT Enhancement
│   ├── README.md
│   ├── config.py                # Centralized configuration
│   ├── pretrain.py              # Phase 1: MAE pre-training
│   ├── train.py                 # Phase 2: Fine-tuning with SSL
│   ├── train_baseline.py        # Phase 3: Baseline comparison
│   └── visualize_results.py     # Inference & visualization
│
└── module_2_mri_super_res/      # Brain MRI Super-Resolution
    ├── README.md
    ├── config.py                # Centralized configuration
    ├── models.py                # MIRAM Generator & Discriminator
    ├── losses.py                # Custom loss functions
    ├── dataset.py               # Data loading utilities
    ├── train.py                 # Training loop
    ├── evaluate.py              # Evaluation & metrics
    └── enhance.py               # Single image inference
```

### Module Descriptions

#### 🖥️ `app/` - GUI Application
**Focus:** Unified Interface for All Enhancement Models

- **Tkinter-based GUI** for easy image enhancement
- Supports **Model Selection**: Switch between M1 (MIRAM+SRGAN) and M2 (MRI Enhancer)
- **Live Preview** with adjustable parameters (Sharpness, Contrast, Brightness)
- **Batch Processing** for multiple images
- **ViT Comparison** visualization (SSL vs Baseline)

#### 1. `module_1_general_ssl/`
**Focus:** Self-Supervised Pre-training & General Translation

- Implements **Masked Autoencoders (MAE)** with Vision Transformers (ViT)
- Used for the initial "Pre-training Phase" to learn anatomical features from unlabeled X-Ray/CT data
- Includes a baseline comparison script to demonstrate SSL effectiveness

#### 2. `module_2_mri_super_res/`
**Focus:** High-Precision Brain MRI Super-Resolution (4x)

- Implements the **MIRAM Generator** with Channel & Spatial Attention
- Optimized for **16-bit Medical TIFF** images
- Uses advanced loss functions: **Edge Loss + Charbonnier Loss + Perceptual Loss**

---

## 🖥️ GUI Application

The repository includes a **Tkinter-based GUI** that provides a unified interface for all enhancement models.

### Features

| Feature | Description |
|---------|-------------|
| **Model Selection** | Switch between M1 (MIRAM+SRGAN) and M2 (MRI Enhancer) |
| **Live Preview** | Real-time enhancement preview with adjustable parameters |
| **Batch Processing** | Enhance multiple images at once |
| **Parameter Sliders** | Adjust Sharpness, Contrast, Brightness |
| **ViT Comparison** | Visualize SSL vs Baseline results |

### Quick Start (GUI)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app/app.py
```

### Screenshot

```
┌──────────────────────────────────────────────────────────┐
│          AI Medical Image Enhancement Tool               │
├──────────────────────────────────────────────────────────┤
│  [M1: MIRAM+SRGAN]    [M2: MRI Enhancer]                │
│  Model: EnhancedMIRAM + SRGAN (M1)                      │
├──────────────────────────────────────────────────────────┤
│  [Select Images]      [Select Output Folder]             │
│  Output: /path/to/output                                 │
├──────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────┐ │
│  │  image1.png                                         │ │
│  │  image2.tif                                         │ │
│  │  image3.jpg                                         │ │
│  └────────────────────────────────────────────────────┘ │
│  [Select Preview Example]                                │
├──────────────────────────────────────────────────────────┤
│  Sharpness    [════════●═══════════] 1.0                │
│  Contrast     [════════●═══════════] 1.0                │
│  Brightness   [════════●═══════════] 1.0                │
│  Preview Size [════════●═══════════] 1.0×               │
│              [Reset to Default]                          │
├──────────────────────────────────────────────────────────┤
│         [      Start Enhancement      ]                  │
│         [ Visualize ViT Results ]                        │
└──────────────────────────────────────────────────────────┘
```

See [`app/README.md`](app/README.md) for detailed usage instructions.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/medical-ssl-sr.git
cd medical-ssl-sr
pip install -r requirements.txt
```

### 2. Prepare Your Data

```bash
# Create data directory
mkdir -p data/medical_images

# Option A: Place your own X-Ray/CT images in data/medical_images/
# Option B: Download a public dataset (e.g., NIH ChestX-ray)
```

### 3. Configure Paths

Edit the configuration file for your chosen module:

```python
# module_1_general_ssl/config.py
DATA_DIR = "./data/medical_images"  # Update to your data path

# module_2_mri_super_res/config.py
DRIVE_PATH = "./data/mri_project"   # Update to your data path
```

Alternatively, use environment variables:
```bash
export DATA_DIR="/path/to/your/data"
export OUTPUT_DIR="/path/to/outputs"
```

### 4. Run Module 1 (X-Ray/CT Enhancement)

```bash
cd module_1_general_ssl

# Phase 1: Pre-train the encoder using MAE
python pretrain.py

# Phase 2: Fine-tune for super-resolution with SSL weights
python train.py

# Phase 3: Train baseline for comparison (no pre-training)
python train_baseline.py

# Visualize results
python visualize_results.py
```

### 5. Run Module 2 (MRI Super-Resolution)

```bash
cd module_2_mri_super_res

# Train the MIRAM model
python train.py

# Evaluate on test set
python evaluate.py

# Enhance a single image
python enhance.py
```

### 6. Run the GUI Application (Optional)

```bash
# From repository root
python app/app.py

# Required model files (place in app/ directory):
# - SR_best.pth (EnhancedMIRAM weights)
# - srgan_best.pth (EnhancedGenerator weights)
# - mri_enhancer_best.pth (MRI Generator weights)
```

---

## ⚙️ Configuration

### Module 1 Settings (`module_1_general_ssl/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_DIR` | `./data` | Path to training images |
| `EPOCHS` | 50 | Training epochs |
| `BATCH_SIZE` | 16 | Batch size (adjust for VRAM) |
| `LEARNING_RATE` | 1e-4 | Learning rate |
| `IMG_SIZE` | 224 | Input image size |
| `MASK_RATIO` | 0.75 | MAE masking ratio |

### Module 2 Settings (`module_2_mri_super_res/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SCALE_FACTOR` | 4 | Super-resolution scale |
| `CROP_SIZE` | 128 | Training patch size |
| `N_EPOCHS` | 200 | Training epochs |
| `USE_16BIT` | True | 16-bit TIFF support |
| `LAMBDA_EDGE` | 0.1 | Edge loss weight |

---

## 📋 Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- See `requirements.txt` for full dependencies

---

## 📜 Citation

If you use this code in your research, please cite:

```bibtex
@thesis{alhabib2024medical,
  title={Improving Medical Imaging With SSL Image Translation},
  author={Alhabib, Ahmed and Alseaf, Fahad and Albaradi, Meshal},
  year={2024},
  school={Qassim University}
}
```

---

---

## 🙏 Acknowledgments

- Dr. Mohmmad Ali A. Hammoudeh for supervision and guidance
- Qassim University for supporting this research
- The authors of MAE and Vision Transformer for foundational work
