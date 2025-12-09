# Module 1: General SSL Medical Image Enhancement (MAE + ViT)

This module implements the **Self-Supervised Learning (SSL)** pipeline for general medical imaging (X-ray, CT). It focuses on learning robust anatomical features from unlabeled data using **Masked Autoencoders (MAE)** before fine-tuning for Super-Resolution.

## 🌟 Key Features

- **Self-Supervised Pre-training:** Uses a Vision Transformer (ViT) backbone masked at **75%** to force the model to learn context and structure without labels.
- **Mixed Precision Training:** Implements `torch.amp` (Automatic Mixed Precision) to accelerate training and reduce memory usage on GPUs.
- **On-the-Fly Generation:** Low-Resolution (LR) images are generated dynamically during training via bicubic downsampling, saving disk space and allowing flexible scaling.
- **Dual-Path Evaluation:** Includes a baseline training script (`train_baseline.py`) that trains from scratch, allowing for direct comparison to prove SSL effectiveness.
- **Advanced GAN Architecture:** The Super-Resolution phase uses a Generator with **MIRAM Blocks** (Channel + Spatial Attention) and a Discriminator trained with Adversarial + Perceptual (VGG) losses.

## 📂 File Structure

```
module_1_general_ssl/
├── config.py              # Centralized configuration
├── pretrain.py            # Phase 1: MAE encoder pre-training
├── train.py               # Phase 2: Fine-tuning with SSL weights
├── train_baseline.py      # Phase 3: Baseline comparison
├── visualize_results.py   # Inference and visualization
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Configure Paths

Edit `config.py` or set environment variables:

```python
# Option 1: Edit config.py directly
DATA_DIR = "/path/to/your/medical/images"

# Option 2: Use environment variables
export DATA_DIR="/path/to/your/data"
export OUTPUT_DIR="/path/to/outputs"
```

### 2. Run the Pipeline

```bash
# Phase 1: Pre-train encoder using MAE (self-supervised)
python pretrain.py

# Phase 2: Fine-tune for super-resolution with SSL weights
python train.py

# Phase 3: Train baseline for comparison (no pre-training)
python train_baseline.py

# Visualize results
python visualize_results.py
# Or with a specific test image:
python visualize_results.py --image /path/to/test.png
```

## ⚙️ Configuration

All settings are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_DIR` | `./data/medical_images` | Path to training images |
| `OUTPUT_DIR` | `./outputs` | Where to save models |
| `IMG_SIZE` | 224 | Input image size |
| `SCALE_FACTOR` | 4 | Super-resolution scale |
| `PRETRAIN_EPOCHS` | 100 | MAE pre-training epochs |
| `FINETUNE_EPOCHS` | 50 | Fine-tuning epochs |
| `MASK_RATIO` | 0.75 | MAE masking ratio |

## 📊 Expected Results

After training, compare PSNR values:

| Model | Expected PSNR |
|-------|---------------|
| Baseline (no SSL) | ~31 dB |
| **Ours (with SSL)** | **~35 dB** |

The ~4 dB improvement demonstrates the effectiveness of self-supervised pre-training.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Phase 1: Pre-training                 │
│  ┌─────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │ Medical │───▶│ 75% Mask │───▶│ ViT Encoder +    │   │
│  │ Images  │    │          │    │ MAE Decoder      │   │
│  └─────────┘    └──────────┘    └──────────────────┘   │
│                                  Reconstruct masked     │
│                                  patches                │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Phase 2: Fine-tuning                    │
│  ┌────┐   ┌──────────────────┐   ┌────────────┐  ┌──┐  │
│  │ LR │──▶│ Pre-trained ViT  │──▶│ MIRAM Block│──▶│SR│  │
│  └────┘   │ Encoder          │   │ + Upsample │  └──┘  │
│           └──────────────────┘   └────────────┘        │
│                    ▲                                    │
│                    │ Load weights                       │
│               [pretrained_encoder.pth]                  │
└─────────────────────────────────────────────────────────┘
```

## 📁 Output Files

After training, these files will be saved:

| File | Description |
|------|-------------|
| `outputs/pretrained_encoder.pth` | MAE pre-trained ViT encoder |
| `outputs/best_generator.pth` | Best SSL fine-tuned model |
| `outputs/best_generator_baseline.pth` | Best baseline model |
| `result_comparison.png` | Visual comparison |

## 🔧 Troubleshooting

**"No images found" error:**
- Ensure `DATA_DIR` points to a folder containing `.png`, `.jpg`, or `.tif` images
- Images can be in subdirectories (recursive search)

**CUDA out of memory:**
- Reduce `BATCH_SIZE` in `config.py`
- Use smaller `IMG_SIZE` (must be divisible by 16)

**Pre-trained weights not found:**
- Run `pretrain.py` before `train.py`
- Check that `PRETRAINED_ENCODER_PATH` in config matches output path
