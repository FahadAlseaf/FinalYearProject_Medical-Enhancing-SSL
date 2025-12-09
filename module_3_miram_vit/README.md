# Module 3: MIRAM ViT-MAE (Masked Image Reconstruction Across Multiple Scales)

This module implements a **Vision Transformer-based Masked Autoencoder** for self-supervised learning on medical images, with applications in **image restoration** and **tumor classification**.

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Self-Supervised Pre-training** | Learn anatomical features without labels using masked autoencoding |
| **Dual-Scale Reconstruction** | Reconstruct at both fine (224×224) and coarse (112×112) scales |
| **Attention Visualization** | Generate heatmaps showing where the model focuses |
| **Tumor Classification** | Fine-tune for binary classification with attention localization |
| **ONNX Export** | Deploy optimized model with speed benchmarking |

## 📂 File Structure

```
module_3_miram_vit/
├── config.py           # Centralized configuration
├── models.py           # MIRAM architecture (ViT encoder + dual decoder)
├── dataset.py          # Data loading utilities
├── losses.py           # Dual-scale patch loss + metrics
├── train.py            # Self-supervised pre-training
├── evaluate.py         # Reconstruction visualization
├── enhance.py          # Single image inference
├── classify.py         # Tumor classification with heatmaps
├── export_deploy.py    # ONNX export & benchmarking
└── README.md           # This file
```

## 🏗️ Architecture

### MIRAM Model

```
Input Image (1, 224, 224)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                        ENCODER                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Patch Embedding: 16×16 patches → 196 tokens            │  │
│  │  + Positional Embedding + [CLS] Token                   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                          │                                     │
│                          ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Random Masking (75% hidden)                            │  │
│  │  → Keep only 49 visible patches                         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                          │                                     │
│                          ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  12× Transformer Blocks                                 │  │
│  │  (384 dim, 6 heads, MLP ratio 4.0)                     │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                        DECODER                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Project to 256 dim + Add mask tokens                   │  │
│  │  Unshuffle to restore original order                    │  │
│  └─────────────────────────────────────────────────────────┘  │
│                          │                                     │
│                          ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  4× Decoder Transformer Blocks                          │  │
│  │  (256 dim, 8 heads)                                     │  │
│  └─────────────────────────────────────────────────────────┘  │
│                          │                                     │
│              ┌───────────┴───────────┐                        │
│              ▼                       ▼                        │
│  ┌─────────────────┐     ┌─────────────────┐                  │
│  │  Fine Head      │     │  Coarse Head    │                  │
│  │  (16×16 patches)│     │  (8×8 patches)  │                  │
│  └─────────────────┘     └─────────────────┘                  │
└───────────────────────────────────────────────────────────────┘
        │                           │
        ▼                           ▼
   Fine Recon (224×224)      Coarse Recon (112×112)
```

## 🚀 Quick Start

### 1. Configure Paths

```python
# Option A: Edit config.py directly
DRIVE_PATH = "/path/to/your/data"

# Option B: Use environment variables
export MIRAM_DATA_PATH="/path/to/your/data"
export MIRAM_OUTPUT_PATH="/path/to/outputs"
```

### 2. Prepare Dataset

```
data/mri_project/Brain_MRI/
├── HR/                    # High-resolution images
│   ├── image001.tif
│   ├── image002.tif
│   └── ...
└── MASK/                  # Tumor masks (for classification)
    ├── image001.tif       # Non-zero pixels = tumor
    ├── image002.tif
    └── ...
```

### 3. Train the Model

```bash
# Phase 1: Self-supervised pre-training
python train.py

# Phase 2: Evaluate reconstruction quality
python evaluate.py

# Phase 3: Train tumor classifier (optional)
python classify.py
```

### 4. Inference

```bash
# Enhance a single image
python enhance.py --input scan.tif --output restored.tif

# Or interactive mode
python enhance.py
```

### 5. Deploy

```bash
# Export to ONNX with benchmarking
python export_deploy.py
```

## ⚙️ Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | 224 | Input image size (ViT standard) |
| `PATCH_SIZE` | 16 | Patch size (224/16 = 14×14 grid) |
| `EMBED_DIM` | 384 | Encoder embedding dimension |
| `DEPTH` | 12 | Number of encoder transformer blocks |
| `MASK_RATIO` | 0.75 | Fraction of patches to mask |
| `N_EPOCHS` | 200 | Pre-training epochs |

### Training Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 16 | Training batch size |
| `LEARNING_RATE` | 1.5e-4 | AdamW learning rate |
| `WEIGHT_DECAY` | 0.05 | L2 regularization |
| `LAMBDA_FINE` | 1.0 | Fine-scale loss weight |
| `LAMBDA_COARSE` | 0.5 | Coarse-scale loss weight |

## 📊 Output Files

### Training Outputs

| File | Description |
|------|-------------|
| `best_miram_mae.pth` | Best pre-trained model weights |
| `loss_curve.png` | Training/validation loss plot |
| `miram_eval_sample.png` | Reconstruction visualization |

### Classification Outputs

| File | Description |
|------|-------------|
| `best_tumor_classifier.pth` | Trained classifier weights |
| `tumor_viz_*.png` | Attention heatmap visualizations |

### Deployment Outputs

| File | Description |
|------|-------------|
| `miram_model_optimized.onnx` | Exported ONNX model |

## 🔬 How It Works

### 1. Self-Supervised Pre-training

The model learns to reconstruct randomly masked patches:

1. **Masking**: 75% of patches are randomly hidden
2. **Encoding**: Visible patches processed by ViT encoder
3. **Decoding**: Predict pixel values for ALL patches
4. **Loss**: MSE only on masked patches (reconstruction target)

This forces the model to learn meaningful anatomical representations.

### 2. Tumor Classification

The pre-trained encoder is fine-tuned for classification:

1. **Feature Extraction**: Use [CLS] token from encoder
2. **Classification Head**: Linear layers → sigmoid
3. **Attention Maps**: Visualize where model focuses

### 3. Attention Visualization

The attention weights from the last transformer block show which image regions influence the classification decision, providing interpretability for medical diagnosis.

## 📈 Expected Results

| Metric | Pre-training | Classification |
|--------|--------------|----------------|
| PSNR | ~30-35 dB | - |
| SSIM | ~0.90-0.95 | - |
| Accuracy | - | ~90-95% |
| Inference | - | <100ms (ONNX) |

## 🔧 Troubleshooting

**"CUDA out of memory"**
- Reduce `BATCH_SIZE` in config.py
- Use gradient accumulation

**"Dataset empty"**
- Check `DATASET_PATH` points to correct location
- Ensure HR/ folder contains images

**"Model not found"**
- Run `train.py` first to generate weights

**"ONNX export failed"**
- Install: `pip install onnx onnxruntime`
- Check PyTorch version compatibility

## 📚 References

- He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
- Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
