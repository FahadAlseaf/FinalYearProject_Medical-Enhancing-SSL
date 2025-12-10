# Module 3: MIRAM Vision Transformer

This module implements the complete **MIRAM (Masked Image Reconstruction Across Multiple Scales)** framework with a **Vision Transformer (ViT)** backbone. It includes self-supervised pretraining, image restoration, tumor classification with attention visualization, and ONNX export for deployment.

## 📋 Overview

MIRAM is a multi-scale self-supervised learning approach that:
1. Masks 75% of image patches during training
2. Reconstructs images at two scales (fine: 224×224, coarse: 112×112)
3. Learns robust anatomical representations without labels
4. Enables downstream tasks like tumor classification

### Key Components

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters and paths |
| `dataset.py` | Data loading with multi-scale targets |
| `models.py` | Full MIRAM architecture with ViT |
| `losses.py` | Dual-scale patch loss + metrics |
| `train.py` | Pretraining pipeline with plotting |
| `enhance.py` | Image restoration/denoising |
| `evaluate.py` | Reconstruction visualization |
| `classify.py` | Tumor classification with attention maps |
| `export_deploy.py` | ONNX export and speed benchmarking |

## 🏗️ Architecture

### MIRAM Model

```
Input Image (224×224)
        ↓
   Patch Embedding (16×16 patches → 196 tokens)
        ↓
   Position Embedding + CLS Token
        ↓
   Random Masking (75%)
        ↓
   ViT Encoder (12 layers, 6 heads, dim=384)
        ↓
   Decoder Embedding (384 → 256)
        ↓
   Mask Token Insertion
        ↓
   Decoder (4 layers, 8 heads)
        ↓
   ┌────────────────┬────────────────┐
   │  Fine Head     │  Coarse Head   │
   │  (256 pixels)  │  (64 pixels)   │
   └────────────────┴────────────────┘
        ↓                  ↓
   224×224 Output    112×112 Output
```

### Dual-Scale Loss

```python
Total Loss = λ_fine × L_fine + λ_coarse × L_coarse

where:
- L_fine: MSE loss on 224×224 reconstruction
- L_coarse: MSE loss on 112×112 reconstruction
- λ_fine = 1.0
- λ_coarse = 0.5
```

## 🛠️ Installation

```bash
cd module_3_miram_vit

# Install dependencies
pip install torch torchvision tifffile pillow numpy tqdm tensorboard matplotlib seaborn scikit-learn opencv-python

# For ONNX export
pip install onnx onnxruntime-gpu
```

## 🚀 Usage

### 1. Self-Supervised Pretraining

```bash
# Train MIRAM on unlabeled images
python train.py
```

### 2. Image Restoration/Denoising

```bash
# Enhance a single image
python enhance.py
# Enter image path when prompted
```

### 3. Evaluate Reconstruction

```bash
# Visualize masked → reconstructed → original
python evaluate.py
```

### 4. Tumor Classification

```bash
# Train classifier and generate attention heatmaps
python classify.py
```

### 5. Export for Deployment

```bash
# Export to ONNX and benchmark speed
python export_deploy.py
```

## ☁️ Google Colab Setup

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Create directory structure
import os
base = '/content/drive/My Drive/mri_project'
os.makedirs(f'{base}/Brain_MRI/HR', exist_ok=True)

# Upload images to HR folder

# Run training
%cd {base}
!python train.py
```

## ⚙️ Configuration

Edit `config.py`:

### Paths
```python
DRIVE_PATH = '/content/drive/My Drive/mri_project'
DATASET_PATH = os.path.join(DRIVE_PATH, 'Brain_MRI')
```

### Model Hyperparameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | 224 | Input image size |
| `PATCH_SIZE` | 16 | Patch size (16×16) |
| `EMBED_DIM` | 384 | ViT embedding dimension |
| `DEPTH` | 12 | Number of transformer layers |
| `NUM_HEADS` | 6 | Attention heads |
| `MASK_RATIO` | 0.75 | Masking percentage |

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 16 | Training batch size |
| `N_EPOCHS` | 200 | Total epochs |
| `LEARNING_RATE` | 1.5e-4 | Learning rate |
| `WARMUP_EPOCHS` | 20 | Warmup epochs |

## 📁 Dataset Structure

```
Brain_MRI/
├── HR/                 # High-resolution images
│   ├── tumor_001.tif
│   ├── normal_001.tif
│   └── ...
└── MASK/               # Segmentation masks (for classification)
    ├── tumor_001_mask.tif
    └── ...
```

## 📊 Expected Outputs

### Training
```
mri_project/
├── best_miram_mae.pth          # Best pretraining weights
├── miram_checkpoint.pth         # Training checkpoint
├── results_miram/
│   ├── loss_curve.png           # Training/validation loss plot
│   └── miram_eval_sample.png    # Reconstruction visualization
└── logs_miram/                  # TensorBoard logs
```

### Classification
```
results_miram/
├── tumor_viz_0.png    # Attention heatmap on tumor case
├── tumor_viz_1.png
└── tumor_viz_2.png
```

### ONNX Export
```
models/
└── miram_model_optimized.onnx   # Optimized ONNX model
```

## 🔬 Technical Details

### Masked Autoencoding

1. **Patchify**: 224×224 → 196 patches of 16×16
2. **Mask**: Keep 25% (49 patches), mask 75% (147 patches)
3. **Encode**: Only visible patches → ViT encoder
4. **Decode**: Insert mask tokens → Decoder → Full reconstruction

### Multi-Scale Reconstruction

| Scale | Target Size | Patch Size | Purpose |
|-------|-------------|------------|---------|
| Fine | 224×224 | 16×16 | Detailed textures |
| Coarse | 112×112 | 8×8 | Global structure |

### Attention Visualization

The `classify.py` generates heatmaps showing:
- Where the model focuses when detecting tumors
- CLS token attention to all image patches
- Overlay on original MRI scan

## 📈 Training Progress

```
🚀 STARTING MIRAM (ViT-MAE) TRAINING | Device: cuda
📁 MIRAM Pre-training Dataset: 2847 images found.
Epoch 1/200: 100%|██████████| 178/178 [01:23<00:00] L:0.0456
   💾 Best Model Saved (Val: 0.0423)
Epoch 5/200: 100%|██████████| 178/178 [01:21<00:00] L:0.0312
📊 Training plots saved to results_miram
...
✅ Pre-training Complete.
```

## ⚡ Speed Benchmark

```
⏱️ RUNNING SPEED BENCHMARK (PyTorch vs ONNX)
--------------------------------------------------
📊 BENCHMARK RESULTS
==================================================
PyTorch Inference: 23.45 ms/image (42.6 FPS)
ONNX Optimized:    18.21 ms/image (54.9 FPS)

✅ REQUIREMENT MET: Inference is < 0.5 seconds.
🚀 Speedup Factor: 1.29x
```

## 🎯 Tumor Classification

### Training
```python
# Automatic labeling based on masks:
# - If mask has non-zero pixels → Tumor (label=1)
# - If mask is empty → Normal (label=0)
```

### Results
```
📊 FINAL DIAGNOSTIC REPORT
Accuracy: 92.34% | Precision: 0.91 | Recall: 0.94
```

## ⚠️ Troubleshooting

### "No images found"
```python
# Verify path in config.py
import os
print(os.path.exists(DATASET_PATH))
print(os.listdir(os.path.join(DATASET_PATH, 'HR')))
```

### CUDA Out of Memory
```python
# Reduce batch size or image size
BATCH_SIZE = 8
IMG_SIZE = 192  # Slightly smaller
```

### ONNX Export Fails
```bash
# Install correct ONNX runtime
pip install onnx onnxruntime-gpu  # For GPU
pip install onnx onnxruntime      # For CPU only
```

### Attention Maps Not Generated
- Ensure masks exist in MASK folder
- Check that some masks have non-zero pixels (tumors)

## 📚 References

- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- [MIRAM: Masked Image Reconstruction Across Multiple Scales](https://arxiv.org/abs/2503.07157)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [ViT: An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
