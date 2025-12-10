# Module 1: Self-Supervised Pretraining (MAE + ViT)

This module implements **Masked Autoencoder (MAE)** pretraining with a **Vision Transformer (ViT)** backbone for learning robust anatomical representations from unlabeled medical images.

## 📋 Overview

The self-supervised pretraining approach enables the model to learn meaningful features without requiring expensive manual annotations. By masking 75% of image patches and training the network to reconstruct them, the model learns to understand anatomical structures and spatial relationships.

### Key Components

| File | Description |
|------|-------------|
| `pretrain.py` | MAE pretraining with ViT encoder (Phase 1) |
| `train.py` | SRGAN training using SSL pretrained weights (Phase 2) |
| `train_baseline.py` | Baseline training from scratch for comparison |
| `visualize_results.py` | Compare SSL vs baseline model outputs |

## 🏗️ Architecture

```
Input Image (224×224)
       ↓
   Patch Embed (16×16 patches)
       ↓
   Random Masking (75%)
       ↓
   ViT Encoder (12 layers, 6 heads)
       ↓
   Decoder (4 layers)
       ↓
   Reconstruction Loss (MSE on masked patches)
```

## 🛠️ Installation

```bash
# Navigate to module directory
cd module_1_general_ssl

# Install dependencies (from project root)
pip install torch torchvision timm tqdm pillow numpy
```

## 🚀 Usage

### Option 1: Local Training

```bash
# Step 1: Pretrain the encoder
python pretrain.py

# Step 2: Train SRGAN with pretrained weights
python train.py

# Step 3 (Optional): Train baseline for comparison
python train_baseline.py

# Step 4: Visualize results
python visualize_results.py
```

### Option 2: Google Colab

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Upload your dataset to Kaggle or Drive
# Update DATA_DIR in pretrain.py to match your path

# Run pretraining
!python pretrain.py

# Run training
!python train.py
```

### Kaggle Notebook

1. Create a new Kaggle notebook
2. Add your medical image dataset as input
3. Update `DATA_DIR` path in scripts:
   ```python
   DATA_DIR = "/kaggle/input/your-dataset-name/medical_images_dataset"
   ```
4. Run cells sequentially

## ⚙️ Configuration

### Pretraining Settings (`pretrain.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_DIR` | `/kaggle/input/...` | Path to image dataset |
| `EPOCHS` | 100 | Number of pretraining epochs |
| `BATCH_SIZE` | 64 | Batch size (adjust for GPU memory) |
| `LEARNING_RATE` | 1.5e-4 | Learning rate |
| `IMG_SIZE` | 224 | Input image size |
| `mask_ratio` | 0.75 | Percentage of patches to mask |

### Training Settings (`train.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPOCHS` | 50 | Number of training epochs |
| `BATCH_SIZE` | 16 | Batch size |
| `LEARNING_RATE` | 1e-4 | Learning rate |
| `scale` | 4 | Super-resolution scale factor |

## 📁 Dataset Structure

```
medical_images_dataset/
├── nih_xray/
│   ├── image001.png
│   ├── image002.png
│   └── ...
├── brain_mri/
│   ├── scan001.tif
│   └── ...
└── ct_scans/
    └── ...
```

The dataset loader recursively finds all images with extensions: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`

## 📊 Expected Outputs

### After Pretraining
- `pretrained_encoder.pth` - Saved encoder weights

### After Training
- `best_generator.pth` - Best SSL-pretrained generator
- `best_generator_baseline.pth` - Best baseline generator (if running baseline)

### Visualization
- `result_comparison_full.png` - Side-by-side comparison image

## 🔬 Technical Details

### MAE (Masked Autoencoder)

The MAE approach:
1. **Patchify**: Divide 224×224 image into 196 patches (14×14 grid of 16×16 patches)
2. **Mask**: Randomly mask 75% of patches
3. **Encode**: Process only visible (25%) patches through ViT encoder
4. **Decode**: Reconstruct all patches including masked ones
5. **Loss**: MSE loss computed only on masked patches

### Generator Architecture

```python
GeneratorViT_SR:
├── ViT Encoder (pretrained)
├── Linear Projection (768 → 256)
├── MIRAM Block (Attention-enhanced residual)
└── Upsampling (4× via PixelShuffle)
    ├── UpsampleBlock (2×)
    ├── UpsampleBlock (2×)
    ├── UpsampleBlock (2×)
    └── UpsampleBlock (2×)
```

## 📈 Training Progress

The training scripts log:
- Loss per epoch
- PSNR (Peak Signal-to-Noise Ratio)
- Model checkpoints on improvement

Example output:
```
Epoch 1/50 Loss: 0.0234
Epoch 2/50 Loss: 0.0198
...
Epoch 50 Val PSNR: 28.34 dB
✅ SUCCESS: Loaded SSL Pre-trained Weights!
```

## 🆚 SSL vs Baseline Comparison

| Metric | Baseline (From Scratch) | With SSL Pretraining |
|--------|-------------------------|----------------------|
| PSNR | ~24 dB | ~28 dB |
| Training Epochs | 50 | 50 |
| Convergence | Slower | Faster |

## ⚠️ Troubleshooting

### Out of Memory
```python
# Reduce batch size
BATCH_SIZE = 32  # or lower
```

### Path Not Found
```python
# Verify your data path exists
import os
print(os.path.exists(DATA_DIR))
```

### Slow Training
- Enable mixed precision (already implemented via `torch.cuda.amp`)
- Use GPU with CUDA support
- Reduce `num_workers` if I/O bottleneck

## 📚 References

- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- [ESRGAN: Enhanced Super-Resolution GANs](https://arxiv.org/abs/1809.00219)
