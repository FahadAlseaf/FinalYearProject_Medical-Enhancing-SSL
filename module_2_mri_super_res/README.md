# Module 2: Brain MRI Super-Resolution (MIRAM)

This module implements 4x super-resolution for brain MRI images using the **MIRAM (Masked Image Reconstruction Across Multiple Scales)** architecture with GAN-based adversarial training.

## 🌟 Key Features

- **16-bit Medical Precision:** Full support for 16-bit TIFF files to preserve diagnostic quality
- **MIRAM Attention Blocks:** Combined Channel + Spatial attention for anatomically-aware enhancement
- **Masked Loss Computation:** Focus training on brain regions, ignore background
- **Advanced Loss Functions:** Charbonnier + Edge + Perceptual + Adversarial losses
- **Warmup Strategy:** Initial pixel-only training for stability before GAN training

## 📂 File Structure

```
module_2_mri_super_res/
├── config.py       # Centralized configuration
├── models.py       # MIRAM Generator & Discriminator
├── losses.py       # Custom loss functions
├── dataset.py      # Data loading utilities
├── train.py        # Training loop
├── evaluate.py     # Evaluation & metrics
├── enhance.py      # Single image inference
└── README.md       # This file
```

## 🚀 Quick Start

### 1. Prepare Your Dataset

Organize your MRI data with this structure:
```
data/mri_project/Brain_MRI/
├── HR/          # High-resolution ground truth
├── LR/          # Low-resolution inputs (auto-generated if missing)
└── MASK/        # Brain masks (optional but recommended)
```

Or let the auto-preparation handle it:
```bash
# Just place all images in Brain_MRI/, the script will organize them
```

### 2. Configure Paths

Edit `config.py` or use environment variables:
```bash
export MRI_DATA_PATH="/path/to/your/data"
export MRI_OUTPUT_PATH="/path/to/outputs"
```

### 3. Train

```bash
python train.py
```

### 4. Evaluate

```bash
python evaluate.py
```

### 5. Enhance Single Images

```bash
python enhance.py --input my_scan.tif --output enhanced_scan.tif
```

## ⚙️ Configuration

All settings are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SCALE_FACTOR` | 4 | Super-resolution scale |
| `CROP_SIZE` | 128 | Training patch size |
| `N_EPOCHS` | 200 | Training epochs |
| `BATCH_SIZE` | 8 | Batch size |
| `WARMUP_PERCENTAGE` | 0.15 | Pixel-only warmup fraction |
| `USE_16BIT` | True | 16-bit TIFF support |

### Loss Weights

| Weight | Default | Purpose |
|--------|---------|---------|
| `LAMBDA_PIXEL` | 1.0 | Charbonnier reconstruction |
| `LAMBDA_EDGE` | 0.1 | Edge preservation |
| `LAMBDA_VGG` | 0.05 | Perceptual quality |
| `LAMBDA_ADV` | 0.005 | GAN realism |

## 🏗️ Architecture

```
Input (LR)
    │
    ▼
┌─────────────────────────────────────────────┐
│  Head: Conv9x9 + PReLU                      │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Body: [ResBlock + MIRAM] × 8               │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │ MIRAM Block:                        │   │
│  │   Conv → PReLU → Conv               │   │
│  │      → ChannelAttn → SpatialAttn    │   │
│  │      → Residual Add                 │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Upsampler: PixelShuffle 2x + 2x = 4x       │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Tail: Conv9x9 → Output                     │
└─────────────────────────────────────────────┘
    │
    ▼
Output (SR, 4x larger)
```

## 📊 Expected Results

| Metric | Expected Value |
|--------|----------------|
| PSNR | ~35 dB |
| SSIM | ~0.94 |
| Training Time | ~4-6 hours (RTX 3080) |

## 📁 Output Files

| File | Description |
|------|-------------|
| `best_generator.pth` | Best model weights |
| `checkpoint.pth` | Latest checkpoint (for resuming) |
| `logs/` | TensorBoard training logs |
| `results/` | Enhanced images |

## 🔧 Troubleshooting

**"No valid image triplets found":**
- Ensure HR, LR, and MASK folders have matching filenames
- Check that mask files contain "mask" in filename or match HR names

**CUDA out of memory:**
- Reduce `BATCH_SIZE` in config.py
- Reduce `CROP_SIZE` (must be divisible by 4)

**Poor results:**
- Increase `N_EPOCHS`
- Check that masks properly cover brain regions
- Try adjusting loss weights

## 📜 Citation

```bibtex
@thesis{alhabib2024miram,
  title={MIRAM: Masked Image Reconstruction Across Multiple Scales for MRI Super-Resolution},
  author={Alhabib, Ahmed and Alseaf, Fahad and Albaradi, Meshal},
  year={2024},
  school={Qassim University}
}
```
