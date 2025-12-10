# Improving Medical Imaging with Self-Supervised Image Translation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)

A comprehensive deep learning framework for enhancing medical images using **Self-Supervised Learning (SSL)**, **Vision Transformers (ViT)**, and **Generative Adversarial Networks (GANs)**. This project addresses the critical challenge of improving diagnostic accuracy in medical imaging while reducing dependence on expensive labeled datasets.

## 🔬 Overview

Medical imaging plays a central role in modern healthcare, enabling accurate diagnosis, treatment planning, and patient monitoring. However, image quality is often compromised by:

- Low resolution
- Noise and artifacts
- Inconsistencies between imaging modalities
- Limited availability of labeled training data

This project introduces a **self-supervised image-to-image translation approach** that:

1. Reduces reliance on labeled datasets by 60-80%
2. Enhances low-quality scans to high-resolution outputs
3. Supports cross-modality translation (e.g., CT to MRI)
4. Achieves real-time inference for clinical deployment

### Key Results

| Metric | Improvement |
|--------|-------------|
| **PSNR** | +4.2 dB over baseline |
| **SSIM** | +12% improvement |
| **Inference Speed** | ~50 FPS (0.02s per image) |

## ✨ Key Features

- **Self-Supervised Learning**: Train on unlabeled medical images using Masked Autoencoders (MAE)
- **Multi-Scale Reconstruction (MIRAM)**: Dual decoder architecture for fine and coarse detail recovery
- **Vision Transformer Backbone**: State-of-the-art ViT-based encoder for global feature extraction
- **Super-Resolution GANs**: 4x upscaling with attention mechanisms
- **Clinical-Ready**: ONNX export support for deployment
- **Tumor Classification**: Downstream task with attention visualization

## 📁 Project Structure

```
├── app/                          # Desktop GUI application
│   └── app.py                    # Tkinter-based image enhancer
│
├── module_1_general_ssl/         # Self-Supervised Pretraining
│   ├── pretrain.py               # MAE pretraining with ViT
│   ├── train.py                  # SRGAN training (with SSL weights)
│   ├── train_baseline.py         # Baseline training (no SSL)
│   └── visualize_results.py      # Model comparison visualization
│
├── module_2_mri_super_res/       # MRI Super-Resolution
│   ├── config.py                 # Configuration settings
│   ├── dataset.py                # Data loading utilities
│   ├── models.py                 # Generator & Discriminator
│   ├── losses.py                 # Loss functions
│   ├── train.py                  # Training pipeline
│   ├── enhance.py                # Single image enhancement
│   └── evaluate.py               # Evaluation & visualization
│
├── module_3_miram_vit/           # MIRAM ViT Implementation
│   ├── config.py                 # Configuration settings
│   ├── dataset.py                # Data loading utilities
│   ├── models.py                 # Full MIRAM architecture
│   ├── losses.py                 # Dual-scale loss functions
│   ├── train.py                  # Pretraining pipeline
│   ├── enhance.py                # Image restoration
│   ├── evaluate.py               # Evaluation & visualization
│   ├── classify.py               # Tumor classification
│   └── export_deploy.py          # ONNX export & benchmarking
│
├── module_4_SR/                  # Enhanced SRGAN
│   └── SR.py                     # Complete SR pipeline with imbalance handling
│
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                     # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-enabled GPU (recommended: NVIDIA T4, RTX 3080, or A100)
- 16GB+ RAM

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/medical-imaging-ssl.git
cd medical-imaging-ssl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Google Colab Setup

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/yourusername/medical-imaging-ssl.git
%cd medical-imaging-ssl

# Install dependencies
!pip install -r requirements.txt
```

## 📦 Modules

| Module | Description | Key Features |
|--------|-------------|--------------|
| **[Module 1](module_1_general_ssl/README.md)** | Self-Supervised Pretraining | MAE, ViT encoder, baseline comparison |
| **[Module 2](module_2_mri_super_res/README.md)** | MRI Super-Resolution | 4x upscaling, mask-focused training |
| **[Module 3](module_3_miram_vit/README.md)** | MIRAM ViT Architecture | Dual-scale reconstruction, tumor classification |
| **[Module 4](module_4_SR/README.md)** | Enhanced SRGAN | Attention mechanisms, imbalance handling |
| **[App](app/README.md)** | Desktop GUI | Interactive image enhancement |

## 📊 Results

### Quantitative Metrics

| Model Configuration | PSNR (dB) | SSIM |
|---------------------|-----------|------|
| CycleGAN (baseline) | 26.4 | 0.85 |
| SimCLR + ViT | 27.0 | 0.87 |
| MAE + ViT | 28.3 | 0.89 |
| ViT + MIRAM | 28.5 | 0.90 |
| **MAE + ViT + MIRAM (Ours)** | **30.6** | **0.93** |

### Visual Results

The framework successfully:
- Restores sharp anatomical edges
- Reduces background noise
- Preserves critical diagnostic features
- Achieves real-time processing (50 FPS)

## 📚 Datasets

This project uses publicly available medical imaging datasets:

| Dataset | Modality | Source |
|---------|----------|--------|
| Chest| X-ray | [NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC) |
| Brain | MRI | [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) |
| Brain | MRI | [Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) |
| Chest | X-ray | [Stanford](https://stanfordmlgroup.github.io/competitions/chexpert/) |

## 📖 BibTeX

If you use this work in your research, please cite:

```bibtex
@thesis{Quni2025medical,
  title={Improving Medical Imaging with Self-Supervised Image Translation},
  author={Alhabib, Ahmed and Alseaf, Fahad and Albaradi, Meshal},
  school={Qassim University},
  department={College of Computer, Information Technology Department},
  year={2025},
  supervisor={Dr. Mohmmad Ali A. Hammoudeh}
}
```

## 👥 Contributors

| Name |
|------|
| Ahmed Alhabib
| Fahad Alseaf
| Meshal Albaradi

**Supervisor**: Dr. Mohmmad Ali A. Hammoudeh

## 🙏 Acknowledgments

- Qassim University, College of Computer, Information Technology Department
- The developers of PyTorch, timm, and the open-source community
- NIH and Kaggle for providing publicly available medical imaging datasets
---

<p align="center">
  <b>Kingdom of Saudi Arabia | Ministry of Education | Qassim University</b><br>
  <i>College of Computer | Information Technology Department</i><br>
  2024/2025
</p>
