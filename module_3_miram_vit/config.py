"""
config.py - MIRAM ViT-MAE Configuration

Centralized configuration for Module 3: MIRAM (Masked Image Reconstruction
Across Multiple Scales) with Vision Transformer architecture.

This module implements:
    - Self-supervised pre-training with masked autoencoding
    - Dual-scale reconstruction (fine + coarse)
    - Tumor classification with attention visualization
    - ONNX export for deployment

Usage:
    from config import *
    
    # Or override with environment variables:
    export MIRAM_DATA_PATH="/path/to/Brain_MRI"
    export MIRAM_OUTPUT_PATH="/path/to/outputs"
"""

import os
import torch
from pathlib import Path

# ===========================================
# PATH CONFIGURATION
# ===========================================

# Auto-detect module directory
_MODULE_DIR = Path(__file__).parent
_ROOT_DIR = _MODULE_DIR.parent

# Base paths - override with environment variables
DRIVE_PATH = os.environ.get(
    "MIRAM_DATA_PATH",
    str(_ROOT_DIR / "data" / "mri_project")
)

DATASET_PATH = os.path.join(DRIVE_PATH, "Brain_MRI")

# Model paths
MODEL_DIR = os.path.join(DRIVE_PATH, "models")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "miram_checkpoint.pth")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_miram_mae.pth")
BEST_CLASSIFIER_PATH = os.path.join(MODEL_DIR, "best_tumor_classifier.pth")

# Output paths
OUTPUT_DIR = os.environ.get(
    "MIRAM_OUTPUT_PATH",
    os.path.join(DRIVE_PATH, "results_miram")
)
LOG_DIR = os.path.join(DRIVE_PATH, "logs_miram")

# Create directories
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ===========================================
# DEVICE CONFIGURATION
# ===========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================================
# IMAGE SETTINGS (ViT Specific)
# ===========================================
IMG_SIZE = 224              # Standard ViT Resolution
PATCH_SIZE = 16             # 16x16 patches -> 14x14 = 196 patches
IN_CHANNELS = 1             # Grayscale MRI
USE_16BIT = True            # 16-bit TIFF support

# ===========================================
# MODEL HYPERPARAMETERS (ViT-Small)
# ===========================================
EMBED_DIM = 384             # Embedding dimension
DEPTH = 12                  # Number of transformer blocks
NUM_HEADS = 6               # Attention heads
MLP_RATIO = 4.0             # MLP hidden dimension ratio

# Decoder (Lightweight)
DECODER_EMBED_DIM = 256
DECODER_DEPTH = 4
DECODER_NUM_HEADS = 8

# MIRAM Specifics
MASK_RATIO = 0.75           # 75% patches hidden (MAE default)
SCALE_COARSE = 0.5          # Coarse target: 112x112
SCALE_FINE = 1.0            # Fine target: 224x224

# ===========================================
# TRAINING SETTINGS
# ===========================================
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True

LEARNING_RATE = 1.5e-4
WEIGHT_DECAY = 0.05
N_EPOCHS = 200
WARMUP_EPOCHS = 20

# Loss Weights
LAMBDA_FINE = 1.0           # Fine-scale reconstruction weight
LAMBDA_COARSE = 0.5         # Coarse-scale reconstruction weight

# ===========================================
# DATA SPLIT
# ===========================================
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# ===========================================
# CLASSIFICATION SETTINGS
# ===========================================
CLASSIFIER_EPOCHS = 10
CLASSIFIER_LR = 1e-4

# ===========================================
# HELPER FUNCTION
# ===========================================

def print_config():
    """Print current configuration for debugging."""
    print("=" * 50)
    print("Module 3 Configuration (MIRAM ViT-MAE)")
    print("=" * 50)
    print(f"DATASET_PATH:  {DATASET_PATH}")
    print(f"OUTPUT_DIR:    {OUTPUT_DIR}")
    print(f"MODEL_DIR:     {MODEL_DIR}")
    print(f"DEVICE:        {DEVICE}")
    print(f"IMG_SIZE:      {IMG_SIZE}")
    print(f"PATCH_SIZE:    {PATCH_SIZE}")
    print(f"EMBED_DIM:     {EMBED_DIM}")
    print(f"DEPTH:         {DEPTH}")
    print(f"MASK_RATIO:    {MASK_RATIO}")
    print(f"N_EPOCHS:      {N_EPOCHS}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
