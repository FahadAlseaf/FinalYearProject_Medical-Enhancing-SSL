"""
Configuration for Module 1: General SSL Medical Image Enhancement

This module provides centralized configuration for all training scripts.
Paths can be overridden using environment variables.

Usage:
    # Option 1: Edit this file directly
    DATA_DIR = "./your/data/path"
    
    # Option 2: Use environment variables
    export DATA_DIR="/path/to/data"
    export OUTPUT_DIR="/path/to/outputs"
"""

import os
from pathlib import Path

# ===========================================
# PATH CONFIGURATION
# ===========================================

# Auto-detect root directory (parent of module_1_general_ssl)
_MODULE_DIR = Path(__file__).parent
_ROOT_DIR = _MODULE_DIR.parent

# Data directory - override with DATA_DIR environment variable
DATA_DIR = os.environ.get(
    "DATA_DIR",
    str(_ROOT_DIR / "data" / "medical_images")
)

# Output directory for saved models and results
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    str(_MODULE_DIR / "outputs")
)

# Model save paths
PRETRAINED_ENCODER_PATH = os.path.join(OUTPUT_DIR, "pretrained_encoder.pth")
BEST_GENERATOR_PATH = os.path.join(OUTPUT_DIR, "best_generator.pth")
BEST_GENERATOR_BASELINE_PATH = os.path.join(OUTPUT_DIR, "best_generator_baseline.pth")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================
# MODEL ARCHITECTURE
# ===========================================

IMG_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 768
DECODER_DIM = 256
SCALE_FACTOR = 4

# MAE Configuration
MASK_RATIO = 0.75
DECODER_EMBED_DIM = 512
DECODER_DEPTH = 4
DECODER_NUM_HEADS = 16

# ===========================================
# TRAINING PARAMETERS
# ===========================================

# Pre-training (pretrain.py)
PRETRAIN_EPOCHS = 100
PRETRAIN_BATCH_SIZE = 64
PRETRAIN_LR = 1.5e-4

# Fine-tuning (train.py, train_baseline.py)
FINETUNE_EPOCHS = 50
FINETUNE_BATCH_SIZE = 16
FINETUNE_LR = 1e-4

# Data loading
NUM_WORKERS = 2
PIN_MEMORY = True

# Validation split
VAL_SPLIT = 0.1

# ===========================================
# LOSS WEIGHTS
# ===========================================

LAMBDA_PIXEL = 1.0       # L1 reconstruction loss
LAMBDA_ADV = 1e-3        # Adversarial loss
LAMBDA_PERCEPTUAL = 6e-3 # VGG perceptual loss

# ===========================================
# DATA AUGMENTATION
# ===========================================

RANDOM_FLIP = True
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]

# ===========================================
# DEVICE CONFIGURATION
# ===========================================

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================================
# HELPER FUNCTION
# ===========================================

def print_config():
    """Print current configuration for debugging."""
    print("=" * 50)
    print("Module 1 Configuration")
    print("=" * 50)
    print(f"DATA_DIR:      {DATA_DIR}")
    print(f"OUTPUT_DIR:    {OUTPUT_DIR}")
    print(f"DEVICE:        {DEVICE}")
    print(f"IMG_SIZE:      {IMG_SIZE}")
    print(f"BATCH_SIZE:    {FINETUNE_BATCH_SIZE}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
