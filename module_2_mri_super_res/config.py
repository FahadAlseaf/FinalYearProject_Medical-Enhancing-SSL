"""
Configuration for Module 2: Brain MRI Super-Resolution

This module provides centralized configuration for all training scripts.
Paths can be overridden using environment variables.

Usage:
    # Option 1: Edit this file directly
    DRIVE_PATH = "./your/data/path"
    
    # Option 2: Use environment variables
    export MRI_DATA_PATH="/path/to/Brain_MRI"
    export MRI_OUTPUT_PATH="/path/to/outputs"
"""

import os
from pathlib import Path

# ===========================================
# PATH CONFIGURATION
# ===========================================

# Auto-detect root directory
_MODULE_DIR = Path(__file__).parent
_ROOT_DIR = _MODULE_DIR.parent

# Base path - override with MRI_DATA_PATH environment variable
DRIVE_PATH = os.environ.get(
    "MRI_DATA_PATH",
    str(_ROOT_DIR / "data" / "mri_project")
)

# Dataset path
DATASET_PATH = os.path.join(DRIVE_PATH, "Brain_MRI")

# Output paths
OUTPUT_BASE = os.environ.get("MRI_OUTPUT_PATH", DRIVE_PATH)
CHECKPOINT_PATH = os.path.join(OUTPUT_BASE, "checkpoint.pth")
BEST_MODEL_PATH = os.path.join(OUTPUT_BASE, "best_generator.pth")
MODEL_DIR = os.path.join(OUTPUT_BASE, "models")
OUTPUT_DIR = os.path.join(OUTPUT_BASE, "results")
LOG_DIR = os.path.join(OUTPUT_BASE, "logs")

# Create directories
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ===========================================
# DATASET SETTINGS
# ===========================================

CROP_SIZE = 128              # Input patch size (Model outputs 512x512)
SCALE_FACTOR = 4             # 4x Super-Resolution
AUTO_PREPARE_LR_HR = True    # Auto-sorts folders (Fast check enabled)
USE_16BIT = True             # Keep medical precision for TIFF files

# ===========================================
# TRAINING PARAMETERS
# ===========================================

BATCH_SIZE = 8
NUM_WORKERS = 2
PIN_MEMORY = True

LEARNING_RATE_G = 1e-4
LEARNING_RATE_D = 1e-4
N_EPOCHS = 200               # Sufficient for convergence
WARMUP_PERCENTAGE = 0.15     # 15% Pixel-only training for stability
PATIENCE = 20                # Early stopping patience

# ===========================================
# LOSS WEIGHTS
# ===========================================

LAMBDA_PIXEL = 1.0           # Base structure (Charbonnier)
LAMBDA_VGG = 0.05            # Perceptual quality (Texture)
LAMBDA_ADV = 0.005           # Adversarial realism
LAMBDA_EDGE = 0.1            # Edge sharpness

# ===========================================
# MODEL ARCHITECTURE
# ===========================================

NUM_RESIDUAL_BLOCKS = 16
IMAGE_CHANNELS = 1           # Grayscale MRI

# ===========================================
# DATA SPLIT
# ===========================================

VAL_SPLIT = 0.1
TEST_SPLIT = 0.2
TRAIN_SPLIT = 0.7

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
    print("Module 2 Configuration (MRI Super-Resolution)")
    print("=" * 50)
    print(f"DATASET_PATH:  {DATASET_PATH}")
    print(f"OUTPUT_DIR:    {OUTPUT_DIR}")
    print(f"DEVICE:        {DEVICE}")
    print(f"SCALE_FACTOR:  {SCALE_FACTOR}x")
    print(f"BATCH_SIZE:    {BATCH_SIZE}")
    print(f"N_EPOCHS:      {N_EPOCHS}")
    print(f"USE_16BIT:     {USE_16BIT}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
