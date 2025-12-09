# config.py - MIRAM CONFIGURATION 
import os
import torch

# ==========================================
# DEVICE CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# PATHS
# ==========================================
DRIVE_PATH = '/content/drive/My Drive/mri_project'
DATASET_PATH = os.path.join(DRIVE_PATH, 'Brain_MRI')

# Models
CHECKPOINT_PATH = os.path.join(DRIVE_PATH, 'miram_checkpoint.pth')
BEST_MODEL_PATH = os.path.join(DRIVE_PATH, 'best_miram_mae.pth')
BEST_CLASSIFIER_PATH = os.path.join(DRIVE_PATH, 'best_tumor_classifier.pth')

# Output Folders
MODEL_DIR = os.path.join(DRIVE_PATH, 'models')
OUTPUT_DIR = os.path.join(DRIVE_PATH, 'results_miram')
LOG_DIR = os.path.join(DRIVE_PATH, 'logs_miram')

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================
# IMAGE SETTINGS (ViT Specific)
# ==========================================
IMG_SIZE = 224              # Standard ViT Resolution
PATCH_SIZE = 16             # 16x16 patches
IN_CHANNELS = 1             # Grayscale
USE_16BIT = True

# ==========================================
# MODEL HYPERPARAMETERS (ViT-Small)
# ==========================================
EMBED_DIM = 384
DEPTH = 12
NUM_HEADS = 6
MLP_RATIO = 4.0

# Decoder (Lightweight)
DECODER_EMBED_DIM = 256
DECODER_DEPTH = 4
DECODER_NUM_HEADS = 8

# MIRAM Specifics
MASK_RATIO = 0.75           # 75% hidden
SCALE_COARSE = 0.5          # 112x112 target
SCALE_FINE = 1.0            # 224x224 target

# ==========================================
# TRAINING SETTINGS
# ==========================================
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True

LEARNING_RATE = 1.5e-4
WEIGHT_DECAY = 0.05
N_EPOCHS = 200
WARMUP_EPOCHS = 20

# Loss Weights
LAMBDA_FINE = 1.0
LAMBDA_COARSE = 0.5

# ==========================================
# DATA SPLIT
# ==========================================
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
TRAIN_SPLIT = 0.8