# config.py - FINAL 4X SUPER-RESOLUTION CONFIG
import os

# ==========================================
# PATHS
# ==========================================
DRIVE_PATH = '/content/drive/My Drive/mri_project'
DATASET_PATH = os.path.join(DRIVE_PATH, 'Brain_MRI')

CHECKPOINT_PATH = os.path.join(DRIVE_PATH, 'checkpoint.pth')
BEST_MODEL_PATH = os.path.join(DRIVE_PATH, 'best_generator.pth')
MODEL_DIR = os.path.join(DRIVE_PATH, 'models')
OUTPUT_DIR = os.path.join(DRIVE_PATH, 'results')
LOG_DIR = os.path.join(DRIVE_PATH, 'logs')

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================
# DATASET SETTINGS
# ==========================================
CROP_SIZE = 128              # Input patch size (Model outputs 512x512)
SCALE_FACTOR = 4             # 4x Super-Resolution
AUTO_PREPARE_LR_HR = True    # Auto-sorts folders (Fast check enabled)
USE_16BIT = True             # Keep medical precision

# ==========================================
# TRAINING PARAMETERS
# ==========================================
BATCH_SIZE = 8
NUM_WORKERS = 2
PIN_MEMORY = True

LEARNING_RATE_G = 1e-4
LEARNING_RATE_D = 1e-4
N_EPOCHS = 200               # Sufficient for convergence
WARMUP_PERCENTAGE = 0.15     # 15% Pixel-only training for stability
PATIENCE = 20                # Stop if no improvement

# ==========================================
# LOSS WEIGHTS (SHARPNESS & MASK FOCUS)
# ==========================================
LAMBDA_PIXEL = 1.0           # Base structure
LAMBDA_VGG = 0.05            # Perceptual quality (Texture)
LAMBDA_ADV = 0.005           # Adversarial realism
LAMBDA_EDGE = 0.1            # Edge sharpness

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
NUM_RESIDUAL_BLOCKS = 16
IMAGE_CHANNELS = 1           # Grayscale MRI
DEVICE = 'cuda'

# ==========================================
# DATA SPLIT
# ==========================================
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2
TRAIN_SPLIT = 0.7