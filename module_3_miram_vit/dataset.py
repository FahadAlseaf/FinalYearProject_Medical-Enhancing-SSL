"""
dataset.py - MIRAM Data Loading Utilities

This module provides dataset classes for loading medical images
for MIRAM pre-training (self-supervised reconstruction).

Key Features:
    - 16-bit TIFF support for medical precision
    - Automatic multi-scale target generation
    - Robust error handling for corrupted files
"""

import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms

# Import tifffile with fallback
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("Warning: tifffile not installed. 16-bit TIFF support disabled.")

from config import (
    DATASET_PATH, IMG_SIZE, SCALE_COARSE,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
)


def load_medical_image(path: str) -> np.ndarray:
    """
    Load a medical image with 16-bit support.
    
    Handles various formats (TIFF, PNG, JPG) and bit depths.
    Gracefully handles corrupted files by returning a blank array.
    
    Args:
        path: Path to the image file
        
    Returns:
        Normalized float32 array with values in [0, 1], shape (H, W)
    """
    path = str(path)
    
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    try:
        # Load based on file type
        if path.lower().endswith(('.tif', '.tiff')) and HAS_TIFFFILE:
            img = tifffile.imread(path)
        else:
            img = np.array(Image.open(path))
        
        # Handle multi-channel images (convert to grayscale)
        if img.ndim == 3:
            if img.shape[0] < 5:
                # Channel-first format (e.g., 3xHxW)
                img = img[0, :, :]
            else:
                # Channel-last format (e.g., HxWx3)
                img = img[:, :, 0]
        
        # Normalize to [0, 1]
        img = img.astype(np.float32)
        if img.max() > 255:
            # 16-bit image
            img /= 65535.0
        elif img.max() > 1:
            # 8-bit image
            img /= 255.0
        
        return img
        
    except Exception as e:
        print(f"Warning: Failed to load {Path(path).name}: {e}")
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)


class MIRAMDataset(Dataset):
    """
    Dataset for MIRAM self-supervised pre-training.
    
    Loads high-resolution images and generates multi-scale targets
    for dual-scale reconstruction training.
    
    Args:
        root_dir: Path to dataset root (should contain HR/ subfolder)
        
    Returns:
        img_fine: Fine-scale image tensor (1, H, W) at IMG_SIZE
        img_coarse: Coarse-scale image tensor (1, H/2, W/2)
        
    Expected folder structure:
        root_dir/
        └── HR/
            ├── image1.tif
            ├── image2.tif
            └── ...
    """
    
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.hr_dir = self.root / "HR"
        
        if not self.hr_dir.exists():
            print(f"⚠️ HR folder not found at {self.hr_dir}")
            self.files = []
        else:
            # Find all image files
            extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
            self.files = []
            for ext in extensions:
                self.files.extend(sorted(self.hr_dir.glob(f'*{ext}')))
                self.files.extend(sorted(self.hr_dir.glob(f'*{ext.upper()}')))
            self.files = list(set(self.files))  # Remove duplicates
        
        print(f"📁 MIRAM Pre-training Dataset: {len(self.files)} images found.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple:
        path = self.files[idx]
        img_np = load_medical_image(path)
        
        # Convert to tensor (1, H, W)
        img_t = torch.from_numpy(img_np).unsqueeze(0).float()
        
        # Resize to standard size (224x224 for ViT)
        img_t = transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True)(img_t)
        
        # Create coarse target (downsampled)
        coarse_size = int(IMG_SIZE * SCALE_COARSE)
        img_coarse = transforms.Resize((coarse_size, coarse_size), antialias=True)(img_t)
        
        return img_t, img_coarse


def get_dataloaders(
    dataset_path: str = DATASET_PATH,
    batch_size: int = BATCH_SIZE
) -> tuple:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        dataset_path: Path to the dataset
        batch_size: Training batch size
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        Returns (None, None, None) if dataset is empty
    """
    full_ds = MIRAMDataset(dataset_path)
    
    if len(full_ds) == 0:
        print("Error: Empty dataset. Cannot create dataloaders.")
        return None, None, None
    
    # Calculate split sizes
    total = len(full_ds)
    train_sz = int(total * TRAIN_SPLIT)
    val_sz = int(total * VAL_SPLIT)
    test_sz = total - train_sz - val_sz
    
    # Split dataset
    train_ds, val_ds, test_ds = random_split(full_ds, [train_sz, val_sz, test_sz])
    
    print(f"📊 Dataset splits: Train={train_sz}, Val={val_sz}, Test={test_sz}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader


# ===========================================
# TESTING
# ===========================================

if __name__ == "__main__":
    print("Testing dataset loading...")
    print(f"Dataset path: {DATASET_PATH}")
    
    train_loader, val_loader, test_loader = get_dataloaders()
    
    if train_loader is not None:
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test one batch
        for img_fine, img_coarse in train_loader:
            print(f"\nSample batch shapes:")
            print(f"  Fine: {img_fine.shape}")
            print(f"  Coarse: {img_coarse.shape}")
            break
        
        print("\n✅ Dataset test complete!")
    else:
        print("\n❌ Failed to create dataloaders. Check dataset path.")
