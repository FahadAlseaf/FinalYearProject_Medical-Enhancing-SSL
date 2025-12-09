"""
dataset.py - Data Loading Utilities for MRI Super-Resolution

This module provides dataset classes and utilities for loading and
preprocessing brain MRI images for super-resolution training.

Key Features:
    - 16-bit TIFF support for medical precision
    - Automatic HR/LR/Mask folder organization
    - On-the-fly LR generation
    - Corruption-safe loading with graceful fallbacks
"""

import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, List
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

# Import tifffile with fallback
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("Warning: tifffile not installed. 16-bit TIFF support disabled.")

from config import (
    CROP_SIZE, SCALE_FACTOR, AUTO_PREPARE_LR_HR,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    NUM_WORKERS, PIN_MEMORY, BATCH_SIZE
)


def load_medical_image(path: str) -> np.ndarray:
    """
    Safely load a medical image with 16-bit support.
    
    Handles various formats (TIFF, PNG, JPG) and bit depths (8-bit, 16-bit).
    Gracefully handles corrupted files by returning a blank array.
    
    Args:
        path: Path to the image file
        
    Returns:
        Normalized float32 array with values in [0, 1]
    """
    path = str(path)
    
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return np.zeros((CROP_SIZE * SCALE_FACTOR, CROP_SIZE * SCALE_FACTOR), dtype=np.float32)
    
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
        return np.zeros((CROP_SIZE * SCALE_FACTOR, CROP_SIZE * SCALE_FACTOR), dtype=np.float32)


def prepare_lr_hr_folders_fast(dataset_path: str, scale: int = 4) -> None:
    """
    Automatically organize dataset into HR, LR, and MASK folders.
    
    This function:
    1. Creates HR, LR, MASK subdirectories if needed
    2. Moves/copies images to appropriate folders based on filename
    3. Generates LR images from HR if missing
    
    Args:
        dataset_path: Root path to the dataset
        scale: Downsampling scale factor
        
    Note:
        Uses fast checks to avoid slow Google Drive scans.
    """
    root = Path(dataset_path)
    hr_dir = root / "HR"
    lr_dir = root / "LR"
    mask_dir = root / "MASK"
    
    # Quick check: skip if folders already populated
    if hr_dir.exists() and lr_dir.exists() and mask_dir.exists():
        try:
            if any(hr_dir.iterdir()):
                print("✅ Dataset folders already organized. Skipping preparation.")
                return
        except StopIteration:
            pass  # Empty folder, proceed with setup
    
    # Create directories
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Organizing dataset in {dataset_path}...")
    
    # Supported extensions
    extensions = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    
    # Scan and organize
    file_count = 0
    for img_path in root.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in extensions:
            continue
        if img_path.parent.name in ["HR", "LR", "MASK"]:
            continue
        
        file_count += 1
        if file_count % 100 == 0:
            print(f"   Processing file {file_count}...", end='\r')
        
        filename = img_path.name.lower()
        
        # Sort masks
        if "mask" in filename:
            target = mask_dir / img_path.name
            if not target.exists():
                shutil.copy2(img_path, target)
            continue
        
        # Sort images
        hr_target = hr_dir / img_path.name
        lr_target = lr_dir / img_path.name
        
        # Copy to HR if not exists
        if not hr_target.exists():
            shutil.copy2(img_path, hr_target)
        
        # Generate LR if not exists
        if not lr_target.exists():
            try:
                img = load_medical_image(str(hr_target))
                img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
                
                # Downsample
                lr_tensor = F.interpolate(
                    img_tensor,
                    scale_factor=1/scale,
                    mode='bicubic',
                    align_corners=False
                )
                lr_np = lr_tensor.squeeze().numpy()
                
                # Save LR
                if img_path.suffix.lower() in ['.tif', '.tiff'] and HAS_TIFFFILE:
                    tifffile.imwrite(str(lr_target), (lr_np * 65535).astype(np.uint16))
                else:
                    Image.fromarray((lr_np * 255).astype(np.uint8)).save(lr_target)
                    
            except Exception as e:
                print(f"Warning: Failed to generate LR for {img_path.name}: {e}")
    
    print(f"\n✅ Dataset organization complete. Processed {file_count} files.")


class PairedSRDataset(Dataset):
    """
    Dataset for paired LR-HR-Mask training.
    
    Loads corresponding Low-Resolution, High-Resolution, and Mask images
    for training the super-resolution model.
    
    Args:
        root_dir: Path to dataset with HR, LR, MASK subdirectories
        crop_size: Size of random crops for HR images
        scale: Super-resolution scale factor
        augment: Whether to apply data augmentation
        
    Returns:
        Tuple of (lr_tensor, hr_tensor, mask_tensor)
    """
    
    def __init__(
        self,
        root_dir: str,
        crop_size: int = 128,
        scale: int = 4,
        augment: bool = False
    ):
        self.root = Path(root_dir)
        self.lr_dir = self.root / "LR"
        self.hr_dir = self.root / "HR"
        self.mask_dir = self.root / "MASK"
        self.scale = scale
        self.crop_size = crop_size
        self.augment = augment
        
        # Build list of valid triplets (lr, hr, mask)
        self.pairs: List[Tuple[Path, Path, Path]] = []
        
        if not self.hr_dir.exists():
            print(f"Warning: HR directory not found: {self.hr_dir}")
            return
            
        hr_files = sorted(list(self.hr_dir.glob("*")))
        
        for hr_path in hr_files:
            if hr_path.is_dir():
                continue
                
            lr_path = self.lr_dir / hr_path.name
            
            # Try multiple mask naming conventions
            mask_candidates = [
                hr_path.name,
                f"{hr_path.stem}_mask{hr_path.suffix}",
                f"{hr_path.stem}_masked{hr_path.suffix}",
                f"{hr_path.stem}mask{hr_path.suffix}"
            ]
            
            mask_path = None
            for candidate in mask_candidates:
                potential_mask = self.mask_dir / candidate
                if potential_mask.exists():
                    mask_path = potential_mask
                    break
            
            if lr_path.exists() and mask_path is not None:
                self.pairs.append((lr_path, hr_path, mask_path))
        
        if len(self.pairs) == 0:
            print(f"Warning: No valid image triplets found in {root_dir}")
            print("  Ensure HR, LR, and MASK folders contain matching images.")
        else:
            print(f"📁 Dataset loaded: {len(self.pairs)} image triplets")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lr_path, hr_path, mask_path = self.pairs[idx]
        
        # Load images
        lr_np = load_medical_image(str(lr_path))
        hr_np = load_medical_image(str(hr_path))
        mask_np = load_medical_image(str(mask_path))
        
        # Convert to tensors
        lr_tensor = torch.from_numpy(lr_np).unsqueeze(0).float()
        hr_tensor = torch.from_numpy(hr_np).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
        
        # Ensure mask matches HR size
        if mask_tensor.shape != hr_tensor.shape:
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0),
                size=hr_tensor.shape[1:],
                mode='nearest'
            ).squeeze(0)
        
        # Random crop
        _, h, w = lr_tensor.shape
        lr_crop_size = self.crop_size // self.scale
        
        if h >= lr_crop_size and w >= lr_crop_size:
            top = np.random.randint(0, h - lr_crop_size + 1)
            left = np.random.randint(0, w - lr_crop_size + 1)
            
            lr_tensor = lr_tensor[
                :,
                top:top + lr_crop_size,
                left:left + lr_crop_size
            ]
            hr_tensor = hr_tensor[
                :,
                top * self.scale:(top + lr_crop_size) * self.scale,
                left * self.scale:(left + lr_crop_size) * self.scale
            ]
            mask_tensor = mask_tensor[
                :,
                top * self.scale:(top + lr_crop_size) * self.scale,
                left * self.scale:(left + lr_crop_size) * self.scale
            ]
        
        # Data augmentation
        if self.augment and np.random.random() > 0.5:
            lr_tensor = torch.flip(lr_tensor, [2])
            hr_tensor = torch.flip(hr_tensor, [2])
            mask_tensor = torch.flip(mask_tensor, [2])
        
        return lr_tensor, hr_tensor, mask_tensor


def get_dataloaders(
    dataset_path: str,
    batch_size: int = BATCH_SIZE,
    crop_size: int = CROP_SIZE,
    scale: int = SCALE_FACTOR
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        dataset_path: Path to the dataset
        batch_size: Batch size for training
        crop_size: Crop size for HR images
        scale: Super-resolution scale factor
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Prepare folders if enabled
    if AUTO_PREPARE_LR_HR:
        prepare_lr_hr_folders_fast(dataset_path, scale)
    
    # Create full dataset
    full_dataset = PairedSRDataset(
        dataset_path,
        crop_size,
        scale,
        augment=True
    )
    
    if len(full_dataset) == 0:
        print("Error: Empty dataset. Cannot create dataloaders.")
        return None, None, None
    
    # Calculate split sizes
    total = len(full_dataset)
    train_size = int(total * TRAIN_SPLIT)
    val_size = int(total * VAL_SPLIT)
    test_size = total - train_size - val_size
    
    # Split dataset
    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [train_size, val_size, test_size]
    )
    
    print(f"📊 Dataset splits: Train={train_size}, Val={val_size}, Test={test_size}")
    
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
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    return train_loader, val_loader, test_loader


# ===========================================
# Testing
# ===========================================

if __name__ == "__main__":
    from config import DATASET_PATH
    
    print("Testing dataset loading...")
    print(f"Dataset path: {DATASET_PATH}")
    
    train_loader, val_loader, test_loader = get_dataloaders(DATASET_PATH)
    
    if train_loader is not None:
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test one batch
        for lr, hr, mask in train_loader:
            print(f"\nSample batch shapes:")
            print(f"  LR: {lr.shape}")
            print(f"  HR: {hr.shape}")
            print(f"  Mask: {mask.shape}")
            break
        
        print("\n✅ Dataset test complete!")
    else:
        print("\n❌ Failed to create dataloaders. Check dataset path.")
