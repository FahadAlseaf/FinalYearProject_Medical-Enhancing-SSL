# dataset.py - FAST DRIVE LOADING + CORRUPTION SAFETY
import os
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import tifffile
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from config import *

# ==========================================
# DATA LOADER UTILS
# ==========================================

def load_medical_image(path):
    """Safe image loader with 16-bit support."""
    path = str(path)
    if not os.path.exists(path):
        # Return blank if file vanished
        return np.zeros((CROP_SIZE * SCALE_FACTOR, CROP_SIZE * SCALE_FACTOR), dtype=np.float32)

    try:
        if path.lower().endswith(('.tif', '.tiff')):
            img = tifffile.imread(path)
        else:
            img = np.array(Image.open(path))
            
        # Handle Dimensions
        if img.ndim == 3:
            if img.shape[0] < 5: img = img[0, :, :] 
            else: img = img[:, :, 0] 
                
        # Normalize
        img = img.astype(np.float32)
        if img.max() > 255: img /= 65535.0
        else: img /= 255.0
            
        return img
    except Exception as e:
        print(f"⚠️ Corrupted file skipped: {Path(path).name}")
        return np.zeros((128, 128), dtype=np.float32) # Return dummy to prevent crash

# ==========================================
# FAST FOLDER PREPARATION
# ==========================================

def prepare_lr_hr_folders_fast(dataset_path: str, scale: int = 4):
    """
    Scans and sorts dataset. Uses 'any()' for speed to avoid Google Drive hang.
    """
    root = Path(dataset_path)
    hr_dir = root / "HR"
    lr_dir = root / "LR"
    mask_dir = root / "MASK"
    
    # 1. Fast Check: If folders contain data, skip scanning
    if hr_dir.exists() and lr_dir.exists() and mask_dir.exists():
        try:
            if any(hr_dir.iterdir()):
                print(f"✅ Dataset folders detected. Skipping slow scan.")
                return
        except:
            pass # Folder empty, proceed

    hr_dir.mkdir(exist_ok=True)
    lr_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    
    print(f"🔍 Organizing dataset (HR, LR, MASK)...")
    
    # 2. Optimized Scan
    for i, img_path in enumerate(root.rglob("*")):
        if i % 100 == 0: print(f"   Scanning... {i}", end='\r')
        
        if img_path.is_file() and img_path.suffix.lower() in [".tif", ".tiff", ".png", ".jpg"]:
            if img_path.parent.name in ["HR", "LR", "MASK"]: continue
            
            # Sort Masks
            if "mask" in img_path.name.lower():
                target = mask_dir / img_path.name
                if not target.exists(): shutil.copy2(img_path, target)
            
            # Sort Images
            else:
                hr_target = hr_dir / img_path.name
                lr_target = lr_dir / img_path.name
                
                if not hr_target.exists(): shutil.copy2(img_path, hr_target)
                
                # Generate LR if missing
                if not lr_target.exists():
                    try:
                        img = load_medical_image(hr_target)
                        img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
                        lr_t = F.interpolate(img_t, scale_factor=1/scale, mode='bicubic', align_corners=False)
                        lr_np = lr_t.squeeze().numpy()
                        
                        if img_path.suffix.lower() in ['.tif', '.tiff']:
                            tifffile.imwrite(str(lr_target), (lr_np * 65535).astype(np.uint16))
                        else:
                            Image.fromarray((lr_np * 255).astype(np.uint8)).save(lr_target)
                    except:
                        pass
    print("\n✅ Dataset Ready.")

# ==========================================
# DATASET CLASS
# ==========================================

class PairedSRDataset(Dataset):
    def __init__(self, root_dir, crop_size=128, scale=4, augment=False):
        self.root = Path(root_dir)
        self.lr_dir = self.root / "LR"
        self.hr_dir = self.root / "HR"
        self.mask_dir = self.root / "MASK"
        self.scale = scale
        self.crop_size = crop_size
        self.augment = augment
        
        self.pairs = []
        # Get HR files
        hr_files = sorted(list(self.hr_dir.glob("*")))
        
        for hr_path in hr_files:
            lr_path = self.lr_dir / hr_path.name
            
            # Flexible Mask Find
            mask_names = [
                hr_path.name,
                f"{hr_path.stem}_mask{hr_path.suffix}",
                f"{hr_path.stem}_masked{hr_path.suffix}",
                f"{hr_path.stem}mask{hr_path.suffix}"
            ]
            
            mask_path = None
            for name in mask_names:
                if (self.mask_dir / name).exists():
                    mask_path = self.mask_dir / name
                    break
            
            if lr_path.exists() and mask_path:
                self.pairs.append((lr_path, hr_path, mask_path))

        print(f"📁 Dataset Loaded: {len(self.pairs)} pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_path, hr_path, mask_path = self.pairs[idx]
        
        lr_np = load_medical_image(lr_path)
        hr_np = load_medical_image(hr_path)
        mask_np = load_medical_image(mask_path)
        
        lr_t = torch.from_numpy(lr_np).unsqueeze(0).float()
        hr_t = torch.from_numpy(hr_np).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).float()
        
        # Ensure mask matches HR size
        if mask_t.shape != hr_t.shape:
             mask_t = F.interpolate(mask_t.unsqueeze(0), size=hr_t.shape[1:], mode='nearest').squeeze(0)

        # Random Crop
        _, h, w = lr_t.shape
        lr_sz = self.crop_size // self.scale
        
        if h >= lr_sz and w >= lr_sz:
            top = np.random.randint(0, h - lr_sz + 1)
            left = np.random.randint(0, w - lr_sz + 1)
            
            lr_t = lr_t[:, top:top+lr_sz, left:left+lr_sz]
            hr_t = hr_t[:, top*self.scale:(top+lr_sz)*self.scale, 
                        left*self.scale:(left+lr_sz)*self.scale]
            mask_t = mask_t[:, top*self.scale:(top+lr_sz)*self.scale, 
                            left*self.scale:(left+lr_sz)*self.scale]
        
        if self.augment:
            if np.random.random() > 0.5: 
                lr_t = torch.flip(lr_t, [2])
                hr_t = torch.flip(hr_t, [2])
                mask_t = torch.flip(mask_t, [2])
                
        return lr_t, hr_t, mask_t

def get_dataloaders(dataset_path, batch_size=8, crop_size=128, scale=4):
    if AUTO_PREPARE_LR_HR: prepare_lr_hr_folders_fast(dataset_path, scale)
    
    full_ds = PairedSRDataset(dataset_path, crop_size, scale, augment=True)
    total = len(full_ds)
    if total == 0: return [], [], []
    
    train_sz = int(total * TRAIN_SPLIT)
    val_sz = int(total * VAL_SPLIT)
    test_sz = total - train_sz - val_sz
    
    train_ds, val_ds, test_ds = random_split(full_ds, [train_sz, val_sz, test_sz])
    
    return (
        DataLoader(train_ds, batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY),
        DataLoader(val_ds, batch_size=1, shuffle=False),
        DataLoader(test_ds, batch_size=1, shuffle=False)
    )