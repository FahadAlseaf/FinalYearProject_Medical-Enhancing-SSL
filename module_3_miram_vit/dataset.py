# dataset.py - MIRAM DATA LOADER
import os
import torch
import numpy as np
import tifffile
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
from config import *

# ==========================================
# IMAGE LOADER
# ==========================================
def load_medical_image(path):
    """Loads image, handles 16-bit, returns numpy array [H, W]"""
    path = str(path)
    if not os.path.exists(path):
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    try:
        if path.lower().endswith(('.tif', '.tiff')):
            img = tifffile.imread(path)
        else:
            img = np.array(Image.open(path))
            
        if img.ndim == 3:
            img = img[0, :, :] if img.shape[0] < 5 else img[:, :, 0]
                
        img = img.astype(np.float32)
        if img.max() > 255:
            img /= 65535.0
        else:
            img /= 255.0
            
        return img
    except:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

# ==========================================
# PRE-TRAINING DATASET (Restoration)
# ==========================================
class MIRAMDataset(Dataset):
    def __init__(self, root_dir):
        self.root = Path(root_dir)
        self.hr_dir = self.root / "HR"
        
        if not self.hr_dir.exists():
            print(f"⚠️ HR folder not found at {self.hr_dir}")
            self.files = []
        else:
            self.files = sorted(list(self.hr_dir.glob("*")))
            
        print(f"📁 MIRAM Pre-training Dataset: {len(self.files)} images found.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img_np = load_medical_image(path) # [H, W]
        
        # To Tensor [1, H, W]
        img_t = torch.from_numpy(img_np).unsqueeze(0)
        
        # Resize to 224x224 (Required for ViT patches)
        img_t = transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True)(img_t)
        
        # Create Coarse Target (Downsampled)
        coarse_size = int(IMG_SIZE * SCALE_COARSE)
        img_coarse = transforms.Resize((coarse_size, coarse_size), antialias=True)(img_t)
        
        return img_t, img_coarse

def get_dataloaders():
    full_ds = MIRAMDataset(DATASET_PATH)
    if len(full_ds) == 0: return None, None, None
    
    total = len(full_ds)
    train_sz = int(total * TRAIN_SPLIT)
    val_sz = int(total * VAL_SPLIT)
    test_sz = total - train_sz - val_sz
    
    train_ds, val_ds, test_ds = random_split(full_ds, [train_sz, val_sz, test_sz])
    
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(test_ds, batch_size=1, shuffle=False)
    )