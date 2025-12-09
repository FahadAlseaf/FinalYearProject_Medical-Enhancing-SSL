"""
classify.py - Tumor Classification with Attention Heatmap Visualization

This script fine-tunes the pre-trained MIRAM encoder for binary tumor
classification (tumor vs no-tumor) and generates attention heatmaps
showing where the model focuses for its predictions.

Key Features:
    - Transfer learning from self-supervised MIRAM encoder
    - Automatic labeling based on mask files (tumor if mask > 0)
    - Attention visualization overlaid on MRI scans
    - Classification metrics (Accuracy, Precision, Recall, F1)

Usage:
    python classify.py

Output:
    - best_tumor_classifier.pth: Trained classifier weights
    - tumor_viz_*.png: Attention heatmap visualizations
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    DEVICE, DATASET_PATH, OUTPUT_DIR,
    BEST_MODEL_PATH, BEST_CLASSIFIER_PATH,
    IMG_SIZE, PATCH_SIZE, EMBED_DIM,
    BATCH_SIZE, TRAIN_SPLIT, CLASSIFIER_EPOCHS, CLASSIFIER_LR
)
from models import MIRAM
from dataset import load_medical_image


# ===========================================
# TUMOR DATASET (Auto-labeled from masks)
# ===========================================

class TumorDataset(Dataset):
    """
    Dataset for tumor classification with automatic labeling.
    
    Labels are determined by examining the corresponding mask file:
    - Label = 1 (Tumor) if mask has any non-zero pixels
    - Label = 0 (No Tumor) if mask is all zeros
    
    Args:
        root_dir: Path to dataset root
        
    Expected structure:
        root_dir/
        ├── HR/
        │   ├── image1.tif
        │   └── image2.tif
        └── MASK/
            ├── image1.tif (or image1_mask.tif)
            └── image2.tif
    """
    
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.hr_dir = self.root / "HR"
        self.mask_dir = self.root / "MASK"
        self.samples = []
        
        if not self.hr_dir.exists():
            print(f"⚠️ HR folder not found at {self.hr_dir}")
            return
        
        if not self.mask_dir.exists():
            print(f"⚠️ MASK folder not found at {self.mask_dir}")
            return
        
        print("🔍 Auto-labeling data based on masks...")
        hr_files = sorted(list(self.hr_dir.glob("*")))
        
        tumor_count = 0
        normal_count = 0
        
        for hr_path in hr_files:
            # Try different mask naming conventions
            mask_names = [
                hr_path.name,
                f"{hr_path.stem}_mask{hr_path.suffix}",
                f"{hr_path.stem}_MASK{hr_path.suffix}"
            ]
            
            mask_path = None
            for name in mask_names:
                candidate = self.mask_dir / name
                if candidate.exists():
                    mask_path = candidate
                    break
            
            if mask_path:
                # Determine label from mask content
                mask = load_medical_image(mask_path)
                label = 1 if np.max(mask) > 0 else 0
                self.samples.append((hr_path, label, mask_path))
                
                if label == 1:
                    tumor_count += 1
                else:
                    normal_count += 1
        
        print(f"📊 Dataset: {len(self.samples)} samples")
        print(f"   Tumor: {tumor_count}, Normal: {normal_count}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, label, mask_path = self.samples[idx]
        
        # Load and preprocess image
        img_np = load_medical_image(img_path)
        img_t = torch.from_numpy(img_np).unsqueeze(0).float()
        img_t = torch.nn.functional.interpolate(
            img_t.unsqueeze(0),
            size=(IMG_SIZE, IMG_SIZE),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        return (
            img_t,
            torch.tensor(label, dtype=torch.float32),
            str(mask_path),
            str(img_path)
        )


def get_classification_dataloaders() -> tuple:
    """Create train and test dataloaders for classification."""
    full_ds = TumorDataset(DATASET_PATH)
    
    if len(full_ds) == 0:
        return None, None
    
    total = len(full_ds)
    train_sz = int(total * TRAIN_SPLIT)
    test_sz = total - train_sz
    
    train_ds, test_ds = random_split(full_ds, [train_sz, test_sz])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)  # batch=1 for visualization
    
    return train_loader, test_loader


# ===========================================
# CLASSIFIER MODEL
# ===========================================

class MIRAMClassifier(nn.Module):
    """
    Tumor classifier using pre-trained MIRAM encoder.
    
    Architecture:
        - Frozen MIRAM encoder (except last block)
        - Classification head on [CLS] token
        
    The last transformer block is unfrozen to allow
    attention tuning for the classification task.
    """
    
    def __init__(self, pretrained_path: str = BEST_MODEL_PATH):
        super().__init__()
        
        # Load pre-trained MIRAM encoder
        self.miram = MIRAM()
        
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location=DEVICE, weights_only=True)
            self.miram.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded pre-trained MIRAM from {pretrained_path}")
        else:
            print("⚠️ No pre-trained weights found. Training from scratch.")
        
        # Freeze most parameters
        for param in self.miram.parameters():
            param.requires_grad = False
        
        # Unfreeze last transformer block for attention tuning
        for param in self.miram.blocks[-1].parameters():
            param.requires_grad = True
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(EMBED_DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> tuple:
        """
        Forward pass with optional attention output.
        
        Args:
            x: Input images (N, 1, H, W)
            return_attention: If True, also return attention weights
            
        Returns:
            pred: Sigmoid predictions (N, 1)
            attn: Optional attention weights from last block
        """
        if return_attention:
            latent, _, _, attn = self.miram.forward_encoder(
                x, mask_ratio=0.0, return_last_attention=True
            )
            cls_token = latent[:, 0, :]  # [CLS] token
            pred = torch.sigmoid(self.head(cls_token))
            return pred, attn
        else:
            latent, _, _ = self.miram.forward_encoder(x, mask_ratio=0.0)
            cls_token = latent[:, 0, :]
            pred = torch.sigmoid(self.head(cls_token))
            return pred


# ===========================================
# ATTENTION VISUALIZATION
# ===========================================

def visualize_tumor_attention(
    model: nn.Module,
    test_loader: DataLoader,
    num_samples: int = 3
):
    """
    Generate attention heatmap visualizations for tumor cases.
    
    Creates side-by-side plots showing:
    1. Original MRI scan
    2. Ground truth tumor mask
    3. Model attention heatmap overlay
    
    Args:
        model: Trained classifier model
        test_loader: Test dataloader
        num_samples: Number of visualizations to generate
    """
    print("\n🖼️ Generating Tumor Attention Maps...")
    model.eval()
    count = 0
    
    for img, label, mask_path, img_path in test_loader:
        # Only visualize tumor-positive cases
        if label.item() != 1:
            continue
        
        img = img.to(DEVICE)
        
        # Get prediction and attention
        with torch.no_grad():
            pred, attn = model(img, return_attention=True)
        
        # Extract attention from [CLS] token to all patches
        # attn shape: [Batch, Heads, Tokens, Tokens]
        if attn.dim() == 4:
            # Average over all heads, get CLS attention to patches
            cls_attn = attn[0, :, 0, 1:].mean(dim=0)
        else:
            # Shape: [Batch, Tokens, Tokens]
            cls_attn = attn[0, 0, 1:]
        
        # Reshape to spatial grid
        grid_size = IMG_SIZE // PATCH_SIZE
        attn_map = cls_attn.reshape(grid_size, grid_size).cpu().numpy()
        
        # Resize heatmap to image size
        attn_map = cv2.resize(attn_map, (IMG_SIZE, IMG_SIZE))
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        # Get original image
        orig_img = img[0, 0].cpu().numpy()
        
        # Get ground truth mask
        gt_mask = load_medical_image(mask_path[0])
        gt_mask = cv2.resize(gt_mask, (IMG_SIZE, IMG_SIZE))
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original MRI
        axes[0].imshow(orig_img, cmap='gray')
        axes[0].set_title(f"MRI (Tumor Positive)\nConfidence: {pred.item():.2f}", fontsize=12)
        axes[0].axis('off')
        
        # Ground truth mask
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title("Ground Truth Mask", fontsize=12)
        axes[1].axis('off')
        
        # Attention heatmap overlay
        axes[2].imshow(orig_img, cmap='gray')
        im = axes[2].imshow(attn_map, cmap='jet', alpha=0.5)
        axes[2].set_title("Model Attention (Localization)", fontsize=12)
        axes[2].axis('off')
        
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, f'tumor_viz_{count}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved: {save_path}")
        
        count += 1
        if count >= num_samples:
            break
    
    if count == 0:
        print("   ⚠️ No tumor-positive samples found in test set.")


# ===========================================
# TRAINING AND EVALUATION
# ===========================================

def train_and_evaluate():
    """Main function to train classifier and generate visualizations."""
    
    print("=" * 60)
    print("MIRAM Tumor Classification")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {CLASSIFIER_EPOCHS}")
    print("=" * 60)
    
    # Load data
    train_loader, test_loader = get_classification_dataloaders()
    
    if train_loader is None:
        print("Error: Could not load dataset.")
        return
    
    # Initialize model
    model = MIRAMClassifier().to(DEVICE)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CLASSIFIER_LR
    )
    criterion = nn.BCELoss()
    
    # Training loop
    print("\n📚 Training Classifier...")
    
    for epoch in range(CLASSIFIER_EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CLASSIFIER_EPOCHS}")
        for batch in pbar:
            imgs = batch[0].to(DEVICE)
            labels = batch[1].to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), BEST_CLASSIFIER_PATH)
    print(f"\n💾 Model saved to {BEST_CLASSIFIER_PATH}")
    
    # Generate attention visualizations
    visualize_tumor_attention(model, test_loader, num_samples=3)
    
    # Evaluation metrics
    print("\n📊 CLASSIFICATION REPORT")
    print("-" * 40)
    
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for imgs, labels, _, _ in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            predictions = (outputs > 0.5).float()
            
            y_pred.extend(predictions.cpu().numpy().flatten())
            y_true.extend(labels.numpy().flatten())
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 40)
    
    print("\n✅ Classification Complete!")


if __name__ == "__main__":
    train_and_evaluate()
