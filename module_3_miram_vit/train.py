"""
train.py - MIRAM Self-Supervised Pre-training

This script trains the MIRAM model using masked autoencoding for
self-supervised learning on medical images.

Training Strategy:
    - Random masking of 75% patches
    - Reconstruction of masked patches at two scales
    - AdamW optimizer with cosine annealing

Usage:
    python train.py

Output:
    - best_miram_mae.pth: Best model weights
    - loss_curve.png: Training visualization
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import (
    DEVICE, BEST_MODEL_PATH, OUTPUT_DIR,
    N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MASK_RATIO
)
from dataset import get_dataloaders
from models import MIRAM
from losses import MIRAMLoss


def plot_training_curves(history: dict, save_dir: str):
    """
    Generate and save training visualization plots.
    
    Args:
        history: Dictionary with 'train_loss' and 'val_loss' lists
        save_dir: Directory to save plots
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.title('MIRAM Model Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training plot saved to {save_path}")


def train():
    """Main training function."""
    
    print("=" * 60)
    print("MIRAM (ViT-MAE) Self-Supervised Pre-training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {N_EPOCHS}")
    print(f"Mask Ratio: {MASK_RATIO}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("=" * 60)
    
    # Load data
    train_dl, val_dl, _ = get_dataloaders()
    if train_dl is None:
        print("Error: Could not load dataset.")
        return
    
    # Initialize model
    model = MIRAM().to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    criterion = MIRAMLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=N_EPOCHS,
        eta_min=1e-6
    )
    
    # Training history
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(N_EPOCHS):
        # === Training ===
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        for img_fine, img_coarse in pbar:
            img_fine = img_fine.to(DEVICE)
            img_coarse = img_coarse.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_fine, pred_coarse, mask = model(img_fine, MASK_RATIO)
            
            # Compute loss
            loss, l_fine, l_coarse = criterion(
                pred_fine, pred_coarse,
                img_fine, img_coarse,
                mask
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train = train_loss / len(train_dl)
        
        # === Validation ===
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for img_fine, img_coarse in val_dl:
                img_fine = img_fine.to(DEVICE)
                img_coarse = img_coarse.to(DEVICE)
                
                pred_fine, pred_coarse, mask = model(img_fine, MASK_RATIO)
                loss, _, _ = criterion(
                    pred_fine, pred_coarse,
                    img_fine, img_coarse,
                    mask
                )
                val_loss += loss.item()
        
        avg_val = val_loss / len(val_dl)
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}: Train={avg_train:.4f}, Val={avg_val:.4f}, "
              f"LR={scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"   💾 Best model saved (Val: {avg_val:.4f})")
        
        # Save plots periodically
        if (epoch + 1) % 5 == 0:
            plot_training_curves(history, OUTPUT_DIR)
    
    # Final save
    plot_training_curves(history, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("✅ Pre-training Complete!")
    print(f"   Best Validation Loss: {best_val_loss:.4f}")
    print(f"   Model saved to: {BEST_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train()
