# train.py - MIRAM TRAINING WITH METRIC PLOTTING
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json

from config import *
from dataset import get_dataloaders
from models import MIRAM
from losses import MIRAMLoss

def plot_training_curves(history, save_dir):
    """Generates and saves professional plots for the thesis"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.title('MIRAM Model Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300)
    plt.close()
    
    print(f"📊 Training plots saved to {save_dir}")

def train():
    print(f"🚀 STARTING MIRAM (ViT-MAE) TRAINING | Device: {DEVICE}")
    
    train_dl, val_dl, _ = get_dataloaders()
    if not train_dl: return

    model = MIRAM().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = MIRAMLoss()
    
    # History tracking
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        for img_fine, img_coarse in pbar:
            img_fine = img_fine.to(DEVICE)
            img_coarse = img_coarse.to(DEVICE)
            
            optimizer.zero_grad()
            pred_fine, pred_coarse, mask = model(img_fine, MASK_RATIO)
            loss, l_f, l_c = criterion(pred_fine, pred_coarse, img_fine, img_coarse, mask)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix_str(f"L:{loss.item():.4f}")
            
        avg_train = train_loss / len(train_dl)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img_fine, img_coarse in val_dl:
                img_fine = img_fine.to(DEVICE)
                img_coarse = img_coarse.to(DEVICE)
                pred_fine, pred_coarse, mask = model(img_fine, MASK_RATIO)
                loss, _, _ = criterion(pred_fine, pred_coarse, img_fine, img_coarse, mask)
                val_loss += loss.item()
        
        avg_val = val_loss / len(val_dl)
        
        # Record Stats
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        
        # Save Best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"   💾 Best Model Saved (Val: {avg_val:.4f})")
            
        # Save plots every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_training_curves(history, OUTPUT_DIR)

    print("✅ Pre-training Complete.")
    plot_training_curves(history, OUTPUT_DIR)

if __name__ == "__main__":
    train()