"""
train.py - Training Loop for MIRAM MRI Super-Resolution

This script trains the MIRAM generator for 4x brain MRI super-resolution
using a combination of pixel, perceptual, edge, and adversarial losses.

Key Features:
    - Warmup strategy: Pixel-only training for initial epochs
    - TensorBoard logging for monitoring
    - Automatic checkpointing and early stopping
    - Masked loss computation for brain region focus

Usage:
    python train.py

Output:
    - best_generator.pth: Best model weights
    - checkpoint.pth: Latest checkpoint for resuming
    - logs/: TensorBoard logs
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import (
    DATASET_PATH, CHECKPOINT_PATH, BEST_MODEL_PATH, LOG_DIR,
    N_EPOCHS, LEARNING_RATE_G, LEARNING_RATE_D,
    WARMUP_PERCENTAGE, PATIENCE, DEVICE,
    LAMBDA_PIXEL, LAMBDA_VGG, LAMBDA_ADV, LAMBDA_EDGE
)
from models import GeneratorMIRAMSR, Discriminator
from losses import (
    PerceptualLoss, CharbonnierLoss, EdgeLoss,
    adversarial_loss, psnr
)
from dataset import get_dataloaders


def train():
    """Main training function."""
    
    print("=" * 60)
    print("MIRAM MRI Super-Resolution Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Epochs: {N_EPOCHS}")
    print(f"Warmup: {int(WARMUP_PERCENTAGE * 100)}% epochs")
    print("=" * 60)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(DATASET_PATH)
    
    if train_loader is None:
        print("Error: Could not create dataloaders. Check dataset path.")
        return
    
    # Initialize models
    generator = GeneratorMIRAMSR(in_channels=1, scale=4).to(DEVICE)
    discriminator = Discriminator(in_channels=1).to(DEVICE)
    
    # Initialize losses
    criterion_pixel = CharbonnierLoss().to(DEVICE)
    criterion_edge = EdgeLoss().to(DEVICE)
    criterion_perceptual = PerceptualLoss().to(DEVICE)
    
    # Optimizers
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE_G,
        betas=(0.9, 0.999)
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE_D,
        betas=(0.9, 0.999)
    )
    
    # Learning rate schedulers
    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g, mode='max', factor=0.5, patience=5
    )
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_d, mode='max', factor=0.5, patience=5
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if DEVICE == 'cuda' else None
    
    # TensorBoard writer
    writer = SummaryWriter(LOG_DIR)
    
    # Training state
    start_epoch = 0
    best_psnr = 0.0
    patience_counter = 0
    warmup_epochs = int(N_EPOCHS * WARMUP_PERCENTAGE)
    
    # Resume from checkpoint if exists
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', 0.0)
        print(f"Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.2f} dB")
    
    # Training loop
    for epoch in range(start_epoch, N_EPOCHS):
        generator.train()
        discriminator.train()
        
        # Determine if we're in warmup phase
        use_gan = epoch >= warmup_epochs
        phase = "GAN" if use_gan else "Warmup"
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{N_EPOCHS} [{phase}]"
        )
        
        for batch_idx, (lr, hr, mask) in enumerate(pbar):
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)
            mask = mask.to(DEVICE)
            
            # === Train Discriminator (only in GAN phase) ===
            if use_gan:
                optimizer_d.zero_grad()
                
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        sr = generator(lr).detach()
                        real_pred = discriminator(hr)
                        fake_pred = discriminator(sr)
                        
                        loss_d_real = adversarial_loss(real_pred, is_real=True)
                        loss_d_fake = adversarial_loss(fake_pred, is_real=False)
                        loss_d = (loss_d_real + loss_d_fake) / 2
                    
                    scaler.scale(loss_d).backward()
                    scaler.step(optimizer_d)
                else:
                    sr = generator(lr).detach()
                    real_pred = discriminator(hr)
                    fake_pred = discriminator(sr)
                    
                    loss_d_real = adversarial_loss(real_pred, is_real=True)
                    loss_d_fake = adversarial_loss(fake_pred, is_real=False)
                    loss_d = (loss_d_real + loss_d_fake) / 2
                    
                    loss_d.backward()
                    optimizer_d.step()
                
                epoch_d_loss += loss_d.item()
            
            # === Train Generator ===
            optimizer_g.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    sr = generator(lr)
                    
                    # Pixel loss (masked)
                    loss_pixel = criterion_pixel(sr, hr, mask)
                    
                    # Edge loss (masked)
                    loss_edge = criterion_edge(sr, hr, mask)
                    
                    # Combined generator loss
                    loss_g = LAMBDA_PIXEL * loss_pixel + LAMBDA_EDGE * loss_edge
                    
                    # Add perceptual and adversarial losses in GAN phase
                    if use_gan:
                        loss_perceptual = criterion_perceptual(sr, hr)
                        fake_pred = discriminator(sr)
                        loss_adv = adversarial_loss(fake_pred, is_real=True)
                        
                        loss_g += LAMBDA_VGG * loss_perceptual
                        loss_g += LAMBDA_ADV * loss_adv
                
                scaler.scale(loss_g).backward()
                scaler.step(optimizer_g)
                scaler.update()
            else:
                sr = generator(lr)
                
                loss_pixel = criterion_pixel(sr, hr, mask)
                loss_edge = criterion_edge(sr, hr, mask)
                loss_g = LAMBDA_PIXEL * loss_pixel + LAMBDA_EDGE * loss_edge
                
                if use_gan:
                    loss_perceptual = criterion_perceptual(sr, hr)
                    fake_pred = discriminator(sr)
                    loss_adv = adversarial_loss(fake_pred, is_real=True)
                    
                    loss_g += LAMBDA_VGG * loss_perceptual
                    loss_g += LAMBDA_ADV * loss_adv
                
                loss_g.backward()
                optimizer_g.step()
            
            epoch_g_loss += loss_g.item()
            
            # Update progress bar
            pbar.set_postfix({
                'G_Loss': f"{loss_g.item():.4f}",
                'D_Loss': f"{epoch_d_loss / (batch_idx + 1):.4f}" if use_gan else "N/A"
            })
        
        # Calculate average losses
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader) if use_gan else 0.0
        
        # === Validation ===
        generator.eval()
        val_psnr = 0.0
        
        with torch.no_grad():
            for lr, hr, mask in val_loader:
                lr = lr.to(DEVICE)
                hr = hr.to(DEVICE)
                
                sr = generator(lr)
                val_psnr += psnr(sr, hr).item()
        
        val_psnr /= len(val_loader)
        
        # Update schedulers
        scheduler_g.step(val_psnr)
        if use_gan:
            scheduler_d.step(val_psnr)
        
        # Logging
        print(f"Epoch {epoch+1}: G_Loss={avg_g_loss:.4f}, "
              f"D_Loss={avg_d_loss:.4f}, Val_PSNR={val_psnr:.2f} dB")
        
        writer.add_scalar('Loss/Generator', avg_g_loss, epoch)
        writer.add_scalar('Loss/Discriminator', avg_d_loss, epoch)
        writer.add_scalar('Metrics/PSNR', val_psnr, epoch)
        writer.add_scalar('LR/Generator', optimizer_g.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
            'best_psnr': best_psnr
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        
        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            patience_counter = 0
            torch.save(generator.state_dict(), BEST_MODEL_PATH)
            print(f"  → New best model saved! (PSNR: {best_psnr:.2f} dB)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {PATIENCE} epochs without improvement.")
                break
    
    # Cleanup
    writer.close()
    
    print("\n" + "=" * 60)
    print(f"✅ Training complete!")
    print(f"   Best PSNR: {best_psnr:.2f} dB")
    print(f"   Model saved to: {BEST_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train()
