"""
train.py - Phase 2: Fine-tuning with SSL Pre-trained Weights

This script fine-tunes the super-resolution model using the encoder
weights pre-trained with MAE in Phase 1. The pre-trained encoder
provides better initialization for learning anatomical features.

Usage:
    python train.py

Prerequisites:
    - Run pretrain.py first to generate pretrained_encoder.pth

Output:
    - best_generator.pth: Best model weights based on validation PSNR
"""

import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import timm

# Import configuration
from config import (
    DATA_DIR, PRETRAINED_ENCODER_PATH, BEST_GENERATOR_PATH,
    IMG_SIZE, EMBED_DIM, DECODER_DIM, SCALE_FACTOR,
    FINETUNE_EPOCHS, FINETUNE_BATCH_SIZE, FINETUNE_LR,
    NUM_WORKERS, PIN_MEMORY, VAL_SPLIT, DEVICE,
    NORMALIZE_MEAN, NORMALIZE_STD,
    LAMBDA_PIXEL, LAMBDA_ADV, LAMBDA_PERCEPTUAL
)


class OnFlySRDataset(Dataset):
    """
    Dataset for Super-Resolution with on-the-fly LR image generation.
    
    Generates Low-Resolution (LR) images dynamically during training by
    downsampling High-Resolution (HR) images. This saves disk space and
    allows flexible scaling factors.
    
    Args:
        root_dir: Path to directory containing HR images
        augment: Whether to apply data augmentation
        scale: Downsampling scale factor (default: 4)
    """
    
    SUPPORTED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    
    def __init__(self, root_dir: str, augment: bool = True, scale: int = 4):
        self.root = Path(root_dir)
        self.scale = scale
        self.crop_size_hr = 224
        self.crop_size_lr = self.crop_size_hr // scale
        self.augment = augment
        
        # Find all images
        self.image_paths = []
        for ext in self.SUPPORTED_EXTENSIONS:
            self.image_paths.extend(sorted(self.root.rglob(f'*{ext}')))
            self.image_paths.extend(sorted(self.root.rglob(f'*{ext.upper()}')))
        self.image_paths = list(set(self.image_paths))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            hr_img = Image.open(self.image_paths[idx]).convert("RGB")
            
            # Ensure image is at least crop_size
            w, h = hr_img.size
            if w < self.crop_size_hr or h < self.crop_size_hr:
                hr_img = hr_img.resize(
                    (max(w, self.crop_size_hr), max(h, self.crop_size_hr)),
                    Image.BICUBIC
                )
            
            # Random crop for HR
            i, j, h, w = transforms.RandomCrop.get_params(
                hr_img,
                output_size=(self.crop_size_hr, self.crop_size_hr)
            )
            hr_img = transforms.functional.crop(hr_img, i, j, h, w)
            
            # Create LR by downsampling
            lr_img = hr_img.resize(
                (self.crop_size_lr, self.crop_size_lr),
                Image.BICUBIC
            )
            
            # Data augmentation
            if self.augment and random.random() > 0.5:
                lr_img = transforms.functional.hflip(lr_img)
                hr_img = transforms.functional.hflip(hr_img)
            
            lr_tensor = self.normalize(self.to_tensor(lr_img))
            hr_tensor = self.normalize(self.to_tensor(hr_img))
            
            return lr_tensor, hr_tensor
            
        except Exception as e:
            print(f"Warning: Failed to load {self.image_paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (SE-style).
    
    Learns to weight channel importance using global average pooling
    followed by a bottleneck MLP.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Learns to weight spatial locations using max and average pooling
    across channels, followed by a convolutional layer.
    """
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = torch.sigmoid(self.conv(y))
        return x * y


class MIRAMBlock(nn.Module):
    """
    MIRAM (Masked Image Reconstruction Across Multiple Scales) Block.
    
    Combines channel and spatial attention mechanisms with residual
    learning for enhanced feature extraction in medical images.
    
    Args:
        channels: Number of input/output channels
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.ca(out)
        out = self.sa(out)
        return out + residual


class UpsampleBlock(nn.Module):
    """
    Sub-pixel convolution upsampling block.
    
    Uses PixelShuffle for efficient learned upsampling.
    
    Args:
        channels: Number of input channels
        scale: Upsampling factor
    """
    
    def __init__(self, channels: int, scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * scale ** 2, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.activation = nn.PReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.pixel_shuffle(self.conv(x)))


class GeneratorViT_SR(nn.Module):
    """
    Vision Transformer-based Super-Resolution Generator.
    
    Uses a ViT encoder (optionally pre-trained with MAE) followed by
    MIRAM attention blocks and sub-pixel upsampling for 4x SR.
    
    Args:
        in_channels: Number of input image channels
        embed_dim: ViT embedding dimension
        decoder_dim: Decoder feature dimension
        scale: Super-resolution scale factor
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 768,
        decoder_dim: int = 256,
        scale: int = 4
    ):
        super().__init__()
        
        # ViT Encoder
        self.encoder = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=0
        )
        
        # Resize input to ViT expected size
        self.resize = transforms.Resize((224, 224), antialias=True)
        
        # Feature projection
        self.feature_conv = nn.Linear(embed_dim, decoder_dim)
        
        # MIRAM attention block
        self.miram = MIRAMBlock(decoder_dim)
        
        # Upsampling tail (4x = 2x * 2x * 2x * 2x for 14->224)
        self.tail = nn.Sequential(
            UpsampleBlock(decoder_dim, 2),
            UpsampleBlock(decoder_dim, 2),
            UpsampleBlock(decoder_dim, 2),
            UpsampleBlock(decoder_dim, 2),
            nn.Conv2d(decoder_dim, in_channels, 9, padding=4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Resize input for ViT
        x_resized = self.resize(x)
        
        # Extract ViT features (exclude CLS token)
        features = self.encoder.forward_features(x_resized)[:, 1:, :]
        
        # Project features
        features = self.feature_conv(features)
        
        # Reshape to spatial format (14x14 patches)
        batch_size = x.shape[0]
        features = features.permute(0, 2, 1).reshape(batch_size, -1, 14, 14)
        
        # Apply MIRAM attention
        features = self.miram(features)
        
        # Upsample to output resolution
        return self.tail(features)


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator for adversarial training.
    
    Uses strided convolutions to classify whether patches are real or fake.
    
    Args:
        in_channels: Number of input image channels
        base_channels: Base number of feature channels
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        
        def block(in_ch, out_ch, stride=1, use_norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, 3, stride, padding=1)]
            if use_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.model = nn.Sequential(
            block(in_channels, 64, use_norm=False),
            block(64, 64, stride=2),
            block(64, 128),
            block(128, 128, stride=2),
            block(128, 256),
            block(256, 256, stride=2),
            block(256, 512),
            block(512, 512, stride=2),
            nn.Conv2d(512, 1, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.
    
    Computes L1 loss between feature maps extracted from a pre-trained
    VGG19 network, encouraging perceptually similar outputs.
    """
    
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.features = nn.Sequential(*list(vgg.children())[:35])
        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.features(sr), self.features(hr))


def compute_psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    # Denormalize from [-1, 1] to [0, 1]
    sr_denorm = sr * 0.5 + 0.5
    hr_denorm = hr * 0.5 + 0.5
    mse = F.mse_loss(sr_denorm, hr_denorm).item()
    if mse < 1e-10:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse))


def main():
    """Main training loop with SSL pre-trained weights."""
    
    print("=" * 60)
    print("Phase 2: Fine-tuning with SSL Pre-trained Weights")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Pre-trained weights: {PRETRAINED_ENCODER_PATH}")
    print("=" * 60)
    
    # Check data directory
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}\n"
            f"Please update DATA_DIR in config.py"
        )
    
    # Create datasets
    full_dataset = OnFlySRDataset(DATA_DIR)
    train_size = int(len(full_dataset) * (1 - VAL_SPLIT))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=FINETUNE_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize models
    generator = GeneratorViT_SR().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    vgg_loss = VGGPerceptualLoss().to(DEVICE)
    
    # Optimizers
    opt_g = torch.optim.Adam(generator.parameters(), lr=FINETUNE_LR)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=FINETUNE_LR)
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if DEVICE == 'cuda' else None
    
    # Load SSL pre-trained weights
    if os.path.exists(PRETRAINED_ENCODER_PATH):
        generator.encoder.load_state_dict(
            torch.load(PRETRAINED_ENCODER_PATH, map_location=DEVICE)
        )
        print("✅ Loaded SSL pre-trained encoder weights!")
    else:
        print("⚠️ Pre-trained weights not found. Training from scratch.")
        print(f"   Run pretrain.py first to generate: {PRETRAINED_ENCODER_PATH}")
    
    # Training loop
    best_psnr = 0.0
    
    for epoch in range(FINETUNE_EPOCHS):
        generator.train()
        discriminator.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{FINETUNE_EPOCHS}")
        
        for lr_img, hr_img in pbar:
            lr_img = lr_img.to(DEVICE)
            hr_img = hr_img.to(DEVICE)
            
            # === Train Discriminator ===
            opt_d.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    sr_img = generator(lr_img).detach()
                    real_pred = discriminator(hr_img)
                    fake_pred = discriminator(sr_img)
                    loss_d = 0.5 * (
                        F.binary_cross_entropy_with_logits(
                            real_pred, torch.ones_like(real_pred)
                        ) +
                        F.binary_cross_entropy_with_logits(
                            fake_pred, torch.zeros_like(fake_pred)
                        )
                    )
                scaler.scale(loss_d).backward()
                scaler.step(opt_d)
            else:
                sr_img = generator(lr_img).detach()
                real_pred = discriminator(hr_img)
                fake_pred = discriminator(sr_img)
                loss_d = 0.5 * (
                    F.binary_cross_entropy_with_logits(
                        real_pred, torch.ones_like(real_pred)
                    ) +
                    F.binary_cross_entropy_with_logits(
                        fake_pred, torch.zeros_like(fake_pred)
                    )
                )
                loss_d.backward()
                opt_d.step()
            
            # === Train Generator ===
            opt_g.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    sr_img = generator(lr_img)
                    fake_pred = discriminator(sr_img)
                    
                    # Combined loss
                    loss_pixel = F.l1_loss(sr_img, hr_img)
                    loss_adv = F.binary_cross_entropy_with_logits(
                        fake_pred, torch.ones_like(fake_pred)
                    )
                    loss_perceptual = vgg_loss(sr_img, hr_img)
                    
                    loss_g = (
                        LAMBDA_PIXEL * loss_pixel +
                        LAMBDA_ADV * loss_adv +
                        LAMBDA_PERCEPTUAL * loss_perceptual
                    )
                
                scaler.scale(loss_g).backward()
                scaler.step(opt_g)
                scaler.update()
            else:
                sr_img = generator(lr_img)
                fake_pred = discriminator(sr_img)
                
                loss_pixel = F.l1_loss(sr_img, hr_img)
                loss_adv = F.binary_cross_entropy_with_logits(
                    fake_pred, torch.ones_like(fake_pred)
                )
                loss_perceptual = vgg_loss(sr_img, hr_img)
                
                loss_g = (
                    LAMBDA_PIXEL * loss_pixel +
                    LAMBDA_ADV * loss_adv +
                    LAMBDA_PERCEPTUAL * loss_perceptual
                )
                loss_g.backward()
                opt_g.step()
            
            pbar.set_postfix(G_Loss=f"{loss_g.item():.4f}", D_Loss=f"{loss_d.item():.4f}")
        
        # === Validation ===
        generator.eval()
        total_psnr = 0.0
        
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img = lr_img.to(DEVICE)
                hr_img = hr_img.to(DEVICE)
                sr_img = generator(lr_img).clamp(-1, 1)
                total_psnr += compute_psnr(sr_img, hr_img)
        
        avg_psnr = total_psnr / len(val_loader)
        print(f"Epoch {epoch+1} - Validation PSNR: {avg_psnr:.2f} dB")
        
        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(generator.state_dict(), BEST_GENERATOR_PATH)
            print(f"  → New best model saved! (PSNR: {best_psnr:.2f} dB)")
    
    print(f"\n✅ Training complete! Best PSNR: {best_psnr:.2f} dB")
    print(f"   Model saved to: {BEST_GENERATOR_PATH}")


if __name__ == "__main__":
    main()
