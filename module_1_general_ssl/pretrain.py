"""
pretrain.py - Phase 1: Self-Supervised Pre-training with MAE+ViT

This script implements Masked Autoencoder (MAE) pre-training for learning
robust anatomical features from unlabeled medical images (X-Ray, CT).

The encoder learns to reconstruct randomly masked patches, forcing it to
understand the underlying structure of medical images without labels.

Usage:
    python pretrain.py

Output:
    - pretrained_encoder.pth: Saved encoder weights for fine-tuning
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import timm
from timm.models.vision_transformer import Block

# Import configuration
from config import (
    DATA_DIR, PRETRAINED_ENCODER_PATH,
    IMG_SIZE, PATCH_SIZE, MASK_RATIO,
    PRETRAIN_EPOCHS, PRETRAIN_BATCH_SIZE, PRETRAIN_LR,
    NUM_WORKERS, PIN_MEMORY, DEVICE,
    NORMALIZE_MEAN, NORMALIZE_STD
)


class UnlabeledMedicalDataset(Dataset):
    """
    Dataset for loading unlabeled medical images for self-supervised learning.
    
    Recursively searches for images in all subdirectories and applies
    standard augmentation suitable for medical imaging.
    
    Args:
        root_dir: Path to the root directory containing images
        transform: Optional torchvision transforms to apply
        
    Supported formats: PNG, JPG, JPEG, TIF, TIFF
    """
    
    SUPPORTED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    
    def __init__(self, root_dir: str, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        
        # Recursively find all images in all subfolders
        for ext in self.SUPPORTED_EXTENSIONS:
            self.image_paths.extend(sorted(self.root.rglob(f'*{ext}')))
            self.image_paths.extend(sorted(self.root.rglob(f'*{ext.upper()}')))
        
        # Remove duplicates
        self.image_paths = list(set(self.image_paths))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Warning: Failed to load {self.image_paths[idx]}: {e}")
            # Return next valid image
            return self.__getitem__((idx + 1) % len(self))


def patchify(imgs: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """
    Convert images to patch sequences for MAE processing.
    
    Args:
        imgs: Tensor of shape (N, C, H, W)
        patch_size: Size of each square patch
        
    Returns:
        Tensor of shape (N, num_patches, patch_size^2 * C)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x


class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder with Vision Transformer backbone.
    
    Implements the MAE pre-training strategy where random patches are masked
    and the model learns to reconstruct them. This forces the encoder to
    learn meaningful representations of the image structure.
    
    Args:
        img_size: Input image size (assumes square)
        patch_size: Size of image patches
        mask_ratio: Fraction of patches to mask (default: 0.75)
        embed_dim: Encoder embedding dimension
        depth: Number of transformer blocks in encoder
        num_heads: Number of attention heads in encoder
        decoder_embed_dim: Decoder embedding dimension
        decoder_depth: Number of transformer blocks in decoder
        decoder_num_heads: Number of attention heads in decoder
        mlp_ratio: MLP hidden dimension ratio
        
    Reference:
        He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 4,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Encoder: ViT backbone
        self.encoder = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=0
        )
        
        # Decoder components
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True)
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size ** 2 * 3,
            bias=True
        )
        
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform random masking on patch embeddings.
        
        Args:
            x: Patch embeddings of shape (N, L, D)
            mask_ratio: Fraction of patches to mask
            
        Returns:
            x_masked: Visible patches
            mask: Binary mask (1 = masked, 0 = visible)
            ids_restore: Indices to restore original order
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        # Random noise for shuffling
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep only visible patches
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )
        
        # Generate binary mask
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing MAE reconstruction loss.
        
        Args:
            imgs: Input images of shape (N, 3, H, W)
            
        Returns:
            loss: Mean squared error on masked patches
        """
        # Encoder forward pass
        x = self.encoder.patch_embed(imgs)
        x = x + self.encoder.pos_embed[:, 1:, :]
        
        # Apply masking
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        
        # Add CLS token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply transformer blocks
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        
        # Decoder forward pass
        x = self.decoder_embed(x)
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0],
            ids_restore.shape[1] + 1 - x.shape[1],
            1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # Add positional embeddings and decode
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        pred = self.decoder_pred(x)
        pred = pred[:, 1:, :]  # Remove CLS token
        
        # Compute loss on masked patches only
        target = patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean over patch dimensions
        loss = (loss * mask).sum() / mask.sum()  # Mean over masked patches
        
        return loss


def main():
    """Main training loop for MAE pre-training."""
    
    print("=" * 60)
    print("Phase 1: Self-Supervised Pre-training with MAE")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output path: {PRETRAINED_ENCODER_PATH}")
    print("=" * 60)
    
    # Check data directory exists
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}\n"
            f"Please update DATA_DIR in config.py or set the DATA_DIR environment variable."
        )
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    
    # Create dataset and dataloader
    dataset = UnlabeledMedicalDataset(root_dir=DATA_DIR, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=PRETRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    print(f"Found {len(dataset)} images.")
    
    # Initialize model
    model = MaskedAutoencoderViT(img_size=IMG_SIZE).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PRETRAIN_LR,
        weight_decay=0.05
    )
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if DEVICE == 'cuda' else None
    
    # Training loop
    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{PRETRAIN_EPOCHS}")
        
        for imgs in pbar:
            imgs = imgs.to(DEVICE)
            optimizer.zero_grad()
            
            if scaler is not None:
                # Mixed precision training for faster GPU processing
                with torch.amp.autocast('cuda'):
                    loss = model(imgs)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training for CPU
                loss = model(imgs)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    # Save encoder weights
    torch.save(model.encoder.state_dict(), PRETRAINED_ENCODER_PATH)
    print(f"\n✅ Pre-training complete! Encoder saved to: {PRETRAINED_ENCODER_PATH}")


if __name__ == "__main__":
    main()
