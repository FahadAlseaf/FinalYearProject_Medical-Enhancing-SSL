"""
models.py - MIRAM Architecture with Attention Visualization

This module implements the MIRAM (Masked Image Reconstruction Across Multiple
Scales) architecture based on Vision Transformers (ViT) and Masked Autoencoders.

Key Components:
    - PatchEmbed: Splits image into non-overlapping patches
    - Block: Transformer block with attention weight access
    - MIRAM: Full encoder-decoder model with dual-scale reconstruction

The model supports:
    - Masked autoencoding for self-supervised pre-training
    - Attention visualization for interpretability
    - Dual-scale reconstruction (fine 224x224 + coarse 112x112)

Reference:
    He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
"""

import torch
import torch.nn as nn
import numpy as np
from config import (
    IMG_SIZE, PATCH_SIZE, IN_CHANNELS, EMBED_DIM,
    DEPTH, NUM_HEADS, MLP_RATIO,
    DECODER_EMBED_DIM, DECODER_DEPTH, DECODER_NUM_HEADS,
    SCALE_COARSE
)


class PatchEmbed(nn.Module):
    """
    Split image into non-overlapping patches and embed them.
    
    Uses a convolution with kernel_size=stride=patch_size to extract
    patch embeddings efficiently.
    
    Args:
        img_size: Input image size (assumes square)
        patch_size: Size of each patch (default: 16)
        in_chans: Number of input channels (default: 1)
        embed_dim: Embedding dimension (default: 768)
        
    Shape:
        - Input: (N, C, H, W)
        - Output: (N, num_patches, embed_dim)
        
    Example:
        >>> patch_embed = PatchEmbed(224, 16, 1, 384)
        >>> x = torch.randn(1, 1, 224, 224)
        >>> patches = patch_embed(x)  # (1, 196, 384)
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W) -> (N, embed_dim, H/P, W/P) -> (N, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Block(nn.Module):
    """
    Transformer Block with optional attention weight output.
    
    Implements standard transformer architecture with:
    - Multi-head self-attention
    - Layer normalization (pre-norm)
    - Feed-forward MLP with GELU activation
    
    Args:
        dim: Input/output dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio (default: 4.0)
        
    Shape:
        - Input: (N, L, dim)
        - Output: (N, L, dim) or ((N, L, dim), (N, num_heads, L, L)) if return_attention
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ):
        """
        Forward pass with optional attention weight output.
        
        Args:
            x: Input tensor (N, L, dim)
            return_attention: If True, also return attention weights
            
        Returns:
            x: Output tensor (N, L, dim)
            attn_weights: Optional attention weights (N, num_heads, L, L)
        """
        # Self-attention with pre-normalization
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(
            normed, normed, normed,
            need_weights=return_attention,
            average_attn_weights=False  # Keep per-head weights
        )
        x = x + attn_out
        
        # Feed-forward MLP
        x = x + self.mlp(self.norm2(x))
        
        if return_attention:
            return x, attn_weights
        return x


class MIRAM(nn.Module):
    """
    MIRAM: Masked Image Reconstruction Across Multiple Scales.
    
    A Vision Transformer-based Masked Autoencoder for self-supervised
    learning on medical images. Key features:
    
    1. Random patch masking (default 75%)
    2. Asymmetric encoder-decoder (full encoder, lightweight decoder)
    3. Dual-scale reconstruction (fine + coarse targets)
    4. Attention visualization for interpretability
    
    Architecture:
        Encoder: ViT-Small (12 blocks, 384 dim, 6 heads)
        Decoder: 4 blocks, 256 dim, 8 heads
        Outputs: Fine patches (16x16) + Coarse patches (8x8)
    
    Args:
        None (uses config.py settings)
        
    Shape:
        - Input: (N, 1, 224, 224)
        - Output: pred_fine (N, 196, 256), pred_coarse (N, 196, 64), mask (N, 196)
    """
    
    def __init__(self):
        super().__init__()
        
        # =====================
        # ENCODER
        # =====================
        self.patch_embed = PatchEmbed(
            IMG_SIZE, PATCH_SIZE, IN_CHANNELS, EMBED_DIM
        )
        num_patches = self.patch_embed.num_patches
        
        # Learnable tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, EMBED_DIM))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(EMBED_DIM, NUM_HEADS, MLP_RATIO)
            for _ in range(DEPTH)
        ])
        self.norm = nn.LayerNorm(EMBED_DIM)
        
        # =====================
        # DECODER
        # =====================
        self.decoder_embed = nn.Linear(EMBED_DIM, DECODER_EMBED_DIM, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, DECODER_EMBED_DIM))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, DECODER_EMBED_DIM)
        )
        
        self.decoder_blocks = nn.ModuleList([
            Block(DECODER_EMBED_DIM, DECODER_NUM_HEADS, MLP_RATIO)
            for _ in range(DECODER_DEPTH)
        ])
        self.decoder_norm = nn.LayerNorm(DECODER_EMBED_DIM)
        
        # =====================
        # PREDICTION HEADS
        # =====================
        # Fine-scale: Full resolution patches
        self.decoder_pred_fine = nn.Linear(
            DECODER_EMBED_DIM,
            PATCH_SIZE ** 2 * IN_CHANNELS,
            bias=True
        )
        
        # Coarse-scale: Downsampled patches
        coarse_patch_size = int(PATCH_SIZE * SCALE_COARSE)
        self.decoder_pred_coarse = nn.Linear(
            DECODER_EMBED_DIM,
            coarse_patch_size ** 2 * IN_CHANNELS,
            bias=True
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights."""
        # Positional embeddings
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=0.02)
        
        # Special tokens
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        # Prediction heads
        torch.nn.init.xavier_uniform_(self.decoder_pred_fine.weight)
        torch.nn.init.xavier_uniform_(self.decoder_pred_coarse.weight)

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float
    ) -> tuple:
        """
        Perform random masking on patch embeddings.
        
        Args:
            x: Patch embeddings (N, L, D)
            mask_ratio: Fraction of patches to mask
            
        Returns:
            x_masked: Visible patches only (N, L*(1-mask_ratio), D)
            mask: Binary mask (N, L) where 1=masked, 0=visible
            ids_restore: Indices to restore original order (N, L)
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
        
        # Generate binary mask (1=masked, 0=visible)
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        return_last_attention: bool = False
    ):
        """
        Forward pass through encoder with optional attention output.
        
        Args:
            x: Input images (N, 1, H, W)
            mask_ratio: Fraction of patches to mask
            return_last_attention: If True, return attention from last block
            
        Returns:
            latent: Encoded representations (N, 1+L_visible, D)
            mask: Binary mask (N, L)
            ids_restore: Restore indices (N, L)
            attn: Optional attention weights from last block
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add positional embedding (excluding CLS position)
        x = x + self.pos_embed[:, 1:, :]
        
        # Random masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Prepend CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply transformer blocks
        last_attn = None
        for i, blk in enumerate(self.blocks):
            if i == len(self.blocks) - 1 and return_last_attention:
                x, last_attn = blk(x, return_attention=True)
            else:
                x = blk(x)
        
        x = self.norm(x)
        
        if return_last_attention:
            return x, mask, ids_restore, last_attn
        return x, mask, ids_restore

    def forward_decoder(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor
    ) -> tuple:
        """
        Forward pass through decoder.
        
        Args:
            x: Encoded representations (N, 1+L_visible, D)
            ids_restore: Indices to restore original patch order
            
        Returns:
            pred_fine: Fine-scale patch predictions (N, L, patch_size^2)
            pred_coarse: Coarse-scale patch predictions (N, L, coarse_size^2)
        """
        # Project to decoder dimension
        x = self.decoder_embed(x)
        
        # Append mask tokens for masked positions
        mask_tokens = self.mask_token.repeat(
            x.shape[0],
            ids_restore.shape[1] + 1 - x.shape[1],
            1
        )
        
        # Unshuffle: restore original patch order
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # Add decoder positional embedding
        x = x + self.decoder_pos_embed
        
        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Remove CLS token
        x = x[:, 1:, :]
        
        # Predict patches at both scales
        pred_fine = self.decoder_pred_fine(x)
        pred_coarse = self.decoder_pred_coarse(x)
        
        return pred_fine, pred_coarse

    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = 0.75
    ) -> tuple:
        """
        Full forward pass: encode with masking, then decode.
        
        Args:
            imgs: Input images (N, 1, H, W)
            mask_ratio: Fraction of patches to mask (default: 0.75)
            
        Returns:
            pred_fine: Fine-scale predictions (N, L, patch_size^2)
            pred_coarse: Coarse-scale predictions (N, L, coarse_size^2)
            mask: Binary mask showing which patches were masked
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred_fine, pred_coarse = self.forward_decoder(latent, ids_restore)
        return pred_fine, pred_coarse, mask


# ===========================================
# UTILITY FUNCTIONS
# ===========================================

def unpatchify(x: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """
    Reconstruct image from patch predictions.
    
    Args:
        x: Patch predictions (N, L, patch_size^2 * C)
        patch_size: Size of each patch
        
    Returns:
        imgs: Reconstructed images (N, C, H, W)
    """
    N, L, D = x.shape
    h = w = int(L ** 0.5)
    c = D // (patch_size ** 2)
    
    x = x.reshape(N, h, w, patch_size, patch_size, c)
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(N, c, h * patch_size, w * patch_size)
    
    return imgs


# ===========================================
# TESTING
# ===========================================

if __name__ == "__main__":
    print("Testing MIRAM model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    model = MIRAM().to(device)
    
    # Test input
    x = torch.randn(2, 1, 224, 224).to(device)
    
    # Forward pass
    pred_fine, pred_coarse, mask = model(x, mask_ratio=0.75)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Pred fine shape: {pred_fine.shape}")
    print(f"Pred coarse shape: {pred_coarse.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Masked patches: {mask.sum().item()} / {mask.numel()}")
    
    # Test reconstruction
    recon = unpatchify(pred_fine, PATCH_SIZE)
    print(f"Reconstructed shape: {recon.shape}")
    
    # Test attention output
    latent, mask, ids, attn = model.forward_encoder(x, 0.0, return_last_attention=True)
    print(f"Attention shape: {attn.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {params:,}")
    
    print("\n✅ All tests passed!")
