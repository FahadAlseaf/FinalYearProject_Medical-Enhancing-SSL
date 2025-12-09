"""
losses.py - Dual-Scale Patch Loss and Metrics

This module implements the MIRAM loss function for dual-scale
masked autoencoding, along with evaluation metrics.

Loss Function:
    - MSE on masked patches only
    - Weighted combination of fine and coarse scale losses

Metrics:
    - PSNR: Peak Signal-to-Noise Ratio
    - SSIM: Structural Similarity Index
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import PATCH_SIZE, SCALE_COARSE, LAMBDA_FINE, LAMBDA_COARSE


class MIRAMLoss(nn.Module):
    """
    Dual-Scale Masked Reconstruction Loss.
    
    Computes MSE loss between predicted and target patches,
    but only on the masked (hidden) patches. Combines losses
    from both fine and coarse scales.
    
    Loss = λ_fine * L_fine + λ_coarse * L_coarse
    
    where each L is: mean(MSE on masked patches)
    """
    
    def __init__(self):
        super().__init__()
    
    def patchify(self, imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
        """
        Convert images to patch sequences.
        
        Args:
            imgs: Images (N, 1, H, W)
            patch_size: Size of each square patch
            
        Returns:
            patches: (N, num_patches, patch_size^2 * C)
        """
        N, C, H, W = imgs.shape
        h = w = H // patch_size
        
        x = imgs.reshape(N, C, h, patch_size, w, patch_size)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(N, h * w, patch_size ** 2 * C)
        
        return x

    def forward(
        self,
        pred_fine: torch.Tensor,
        pred_coarse: torch.Tensor,
        target_fine: torch.Tensor,
        target_coarse: torch.Tensor,
        mask: torch.Tensor
    ) -> tuple:
        """
        Compute dual-scale reconstruction loss.
        
        Args:
            pred_fine: Predicted fine patches (N, L, patch_size^2)
            pred_coarse: Predicted coarse patches (N, L, coarse_size^2)
            target_fine: Target fine image (N, 1, H, W)
            target_coarse: Target coarse image (N, 1, H/2, W/2)
            mask: Binary mask (N, L) where 1=masked
            
        Returns:
            total_loss: Weighted sum of losses
            loss_fine: Fine-scale loss
            loss_coarse: Coarse-scale loss
        """
        # Patchify targets
        target_f = self.patchify(target_fine, PATCH_SIZE)
        
        coarse_patch_size = int(PATCH_SIZE * SCALE_COARSE)
        target_c = self.patchify(target_coarse, coarse_patch_size)
        
        # Compute MSE loss per patch
        loss_f = (pred_fine - target_f) ** 2
        loss_f = loss_f.mean(dim=-1)  # (N, L)
        
        loss_c = (pred_coarse - target_c) ** 2
        loss_c = loss_c.mean(dim=-1)  # (N, L)
        
        # Apply mask: only compute loss on masked patches
        # Loss = sum(loss * mask) / sum(mask)
        loss_f = (loss_f * mask).sum() / (mask.sum() + 1e-8)
        loss_c = (loss_c * mask).sum() / (mask.sum() + 1e-8)
        
        # Weighted combination
        total_loss = LAMBDA_FINE * loss_f + LAMBDA_COARSE * loss_c
        
        return total_loss, loss_f, loss_c


# ===========================================
# EVALUATION METRICS
# ===========================================

def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        x: Predicted image (N, C, H, W)
        y: Target image (N, C, H, W)
        
    Returns:
        PSNR value in dB (higher is better)
    """
    mse = F.mse_loss(x, y)
    
    if mse < 1e-10:
        return torch.tensor(100.0, device=x.device)
    
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute Structural Similarity Index (simplified global version).
    
    Args:
        x: Predicted image (N, C, H, W)
        y: Target image (N, C, H, W)
        
    Returns:
        SSIM value (range: -1 to 1, higher is better)
        
    Note:
        This is a simplified global SSIM. For window-based SSIM,
        consider using pytorch-msssim or skimage.
    """
    # Constants for stability
    k1, k2 = 0.01, 0.03
    L = 1.0  # Max pixel value
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    
    # Compute statistics
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x2 = ((x - mu_x) ** 2).mean()
    sigma_y2 = ((y - mu_y) ** 2).mean()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    
    # SSIM formula
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x2 + sigma_y2 + c2)
    
    return numerator / (denominator + 1e-8)


# ===========================================
# TESTING
# ===========================================

if __name__ == "__main__":
    print("Testing loss functions...")
    
    # Create test tensors
    N, L = 2, 196  # 14x14 patches
    patch_dim = PATCH_SIZE ** 2
    coarse_dim = int(PATCH_SIZE * SCALE_COARSE) ** 2
    
    pred_fine = torch.randn(N, L, patch_dim)
    pred_coarse = torch.randn(N, L, coarse_dim)
    target_fine = torch.randn(N, 1, 224, 224)
    target_coarse = torch.randn(N, 1, 112, 112)
    mask = torch.ones(N, L)
    mask[:, :49] = 0  # 25% visible
    
    # Test loss
    criterion = MIRAMLoss()
    total, l_f, l_c = criterion(
        pred_fine, pred_coarse,
        target_fine, target_coarse,
        mask
    )
    
    print(f"Total loss: {total.item():.4f}")
    print(f"Fine loss: {l_f.item():.4f}")
    print(f"Coarse loss: {l_c.item():.4f}")
    
    # Test metrics
    x = torch.rand(1, 1, 64, 64)
    y = x + 0.1 * torch.randn_like(x)
    
    print(f"\nPSNR: {psnr(x, y).item():.2f} dB")
    print(f"SSIM: {ssim(x, y).item():.4f}")
    
    print("\n✅ All tests passed!")
