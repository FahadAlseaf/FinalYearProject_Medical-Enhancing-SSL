"""
losses.py - Custom Loss Functions for MRI Super-Resolution

This module implements specialized loss functions for training the
MIRAM super-resolution model on medical images.

Loss Functions:
    - PerceptualLoss: VGG19-based perceptual similarity
    - CharbonnierLoss: Robust L1-like loss (handles outliers)
    - EdgeLoss: Laplacian-based edge preservation
    - adversarial_loss: GAN training loss
    - psnr: Peak Signal-to-Noise Ratio metric
    - ssim: Structural Similarity Index metric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG19 features.
    
    Computes L1 loss between deep feature representations extracted
    from a pre-trained VGG19 network. This encourages outputs that
    are perceptually similar to the target, rather than just
    pixel-wise similar.
    
    The features are extracted from layer 35 (conv5_4), which captures
    high-level semantic information.
    
    Shape:
        - sr: Super-resolved image (N, 1, H, W) or (N, 3, H, W)
        - hr: High-resolution target (N, 1, H, W) or (N, 3, H, W)
        - output: Scalar loss value
        
    Note:
        Grayscale images are automatically converted to 3-channel
        for VGG compatibility.
        
    Reference:
        Johnson et al., "Perceptual Losses for Real-Time Style Transfer
        and Super-Resolution", ECCV 2016
    """
    
    def __init__(self):
        super().__init__()
        
        # Load pre-trained VGG19 features
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.features = nn.Sequential(*list(vgg.children())[:35])
        self.features.eval()
        
        # Freeze VGG weights
        for param in self.features.parameters():
            param.requires_grad = False
        
        # ImageNet normalization parameters
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # Convert grayscale to RGB for VGG
        if sr.size(1) == 1:
            sr = sr.repeat(1, 3, 1, 1)
            hr = hr.repeat(1, 3, 1, 1)
        
        # Normalize using ImageNet statistics
        sr_norm = (sr - self.mean) / self.std
        hr_norm = (hr - self.mean) / self.std
        
        # Extract features and compute loss
        sr_features = self.features(sr_norm)
        hr_features = self.features(hr_norm)
        
        return F.l1_loss(sr_features, hr_features)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (Smooth L1 variant).
    
    A differentiable variant of L1 loss that is less sensitive to
    outliers. The loss function is:
    
        L = sqrt((x - y)^2 + eps^2)
    
    This provides a smooth approximation to L1 loss near zero,
    which can improve gradient flow during training.
    
    Args:
        eps: Small constant for numerical stability (default: 1e-6)
        
    Shape:
        - x: Predicted image (N, C, H, W)
        - y: Target image (N, C, H, W)
        - mask: Optional spatial mask (N, 1, H, W)
        - output: Scalar loss value
        
    Reference:
        Charbonnier et al., "Two deterministic half-quadratic
        regularization algorithms for computed imaging", ICIP 1994
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Compute Charbonnier loss
        diff = x - y
        loss = torch.sqrt(diff ** 2 + self.eps ** 2)
        
        # Apply mask if provided
        if mask is not None:
            # Weight loss by mask (focus on brain region)
            weighted_loss = loss * mask
            return weighted_loss.sum() / (mask.sum() + 1e-8)
        
        return loss.mean()


class EdgeLoss(nn.Module):
    """
    Edge Preservation Loss using Laplacian filter.
    
    Computes L1 loss between edge maps extracted using a Laplacian
    kernel. This encourages the model to preserve sharp edges and
    fine anatomical details.
    
    The 3x3 Laplacian kernel:
        [[-1, -1, -1],
         [-1,  8, -1],
         [-1, -1, -1]]
    
    Shape:
        - sr: Super-resolved image (N, 1, H, W)
        - hr: High-resolution target (N, 1, H, W)
        - mask: Optional spatial mask (N, 1, H, W)
        - output: Scalar loss value
    """
    
    def __init__(self):
        super().__init__()
        
        # Define Laplacian edge detection kernel
        kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32)
        
        # Register as buffer (not a parameter)
        self.register_buffer(
            'kernel',
            kernel.view(1, 1, 3, 3)
        )
    
    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Extract edges using Laplacian convolution
        sr_edges = F.conv2d(sr, self.kernel, padding=1)
        hr_edges = F.conv2d(hr, self.kernel, padding=1)
        
        # Compute L1 loss on edges
        loss = F.l1_loss(sr_edges, hr_edges, reduction='none')
        
        # Apply mask if provided
        if mask is not None:
            weighted_loss = loss * mask
            return weighted_loss.sum() / (mask.sum() + 1e-8)
        
        return loss.mean()


def adversarial_loss(
    predictions: torch.Tensor,
    is_real: bool = True
) -> torch.Tensor:
    """
    Compute adversarial loss for GAN training.
    
    Uses Binary Cross Entropy with Logits for stable training.
    
    Args:
        predictions: Discriminator output logits
        is_real: True for real images, False for generated
        
    Returns:
        BCE loss value
    """
    target = torch.ones_like(predictions) if is_real else torch.zeros_like(predictions)
    return F.binary_cross_entropy_with_logits(predictions, target)


def psnr(
    x: torch.Tensor,
    y: torch.Tensor,
    max_val: float = 1.0
) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    PSNR measures the ratio between the maximum possible power of a
    signal and the power of corrupting noise. Higher is better.
    
    Args:
        x: Predicted image (N, C, H, W)
        y: Target image (N, C, H, W)
        max_val: Maximum pixel value (default: 1.0 for normalized images)
        
    Returns:
        PSNR value in dB (higher is better, typical range: 20-50 dB)
        
    Formula:
        PSNR = 20 * log10(MAX) - 10 * log10(MSE)
    """
    mse = F.mse_loss(x, y)
    
    # Avoid log(0)
    if mse < 1e-10:
        return torch.tensor(100.0, device=x.device)
    
    return 20 * torch.log10(torch.tensor(max_val, device=x.device)) - 10 * torch.log10(mse)


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    c1: float = 0.01,
    c2: float = 0.03
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM).
    
    SSIM measures the structural similarity between two images,
    considering luminance, contrast, and structure.
    
    Args:
        x: Predicted image (N, C, H, W)
        y: Target image (N, C, H, W)
        c1: Luminance stability constant
        c2: Contrast stability constant
        
    Returns:
        SSIM value (range: -1 to 1, higher is better)
        
    Note:
        This is a simplified global SSIM. For more accurate results,
        use sliding window implementation from skimage or pytorch-msssim.
        
    Reference:
        Wang et al., "Image Quality Assessment: From Error Visibility
        to Structural Similarity", IEEE TIP 2004
    """
    # Compute means
    mu_x = x.mean()
    mu_y = y.mean()
    
    # Compute variances and covariance
    sigma_x = x.std()
    sigma_y = y.std()
    sigma_xy = torch.mean((x - mu_x) * (y - mu_y))
    
    # Compute SSIM
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    
    return numerator / denominator


# ===========================================
# Testing
# ===========================================

if __name__ == "__main__":
    print("Testing loss functions...")
    
    # Create test tensors
    x = torch.rand(1, 1, 64, 64)
    y = torch.rand(1, 1, 64, 64)
    mask = torch.ones(1, 1, 64, 64)
    
    # Test each loss
    perceptual = PerceptualLoss()
    charbonnier = CharbonnierLoss()
    edge = EdgeLoss()
    
    print(f"Perceptual Loss: {perceptual(x, y).item():.4f}")
    print(f"Charbonnier Loss: {charbonnier(x, y).item():.4f}")
    print(f"Charbonnier Loss (masked): {charbonnier(x, y, mask).item():.4f}")
    print(f"Edge Loss: {edge(x, y).item():.4f}")
    print(f"Adversarial Loss (real): {adversarial_loss(x, True).item():.4f}")
    print(f"PSNR: {psnr(x, y).item():.2f} dB")
    print(f"SSIM: {ssim(x, y).item():.4f}")
    
    print("\n✅ All loss functions working!")
