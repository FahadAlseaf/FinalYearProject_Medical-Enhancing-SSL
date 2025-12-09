"""
models.py - MIRAM Generator & Discriminator for MRI Super-Resolution

This module implements the neural network architectures for 4x brain MRI
super-resolution using the MIRAM (Masked Image Reconstruction Across
Multiple Scales) framework.

Key Components:
    - ChannelAttention: SE-style channel attention
    - SpatialAttention: Spatial attention using pooling
    - MIRAMBlock: Combined attention with residual learning
    - GeneratorMIRAMSR: Full super-resolution generator
    - Discriminator: PatchGAN discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (Squeeze-and-Excitation style).
    
    Learns adaptive channel-wise feature recalibration by explicitly
    modeling interdependencies between channels.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck (default: 16)
        
    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W)
        
    Reference:
        Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.fc(self.pool(x))
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Generates a spatial attention map by applying max and average pooling
    along the channel axis, followed by a convolution layer.
    
    Args:
        kernel_size: Convolution kernel size (default: 7)
        
    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W)
        
    Reference:
        Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        
        # Concatenate and apply convolution
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        
        return x * attention


class MIRAMBlock(nn.Module):
    """
    MIRAM (Masked Image Reconstruction Across Multiple Scales) Block.
    
    Combines channel and spatial attention mechanisms with residual
    connections for enhanced feature extraction in medical images.
    This block is designed to focus on anatomically relevant features
    while suppressing background noise.
    
    Args:
        channels: Number of input/output channels
        
    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W)
        
    Architecture:
        x -> Conv -> PReLU -> Conv -> ChannelAttn -> SpatialAttn -> (+x) -> out
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.relu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Feature extraction
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        
        # Attention mechanisms
        out = self.ca(out)
        out = self.sa(out)
        
        # Residual connection
        return out + residual


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with Batch Normalization.
    
    Args:
        channels: Number of input/output channels
        
    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W)
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    """
    Sub-pixel convolution upsampling block (PixelShuffle).
    
    Efficiently upsamples feature maps using learned convolutions
    followed by pixel shuffling.
    
    Args:
        channels: Number of input channels
        scale: Upsampling factor (default: 2)
        
    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H*scale, W*scale)
        
    Reference:
        Shi et al., "Real-Time Single Image and Video Super-Resolution
        Using an Efficient Sub-Pixel Convolutional Neural Network", CVPR 2016
    """
    
    def __init__(self, channels: int, scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * scale ** 2, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.activation = nn.PReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.pixel_shuffle(self.conv(x)))


class GeneratorMIRAMSR(nn.Module):
    """
    MIRAM-based Generator for Medical Image Super-Resolution.
    
    Performs 4x super-resolution on grayscale MRI images using a
    combination of residual blocks and MIRAM attention blocks.
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        base_channels: Base number of feature channels (default: 64)
        num_residual: Number of residual blocks (default: 8)
        scale: Super-resolution scale factor (default: 4)
        
    Shape:
        - Input: (N, 1, H, W)
        - Output: (N, 1, H*4, W*4)
        
    Architecture:
        Input -> Head -> [ResBlock + MIRAM]×N -> Skip -> Upsample -> Tail -> Output
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_residual: int = 8,
        scale: int = 4
    ):
        super().__init__()
        
        # Head: Initial feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 9, padding=4),
            nn.PReLU()
        )
        
        # Body: Residual blocks with periodic MIRAM attention
        body = []
        for i in range(num_residual):
            body.append(ResidualBlock(base_channels))
            # Add MIRAM block every 3 residual blocks
            if (i + 1) % 3 == 0:
                body.append(MIRAMBlock(base_channels))
        self.body = nn.Sequential(*body)
        
        # Upsampler: 4x = 2x + 2x
        self.upsampler = nn.Sequential(
            UpsampleBlock(base_channels, 2),
            UpsampleBlock(base_channels, 2)
        )
        
        # Tail: Final reconstruction
        self.tail = nn.Conv2d(base_channels, in_channels, 9, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial features
        head_features = self.head(x)
        
        # Deep feature extraction with skip connection
        body_features = self.body(head_features)
        features = head_features + body_features
        
        # Upsample and reconstruct
        upsampled = self.upsampler(features)
        output = self.tail(upsampled)
        
        # Clamp to valid range [0, 1]
        return torch.clamp(output, 0, 1)


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator for adversarial training.
    
    Classifies whether image patches are real or generated.
    Uses strided convolutions for progressive downsampling.
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        
    Shape:
        - Input: (N, 1, H, W)
        - Output: (N, 1, H/16, W/16) - patch predictions
        
    Reference:
        Isola et al., "Image-to-Image Translation with Conditional
        Adversarial Networks", CVPR 2017
    """
    
    def __init__(self, in_channels: int = 1):
        super().__init__()
        
        def block(
            in_ch: int,
            out_ch: int,
            stride: int = 1,
            use_norm: bool = True
        ) -> nn.Sequential:
            """Create a discriminator block."""
            layers = [nn.Conv2d(in_ch, out_ch, 3, stride, padding=1)]
            if use_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.net = nn.Sequential(
            block(in_channels, 64, use_norm=False),  # No norm for first layer
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
        return self.net(x)


# ===========================================
# Testing
# ===========================================

if __name__ == "__main__":
    # Test shapes
    print("Testing model architectures...")
    
    # Generator test
    gen = GeneratorMIRAMSR(in_channels=1, scale=4)
    x = torch.randn(1, 1, 64, 64)
    y = gen(x)
    print(f"Generator: {x.shape} -> {y.shape}")
    assert y.shape == (1, 1, 256, 256), "Generator output shape mismatch!"
    
    # Discriminator test
    disc = Discriminator(in_channels=1)
    d_out = disc(y)
    print(f"Discriminator: {y.shape} -> {d_out.shape}")
    
    # Count parameters
    gen_params = sum(p.numel() for p in gen.parameters())
    disc_params = sum(p.numel() for p in disc.parameters())
    print(f"\nGenerator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    
    print("\n✅ All tests passed!")
