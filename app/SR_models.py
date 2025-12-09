"""
SR_models.py - Enhanced MIRAM + SRGAN Models for General Image Enhancement

This module contains the EnhancedMIRAM and EnhancedGenerator models used
by the GUI application for two-stage image enhancement:
    Stage 1: EnhancedMIRAM - Multi-scale feature extraction with attention
    Stage 2: EnhancedGenerator - SRGAN-style super-resolution

These models differ from the ViT-based models in module_1 and are optimized
for general medical image enhancement rather than SSL pre-training.

Usage:
    from SR_models import EnhancedMIRAM, EnhancedGenerator
    
    miram = EnhancedMIRAM().to(device)
    generator = EnhancedGenerator().to(device)
    
    # Two-stage enhancement
    restored = miram(input_tensor)
    sr_output = generator(restored)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================
# ATTENTION MECHANISMS
# ===========================================

class ChannelAttention(nn.Module):
    """
    Channel Attention with both Average and Max pooling.
    
    Combines information from both pooling operations for more
    robust channel-wise feature recalibration.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck (default: 16)
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention for focusing on important image regions.
    
    Combines average and max pooling along channel dimension
    to generate spatial attention weights.
    
    Args:
        kernel_size: Convolution kernel size (default: 7)
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention


# ===========================================
# ENHANCED MIRAM (STAGE 1)
# ===========================================

class EnhancedMIRAM(nn.Module):
    """
    Enhanced MIRAM with Multi-Scale Feature Extraction.
    
    This model uses multiple convolutional branches with different
    kernel sizes (3x3, 5x5, 7x7) to capture features at different
    scales, followed by attention-based fusion.
    
    This is Stage 1 of the two-stage enhancement pipeline.
    
    Args:
        channels: Number of input/output channels (default: 1 for grayscale)
        features: Number of intermediate feature channels (default: 64)
        
    Shape:
        - Input: (N, 1, H, W)
        - Output: (N, 1, H, W) - same size, enhanced features
    """
    
    def __init__(self, channels: int = 1, features: int = 64):
        super().__init__()
        
        # Initial feature extraction
        self.head = nn.Conv2d(channels, features, 3, 1, 1)
        
        # Multi-scale convolutions
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1)  # 3x3
        self.conv2 = nn.Conv2d(features, features, 5, 1, 2)  # 5x5
        self.conv3 = nn.Conv2d(features, features, 7, 1, 3)  # 7x7
        
        # Attention modules
        self.channel_att = ChannelAttention(features * 3)
        self.spatial_att = SpatialAttention()
        
        # Feature fusion
        self.fusion = nn.Conv2d(features * 3, features, 1, 1, 0)
        self.tail = nn.Conv2d(features, channels, 3, 1, 1)
        
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial features
        feat = self.act(self.head(x))
        
        # Multi-scale feature extraction
        f1 = self.act(self.conv1(feat))
        f2 = self.act(self.conv2(feat))
        f3 = self.act(self.conv3(feat))
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([f1, f2, f3], dim=1)
        
        # Apply attention
        multi_scale = self.channel_att(multi_scale)
        multi_scale = self.spatial_att(multi_scale)
        
        # Fuse features
        fused = self.act(self.fusion(multi_scale))
        
        # Residual output
        output = self.tail(fused) + x
        
        return torch.clamp(output, 0, 1)


# ===========================================
# ATTENTION RESIDUAL BLOCK
# ===========================================

class AttentionResidualBlock(nn.Module):
    """
    Residual Block with integrated Channel and Spatial Attention.
    
    Uses scaled residual learning (0.2x) for training stability.
    
    Args:
        in_features: Number of input/output channels
    """
    
    def __init__(self, in_features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features)
        )
        self.channel_att = ChannelAttention(in_features)
        self.spatial_att = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.block(x)
        residual = self.channel_att(residual)
        residual = self.spatial_att(residual)
        return x + residual * 0.2


# ===========================================
# ENHANCED GENERATOR (STAGE 2)
# ===========================================

class EnhancedGenerator(nn.Module):
    """
    Enhanced SRGAN-style Generator with Attention.
    
    This is Stage 2 of the two-stage enhancement pipeline.
    Performs 4x upsampling using PixelShuffle with attention blocks.
    
    Architecture:
        - 9x9 head convolution
        - 16 Attention Residual Blocks
        - 2x PixelShuffle upsampling (2x) = 4x total
        - 9x9 tail convolution
    
    Args:
        None (fixed architecture)
        
    Shape:
        - Input: (N, 1, H, W)
        - Output: (N, 1, H*4, W*4)
    """
    
    def __init__(self):
        super().__init__()
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(1, 64, 9, 1, 4),
            nn.PReLU()
        )
        
        # Deep residual blocks with attention
        self.res = nn.Sequential(*[AttentionResidualBlock(64) for _ in range(16)])
        
        # Post-residual convolution
        self.post_res = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        
        # Upsampling with attention (2x + 2x = 4x)
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            ChannelAttention(64),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            ChannelAttention(64)
        )
        
        # Tail
        self.tail = nn.Conv2d(64, 1, 9, 1, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        initial = self.head(x)
        trunk = self.res(initial)
        trunk = self.post_res(trunk)
        feat = initial + trunk
        upsampled = self.upsample(feat)
        return torch.clamp(self.tail(upsampled), 0, 1)


# ===========================================
# DISCRIMINATOR (FOR TRAINING)
# ===========================================

class EnhancedDiscriminator(nn.Module):
    """
    Enhanced Discriminator with Spectral Normalization.
    
    Uses spectral normalization for training stability.
    
    Shape:
        - Input: (N, 1, H, W)
        - Output: (N, 1) - real/fake score
    """
    
    def __init__(self):
        super().__init__()

        def conv_block(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 3, stride, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            )

        def conv_bn_block(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 3, stride, 1)),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.net = nn.Sequential(
            conv_block(1, 64, 1),
            conv_block(64, 64, 2),
            conv_bn_block(64, 128, 1),
            conv_bn_block(128, 128, 2),
            conv_bn_block(128, 256, 1),
            conv_bn_block(256, 256, 2),
            conv_bn_block(256, 512, 1),
            conv_bn_block(512, 512, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===========================================
# TESTING
# ===========================================

if __name__ == "__main__":
    print("Testing SR models...")
    
    # Test EnhancedMIRAM (Stage 1)
    miram = EnhancedMIRAM()
    x = torch.randn(1, 1, 64, 64)
    y1 = miram(x)
    print(f"EnhancedMIRAM: {x.shape} -> {y1.shape}")
    
    # Test EnhancedGenerator (Stage 2)
    gen = EnhancedGenerator()
    y2 = gen(y1)
    print(f"EnhancedGenerator: {y1.shape} -> {y2.shape}")
    
    # Test Discriminator
    disc = EnhancedDiscriminator()
    d_out = disc(y2)
    print(f"EnhancedDiscriminator: {y2.shape} -> {d_out.shape}")
    
    # Count parameters
    print(f"\nEnhancedMIRAM params: {sum(p.numel() for p in miram.parameters()):,}")
    print(f"EnhancedGenerator params: {sum(p.numel() for p in gen.parameters()):,}")
    print(f"EnhancedDiscriminator params: {sum(p.numel() for p in disc.parameters()):,}")
    
    print("\n✅ All tests passed!")
