"""
visualize_results.py - Inference & Visual Comparison

This script loads a test image, runs it through both the baseline and
SSL-pretrained models, and generates a side-by-side comparison plot.

Usage:
    python visualize_results.py

    # Or specify a custom image:
    python visualize_results.py --image /path/to/test_image.png

Output:
    - result_comparison.png: Side-by-side comparison image
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import timm

# Import configuration
from config import (
    BEST_GENERATOR_PATH, BEST_GENERATOR_BASELINE_PATH,
    DATA_DIR, DEVICE, NORMALIZE_MEAN, NORMALIZE_STD
)


# === Model Architecture (must match train.py) ===

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)
    
    def forward(self, x):
        y = self.pool(x)
        y = F.relu(self.fc1(y))
        return x * torch.sigmoid(self.fc2(y))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
    
    def forward(self, x):
        y = torch.cat([
            torch.max(x, 1, keepdim=True)[0],
            torch.mean(x, 1, keepdim=True)
        ], 1)
        return x * torch.sigmoid(self.conv(y))


class MIRAMBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.ca(out)
        out = self.sa(out)
        return out + x


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int, scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * scale ** 2, 3, padding=1)
        self.ps = nn.PixelShuffle(scale)
        self.act = nn.PReLU()
    
    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class GeneratorViT_SR(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, decoder_dim=256, scale=4):
        super().__init__()
        self.encoder = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=0
        )
        self.resize = transforms.Resize((224, 224), antialias=True)
        self.feature_conv = nn.Linear(embed_dim, decoder_dim)
        self.miram = MIRAMBlock(decoder_dim)
        self.tail = nn.Sequential(
            UpsampleBlock(decoder_dim, 2),
            UpsampleBlock(decoder_dim, 2),
            UpsampleBlock(decoder_dim, 2),
            UpsampleBlock(decoder_dim, 2),
            nn.Conv2d(decoder_dim, in_channels, 9, padding=4)
        )

    def forward(self, x):
        x_resized = self.resize(x)
        features = self.encoder.forward_features(x_resized)[:, 1:, :]
        features = self.feature_conv(features)
        features = features.permute(0, 2, 1).reshape(x.shape[0], -1, 14, 14)
        features = self.miram(features)
        return self.tail(features)


def load_model(model_path: str) -> GeneratorViT_SR:
    """Load a trained model from checkpoint."""
    print(f"Loading model from: {model_path}")
    
    model = GeneratorViT_SR().to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def find_test_image() -> str:
    """Find a test image from the data directory."""
    from pathlib import Path
    
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    data_path = Path(DATA_DIR)
    
    for ext in extensions:
        images = list(data_path.rglob(f'*{ext}'))
        if images:
            return str(images[0])
    
    raise FileNotFoundError(f"No images found in {DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Visualize SR results")
    parser.add_argument(
        '--image', '-i',
        type=str,
        default=None,
        help='Path to test image (default: auto-find from DATA_DIR)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='result_comparison.png',
        help='Output path for comparison image'
    )
    args = parser.parse_args()
    
    # Find or validate image path
    if args.image:
        image_path = args.image
    else:
        image_path = find_test_image()
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"Test image: {image_path}")
    
    # Check model files exist
    if not os.path.exists(BEST_GENERATOR_PATH):
        raise FileNotFoundError(
            f"SSL model not found: {BEST_GENERATOR_PATH}\n"
            f"Run train.py first."
        )
    
    if not os.path.exists(BEST_GENERATOR_BASELINE_PATH):
        raise FileNotFoundError(
            f"Baseline model not found: {BEST_GENERATOR_BASELINE_PATH}\n"
            f"Run train_baseline.py first."
        )
    
    # Load and prepare image
    hr_img = Image.open(image_path).convert("RGB")
    hr_img = hr_img.resize((224, 224), Image.BICUBIC)
    
    # Create LR image (4x downsampling)
    lr_img = hr_img.resize((56, 56), Image.BICUBIC)
    
    # Prepare tensor
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    lr_tensor = normalize(to_tensor(lr_img)).unsqueeze(0).to(DEVICE)
    
    # Load models
    model_baseline = load_model(BEST_GENERATOR_BASELINE_PATH)
    model_ssl = load_model(BEST_GENERATOR_PATH)
    
    # Run inference
    print("Generating super-resolved images...")
    
    with torch.no_grad():
        # Baseline output
        out_baseline = model_baseline(lr_tensor)
        out_baseline = out_baseline.clamp(-1, 1).cpu().squeeze()
        out_baseline = (out_baseline * 0.5 + 0.5).permute(1, 2, 0).numpy()
        
        # SSL output
        out_ssl = model_ssl(lr_tensor)
        out_ssl = out_ssl.clamp(-1, 1).cpu().squeeze()
        out_ssl = (out_ssl * 0.5 + 0.5).permute(1, 2, 0).numpy()
    
    # Create comparison plot
    print("Creating comparison plot...")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    titles = [
        "Low Resolution\n(56×56 input)",
        "Baseline\n(From Scratch)",
        "Ours\n(SSL Pre-trained)",
        "Ground Truth\n(High Resolution)"
    ]
    
    # Resize LR for display
    lr_display = lr_img.resize((224, 224), Image.NEAREST)
    images = [lr_display, out_baseline, out_ssl, hr_img]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Comparison saved to: {args.output}")
    
    # Try to display if running interactively
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    main()
