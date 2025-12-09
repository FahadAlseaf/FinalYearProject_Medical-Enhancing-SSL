import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.models import vgg16, VGG16_Weights
vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from tqdm import tqdm
import os
import random
import shutil
import zipfile
import numpy as np
from collections import Counter

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".heif", ".heic", ".webp")
import glob

def dataset_has_images(folder, extensions=SUPPORTED_EXTENSIONS, min_samples=1):
    """Check if a folder contains at least min_samples images (with any supported extension)."""
    count = 0
    for ext in extensions:
        count += len(glob.glob(os.path.join(folder, f"*{ext}")))
    return count >= min_samples

def safe_image_save(img, path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg", ".heif", ".heic", ".webp"]:
        if img.mode not in ['L', 'RGB']:
            img = img.convert('RGB')
    img.save(path)


# --------------------------
# Enhanced Loss Functions
# --------------------------
def psnr_val(output, target):
    """Calculate PSNR between output and target"""
    mse = F.mse_loss(output, target)
    return 10 * torch.log10(1 / (mse + 1e-8))


def ssim_index(img1, img2, window_size=11):
    """Enhanced SSIM calculation with better stability"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = F.avg_pool2d(img1, window_size, 1, window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, 1, window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, 1, window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, 1, window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def accuracy_metric(output, target, threshold=0.05):
    """Calculate pixel-wise accuracy - percentage of pixels within threshold"""
    diff = torch.abs(output - target)
    accurate_pixels = (diff < threshold).float().mean()
    return accurate_pixels * 100


# --------------------------
# FOCAL LOSS FOR IMBALANCED DATA
# --------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss - designed to handle class imbalance by down-weighting easy examples
    and focusing on hard examples

    Formula: FL(pt) = -α(1-pt)^γ * log(pt)
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.where(targets == 1, p, 1 - p)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss - more robust than L1/L2"""

    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return torch.mean(loss)


class LPIPSLoss(nn.Module):
    """Simplified LPIPS-style perceptual loss using VGG features"""
    def __init__(self, device=None):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
            self.slice1 = nn.Sequential(*list(vgg[:4]))
            self.slice2 = nn.Sequential(*list(vgg[4:9]))
            self.slice3 = nn.Sequential(*list(vgg[9:16]))
            self.slice4 = nn.Sequential(*list(vgg[16:23]))
            for param in self.parameters():
                param.requires_grad = False

            self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Move all slices to the correct device at model creation
            self.slice1 = self.slice1.to(self.device)
            self.slice2 = self.slice2.to(self.device)
            self.slice3 = self.slice3.to(self.device)
            self.slice4 = self.slice4.to(self.device)
            self.available = True
        except Exception as e:
            print("LPIPSLoss: Error loading VGG or moving to device:", e)
            self.available = False
            self.device = device if device else torch.device("cpu")

    def forward(self, pred, target):
        if not self.available:
            return F.mse_loss(pred, target)
        pred = pred.to(self.device)
        target = target.to(self.device)
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        pred = (pred - 0.5) / 0.5
        target = (target - 0.5) / 0.5

        pred_feat1 = self.slice1(pred)
        target_feat1 = self.slice1(target)
        pred_feat2 = self.slice2(pred_feat1)
        target_feat2 = self.slice2(target_feat1)
        pred_feat3 = self.slice3(pred_feat2)
        target_feat3 = self.slice3(target_feat2)
        pred_feat4 = self.slice4(pred_feat3)
        target_feat4 = self.slice4(target_feat3)
        loss1 = F.l1_loss(pred_feat1, target_feat1)
        loss2 = F.l1_loss(pred_feat2, target_feat2)
        loss3 = F.l1_loss(pred_feat3, target_feat3)
        loss4 = F.l1_loss(pred_feat4, target_feat4)
        return loss1 + loss2 + loss3 + loss4

class HybridLoss(nn.Module):
    def __init__(self, lambda_char=1.0, lambda_ssim=10.0, lambda_lpips=5.0, device=None):
        super().__init__()
        self.lambda_char = lambda_char
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.charbonnier = CharbonnierLoss()
        self.lpips = LPIPSLoss(device=device)


    def forward(self, output, target):
        char_loss = self.charbonnier(output, target)
        ssim_val = ssim_index(output, target)
        ssim_loss = 1 - ssim_val

        if self.lpips is not None:
            lpips_loss = self.lpips(output, target)
        else:
            lpips_loss = F.mse_loss(output, target)

        psnr_value = psnr_val(output, target)
        accuracy = accuracy_metric(output, target)

        total_loss = (self.lambda_char * char_loss +
                      self.lambda_ssim * ssim_loss +
                      self.lambda_lpips * lpips_loss)

        return total_loss, psnr_value, ssim_val, accuracy


# --------------------------
# MEDICAL IMAGE AUGMENTATION
# --------------------------
class MedicalImageAugmentation:
    """Medical imaging-specific augmentation strategies"""

    def __init__(self, p=0.5):
        self.p = p

    def _rotation_aug(self, img):
        """Rotate image by ±10 degrees"""
        if np.random.rand() < self.p:
            angle = np.random.uniform(-10, 10)
            return img.rotate(angle, expand=False, fillcolor='white')
        return img

    def _translation_aug(self, img):
        """Translate image randomly"""
        if np.random.rand() < self.p:
            img_array = np.array(img)
            shift_x = np.random.randint(-10, 10)
            shift_y = np.random.randint(-10, 10)
            img_array = np.roll(img_array, (shift_x, shift_y), axis=(0, 1))
            return Image.fromarray(img_array.astype('uint8'))
        return img

    def _scaling_aug(self, img):
        """Scale image by 90-110%"""
        if np.random.rand() < self.p:
            scale = np.random.uniform(0.9, 1.1)
            new_size = (int(256 * scale), int(256 * scale))
            img = img.resize(new_size, Image.BILINEAR)

            if img.size[0] > 256:
                left = (img.size[0] - 256) // 2
                top = (img.size[1] - 256) // 2
                img = img.crop((left, top, left + 256, top + 256))
            else:
                new_img = Image.new('L', (256, 256), color=255)
                offset_x = (256 - img.size[0]) // 2
                offset_y = (256 - img.size[1]) // 2
                new_img.paste(img, (offset_x, offset_y))
                img = new_img
        return img

    def _horizontal_flip_aug(self, img):
        """Horizontal flip"""
        if np.random.rand() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def _contrast_aug(self, img):
        """Enhance contrast"""
        if np.random.rand() < self.p:
            enhancer = ImageEnhance.Contrast(img)
            return enhancer.enhance(np.random.uniform(0.8, 1.2))
        return img

    def _brightness_aug(self, img):
        """Adjust brightness"""
        if np.random.rand() < self.p:
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(np.random.uniform(0.8, 1.2))
        return img

    def __call__(self, img):
        """Apply augmentation pipeline"""
        img = self._rotation_aug(img)
        img = self._translation_aug(img)
        img = self._scaling_aug(img)
        img = self._horizontal_flip_aug(img)
        img = self._contrast_aug(img)
        img = self._brightness_aug(img)
        return img


# --------------------------
# ATTENTION MECHANISMS
# --------------------------
class ChannelAttention(nn.Module):
    """Channel Attention for feature recalibration"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial Attention for focusing on important regions"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention


# --------------------------
# ENHANCED MIRAM WITH ATTENTION
# --------------------------
class EnhancedMIRAM(nn.Module):
    def __init__(self, channels=1, features=64):
        super().__init__()
        self.head = nn.Conv2d(channels, features, 3, 1, 1)

        self.conv1 = nn.Conv2d(features, features, 3, 1, 1)
        self.conv2 = nn.Conv2d(features, features, 5, 1, 2)
        self.conv3 = nn.Conv2d(features, features, 7, 1, 3)

        self.channel_att = ChannelAttention(features * 3)
        self.spatial_att = SpatialAttention()

        self.fusion = nn.Conv2d(features * 3, features, 1, 1, 0)
        self.tail = nn.Conv2d(features, channels, 3, 1, 1)

        self.act = nn.GELU()

    def forward(self, x):
        feat = self.act(self.head(x))

        f1 = self.act(self.conv1(feat))
        f2 = self.act(self.conv2(feat))
        f3 = self.act(self.conv3(feat))

        multi_scale = torch.cat([f1, f2, f3], dim=1)

        multi_scale = self.channel_att(multi_scale)
        multi_scale = self.spatial_att(multi_scale)

        fused = self.act(self.fusion(multi_scale))

        output = self.tail(fused) + x

        return torch.clamp(output, 0, 1)


# --------------------------
# ENHANCED RESIDUAL BLOCK
# --------------------------
class AttentionResidualBlock(nn.Module):
    def __init__(self, in_features):
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

    def forward(self, x):
        residual = self.block(x)
        residual = self.channel_att(residual)
        residual = self.spatial_att(residual)
        return x + residual * 0.2


# --------------------------
# ENHANCED GENERATOR
# --------------------------
class EnhancedGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(1, 64, 9, 1, 4), nn.PReLU())

        self.res = nn.Sequential(*[AttentionResidualBlock(64) for _ in range(16)])

        self.post_res = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )

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

        self.tail = nn.Conv2d(64, 1, 9, 1, 4)

    def forward(self, x):
        initial = self.head(x)
        trunk = self.res(initial)
        trunk = self.post_res(trunk)
        feat = initial + trunk
        upsampled = self.upsample(feat)
        return torch.clamp(self.tail(upsampled), 0, 1)


# --------------------------
# ENHANCED DISCRIMINATOR
# --------------------------
class EnhancedDiscriminator(nn.Module):
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

    def forward(self, x):
        return self.net(x)


# --------------------------
# DATASET WITH AUGMENTATION
# --------------------------
class AugmentedImageFolderDataset(Dataset):
    def __init__(self, folder, augmentation=True, augmentation_prob=0.5):
        self.paths = [
            os.path.join(folder, p)
            for p in os.listdir(folder)
            if p.lower().endswith(SUPPORTED_EXTENSIONS)
        ]

        self.base_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        if augmentation:
            self.augmenter = MedicalImageAugmentation(p=augmentation_prob)
        else:
            self.augmenter = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        if self.augmenter is not None:
            img = self.augmenter(img)
        tensor = self.base_transform(img)
        return tensor, 0

# --------------------------
# WEIGHTED DATALOADER FOR IMBALANCED DATA
# --------------------------
def create_weighted_dataloader(dataset, batch_size=8, num_workers=2):
    """Create DataLoader with weighted sampling for imbalanced data"""

    class_counts = {}
    for idx in range(len(dataset)):
        img, _ = dataset[idx]
        mean_val = img.mean().item()
        class_label = 0 if mean_val > 0.5 else 1
        class_counts[class_label] = class_counts.get(class_label, 0) + 1

    total = len(dataset)
    class_weights = {
        cls: total / (len(class_counts) * count)
        for cls, count in class_counts.items()
    }

    print(f"\n📊 Class Distribution:")
    for cls, count in class_counts.items():
        print(f"   Class {cls}: {count} samples (weight: {class_weights[cls]:.4f})")

    sample_weights = []
    for idx in range(len(dataset)):
        img, _ = dataset[idx]
        mean_val = img.mean().item()
        class_label = 0 if mean_val > 0.5 else 1
        sample_weights.append(class_weights[class_label])

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )


# --------------------------
# OVERSAMPLE MINORITY CLASS
# --------------------------
def oversample_minority_class(dataset_folder, target_folder, oversample_ratio=2.0):
    os.makedirs(target_folder, exist_ok=True)
    augmenter = MedicalImageAugmentation(p=0.8)
    print(f"\n🔄 Oversampling minority class with ratio {oversample_ratio}x...")

    image_files = [f for f in os.listdir(dataset_folder) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    for img_file in image_files:
        src = os.path.join(dataset_folder, img_file)
        dst = os.path.join(target_folder, img_file)
        with Image.open(src) as img:
            safe_image_save(img, dst)

    minority_count = 0
    for img_file in image_files:
        img_path = os.path.join(dataset_folder, img_file)
        img = Image.open(img_path).convert("L")
        if np.array(img).mean() < 128:
            for aug_idx in range(int(oversample_ratio)):
                augmented = augmenter(img)
                base_name = os.path.splitext(img_file)[0]
                ext = os.path.splitext(img_file)[1].lower()
                new_filename = f"{base_name}_aug_{aug_idx}{ext}"
                save_path = os.path.join(target_folder, new_filename)
                safe_image_save(augmented, save_path)
                minority_count += 1

    print(f"✅ Oversampling complete!")
    print(f"   Original images: {len(image_files)}")
    print(f"   Augmented minority samples: {minority_count}")
    print(f"   Total images: {len(image_files) + minority_count}")



# --------------------------
# Zip extraction & train/test split
# --------------------------
def prepare_dataset_from_zip(zip_path, base_folder="dataset"):
    if not os.path.exists(zip_path):
        print("❌ Zip file does not exist.")
        return None, None

    temp_folder = os.path.join(base_folder, "extracted")
    os.makedirs(temp_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_folder)
    print(f"✅ Zip extracted to {temp_folder}")

    all_images = []
    for root, _, files in os.walk(temp_folder):
        for f in files:
            if f.lower().endswith(SUPPORTED_EXTENSIONS):
                all_images.append(os.path.join(root, f))
    if not all_images:
        print("❌ No images found in the zip.")
        return None, None

    random.shuffle(all_images)
    split_idx = int(0.8 * len(all_images))
    train_images = all_images[:split_idx]
    test_images = all_images[split_idx:]

    train_folder = os.path.join(base_folder, "train")
    test_folder = os.path.join(base_folder, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for img_path in train_images:
        shutil.move(img_path, os.path.join(train_folder, os.path.basename(img_path)))
    for img_path in test_images:
        shutil.move(img_path, os.path.join(test_folder, os.path.basename(img_path)))
    print(f"✅ Dataset prepared: {len(train_images)} train, {len(test_images)} test")
    return train_folder, test_folder


# --------------------------
# ENHANCED TRAINING FUNCTION WITH IMBALANCE HANDLING
# --------------------------
def train_miram_srgan(data_path, max_epochs=100, batch_size=8, patience_epochs=10,
                      min_improvement=0.002, use_gradient_penalty=True,
                      handle_imbalance=True, oversample_ratio=2.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HybridLoss(lambda_char=1.0, lambda_ssim=10.0, lambda_lpips=5.0, device=device)
    print(f"🔧 Using device: {device}")

    # Handle imbalanced data
    if handle_imbalance:
        print("\n" + "=" * 60)
        print("⚙️  SETTING UP IMBALANCED DATA HANDLING")
        print("=" * 60)

        # Oversample minority class
        balanced_path = os.path.join(os.path.dirname(data_path), "train_balanced")
        oversample_minority_class(data_path, balanced_path, oversample_ratio)
        data_path = balanced_path

    # Create dataset with augmentation
    dataset = AugmentedImageFolderDataset(
        data_path,
        augmentation=True,
        augmentation_prob=0.6
    )

    # Create weighted dataloader
    if handle_imbalance:
        print("\n📊 Using Weighted Random Sampler for class imbalance...")
        loader = create_weighted_dataloader(dataset, batch_size=batch_size)
    else:
        print("\n📊 Using standard DataLoader")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True)

    # Initialize enhanced models
    miram = EnhancedMIRAM().to(device)
    gen = EnhancedGenerator().to(device)
    disc = EnhancedDiscriminator().to(device)

    # Advanced loss functions
    hybrid_loss = HybridLoss(lambda_char=1.0, lambda_ssim=10.0, lambda_lpips=5.0)
    focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0) if handle_imbalance else None
    adversarial_loss = nn.BCEWithLogitsLoss()

    # Optimizers with weight decay
    optimizer_G = optim.AdamW(
        list(miram.parameters()) + list(gen.parameters()),
        lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4
    )
    optimizer_D = optim.AdamW(
        disc.parameters(),
        lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4
    )

    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G, mode='max', factor=0.5, patience=5
    )
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_D, mode='min', factor=0.5, patience=5
    )

    os.makedirs("training_results", exist_ok=True)

    best_psnr, best_ssim, best_accuracy = 0, 0, 0
    no_improve_epochs = 0

    print("\n🚀 Starting Enhanced Training with Imbalance Handling...\n")

    for epoch in range(max_epochs):
        epoch_psnr, epoch_ssim, epoch_accuracy = 0, 0, 0
        epoch_loss_G, epoch_loss_D = 0, 0
        count = 0

        progress_bar = tqdm(loader, desc=f"Epoch [{epoch + 1}/{max_epochs}]", ncols=120)

        for imgs, _ in progress_bar:
            imgs = imgs.to(device)
            batch_size_actual = imgs.size(0)

            # Create low-resolution images
            low_res = F.interpolate(imgs, scale_factor=0.25, mode="bicubic", align_corners=False)
            low_res = low_res + torch.randn_like(low_res) * 0.01
            low_res = torch.clamp(low_res, 0, 1)

            valid = torch.ones(batch_size_actual, 1, device=device)
            fake = torch.zeros(batch_size_actual, 1, device=device)

            # ===========================
            # Train Generator + MIRAM
            # ===========================
            optimizer_G.zero_grad()

            restored = miram(low_res)
            sr_imgs = gen(restored)

            # Hybrid loss calculation
            loss_hybrid, psnr_value, ssim_value, accuracy = hybrid_loss(sr_imgs, imgs)

            # Adversarial loss
            loss_gan = adversarial_loss(disc(sr_imgs), valid)

            # Total generator loss
            total_G = loss_hybrid + 5e-3 * loss_gan
            total_G.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(miram.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)

            optimizer_G.step()

            # ===========================
            # Train Discriminator
            # ===========================
            optimizer_D.zero_grad()

            loss_real = adversarial_loss(disc(imgs), valid)
            loss_fake = adversarial_loss(disc(sr_imgs.detach()), fake)
            loss_D = (loss_real + loss_fake) / 2

            # Gradient Penalty
            if use_gradient_penalty and count % 5 == 0:
                alpha = torch.rand(batch_size_actual, 1, 1, 1, device=device)
                interpolates = (alpha * imgs + (1 - alpha) * sr_imgs.detach()).requires_grad_(True)
                d_interpolates = disc(interpolates)

                gradients = torch.autograd.grad(
                    outputs=d_interpolates,
                    inputs=interpolates,
                    grad_outputs=torch.ones_like(d_interpolates),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]

                gradients = gradients.view(batch_size_actual, -1)
                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                loss_D = loss_D + 10 * gradient_penalty

            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
            optimizer_D.step()

            # Accumulate metrics
            epoch_psnr += psnr_value.item()
            epoch_ssim += ssim_value.item()
            epoch_accuracy += accuracy.item()
            epoch_loss_G += total_G.item()
            epoch_loss_D += loss_D.item()
            count += 1

            progress_bar.set_postfix({
                "PSNR": f"{psnr_value.item():.2f}",
                "SSIM": f"{ssim_value.item():.4f}",
                "Acc": f"{accuracy.item():.2f}%"
            })

        # Calculate averages
        avg_psnr = epoch_psnr / count
        avg_ssim = epoch_ssim / count
        avg_accuracy = epoch_accuracy / count
        avg_loss_G = epoch_loss_G / count
        avg_loss_D = epoch_loss_D / count

        print(f"\n📊 Epoch [{epoch + 1}/{max_epochs}]")
        print(f"   PSNR: {avg_psnr:.3f} dB | SSIM: {avg_ssim:.4f} | Accuracy: {avg_accuracy:.2f}%")
        print(f"   Loss_G: {avg_loss_G:.4f} | Loss_D: {avg_loss_D:.4f}\n")

        # Update learning rate
        scheduler_G.step(avg_psnr)
        scheduler_D.step(avg_loss_D)

        # Save sample images
        if (epoch + 1) % 5 == 0:
            save_image(sr_imgs[:16], f"training_results/epoch_{epoch + 1}.png",
                       nrow=4, normalize=True)

        # Early Stopping
        improvement = (avg_psnr > best_psnr + min_improvement or
                       avg_ssim > best_ssim + min_improvement or
                       avg_accuracy > best_accuracy + min_improvement)

        if improvement:
            best_psnr = max(avg_psnr, best_psnr)
            best_ssim = max(avg_ssim, best_ssim)
            best_accuracy = max(avg_accuracy, best_accuracy)
            no_improve_epochs = 0

            torch.save(miram.state_dict(), "miram_best.pth")
            torch.save(gen.state_dict(), "srgan_best.pth")
            torch.save(disc.state_dict(), "disc_best.pth")

            print("✅ Improvement detected — saved best models.")
            print(f"   Best PSNR: {best_psnr:.3f} | Best SSIM: {best_ssim:.4f} | Best Acc: {best_accuracy:.2f}%")
        else:
            no_improve_epochs += 1
            print(f"⚠️ No improvement for {no_improve_epochs}/{patience_epochs} epochs.")

        if no_improve_epochs >= patience_epochs:
            print("\n🛑 Early stopping — no improvement for too long.")
            break

    print("\n✅ Training finished!")
    print(f"📈 Final Best Metrics:")
    print(f"   PSNR: {best_psnr:.3f} dB")
    print(f"   SSIM: {best_ssim:.4f}")
    print(f"   Accuracy: {best_accuracy:.2f}%")


# --------------------------
# ENHANCED ENHANCEMENT FUNCTION
# --------------------------
def enhance_miram_srgan(input_folder, output_folder, sharpen_factor=8,
                        contrast_factor=1.1, brightness_factor=1.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    # Load enhanced models
    miram = EnhancedMIRAM().to(device)
    gen = EnhancedGenerator().to(device)

    miram.load_state_dict(torch.load("miram_best.pth", map_location=device))
    gen.load_state_dict(torch.load("srgan_best.pth", map_location=device))

    miram.eval()
    gen.eval()

    os.makedirs(output_folder, exist_ok=True)
    print("\n🎨 Starting image enhancement...\n")

    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    for filename in tqdm(image_files, desc="Enhancing images"):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("L")
        tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        with torch.no_grad():
            restored = miram(tensor)
            sr_img = gen(restored)
        sr_img_pil = transforms.ToPILImage()(sr_img.squeeze(0).cpu())

        enhancer_sharp = ImageEnhance.Sharpness(sr_img_pil)
        sr_img_pil = enhancer_sharp.enhance(sharpen_factor)

        enhancer_contrast = ImageEnhance.Contrast(sr_img_pil)
        sr_img_pil = enhancer_contrast.enhance(contrast_factor)
        enhancer_brightness = ImageEnhance.Brightness(sr_img_pil)
        sr_img_pil = enhancer_brightness.enhance(brightness_factor)
        output_path = os.path.join(output_folder, filename)
        safe_image_save(sr_img_pil, output_path)

    print(f"\n✅ Enhancement complete! Processed {len(image_files)} images.")
    print(f"📁 Results saved to: {output_folder}")



# --------------------------
# INTERACTIVE USER MENU
# --------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 ENHANCED MIRAM + SRGAN Medical Image Enhancement System")
    print("=" * 60)
    print("\n✨ Features with Imbalance Handling:")
    print("   • Multi-scale feature extraction")
    print("   • Channel & Spatial Attention mechanisms")
    print("   • Hybrid loss (Charbonnier + SSIM + LPIPS)")
    print("   • Spectral Normalization for stable training")
    print("   • Enhanced metrics: PSNR, SSIM, and Accuracy")
    print("   • Focal Loss for imbalanced data")
    print("   • Weighted Random Sampling")
    print("   • Medical imaging-specific augmentation")
    print("   • Minority class oversampling")
    print("\n" + "=" * 60)
    print("\n1️⃣  Train a new model")
    print("2️⃣  Enhance medical images")
    print("\n" + "=" * 60)

    choice = input("\nEnter your choice (1 or 2): ").strip()

    if choice == "1":
        dataset_input = input("Enter the path to your zip, 'train' image folder, or a parent dataset folder: ").strip()
        train_folder, test_folder = None, None

        # Jump to training if train/test already exist and have images
        if os.path.isdir(dataset_input):
            candidate_train = os.path.join(dataset_input, "train")
            candidate_test = os.path.join(dataset_input, "test")
            if dataset_has_images(candidate_train) and dataset_has_images(candidate_test):
                train_folder, test_folder = candidate_train, candidate_test
                print(f"✅ Found existing 'train' and 'test' sets: {train_folder}, {test_folder}")
            elif dataset_has_images(dataset_input):
                train_folder = dataset_input
                print(f"✅ Found images directly under folder: {train_folder} (all will be used for training)")
            else:
                # Try extraction if a zipfile is mistakenly given as folder path
                print("⚠️ No images found in given folder, checking for zip extraction fallback...")
                train_folder, test_folder = prepare_dataset_from_zip(dataset_input)
        elif os.path.isfile(dataset_input) and dataset_input.lower().endswith(".zip"):
            train_folder, test_folder = prepare_dataset_from_zip(dataset_input)
        else:
            print("❌ Provided path is not valid. Please check it.")

        if train_folder:
            epochs = int(input("Enter number of training epochs (recommended: 100): ").strip() or "100")
            batch_size = int(input("Enter batch size (recommended: 8): ").strip() or "8")
            oversample = float(
                input("Enter oversampling ratio for minority class (recommended: 2.0): ").strip() or "2.0")
            handle_imbalance = input("Handle imbalanced dataset? (y/n): ").strip().lower() == 'y'
            print(f"\n🎯 Training Configuration:")
            print(f"   Epochs: {epochs}")
            print(f"   Batch Size: {batch_size}")
            print(f"   Oversampling Ratio: {oversample}")
            print(f"   Handle Imbalance: {handle_imbalance}")
            print(f"   Dataset: {train_folder}")

            train_miram_srgan(train_folder, max_epochs=epochs, batch_size=batch_size,
                              handle_imbalance=handle_imbalance, oversample_ratio=oversample)
        else:
            print("❌ No valid training dataset found. Check your path, folder, or zip file.")

    elif choice == "2":
        input_folder = input("Enter the folder containing images to enhance: ").strip()
        output_folder = input("Enter the folder to save enhanced images: ").strip()
        if not os.path.exists(input_folder):
            print("❌ Invalid input folder path.")
        else:
            use_defaults = input("\nUse default enhancement settings? (y/n): ").strip().lower()
            if use_defaults == 'n':
                sharpen = float(input("Sharpen factor (default 8): ").strip() or "8")
                contrast = float(input("Contrast factor (default 1.1): ").strip() or "1.1")
                brightness = float(input("Brightness factor (default 1.05): ").strip() or "1.05")
                enhance_miram_srgan(input_folder, output_folder, sharpen, contrast, brightness)
            else:
                enhance_miram_srgan(input_folder, output_folder)
    else:
        print("❌ Invalid choice. Please enter 1 or 2.")

