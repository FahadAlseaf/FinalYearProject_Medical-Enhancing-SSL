#
# train_baseline.py (Finalized for Kaggle)
# Phase 3: Baseline Training (From Scratch)
#
import os, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg19
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import timm

# --- CONFIGURATION ---
DATA_DIR = "/kaggle/input/fyp-medical-images/medical_images_dataset"
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# --- 1. Dataset ---
class OnFlySRDataset(Dataset):
    def __init__(self, root_dir, augment=True, scale=4):
        self.root = Path(root_dir)
        self.scale = scale
        self.crop_size_hr = 224
        self.crop_size_lr = self.crop_size_hr // scale
        self.augment = augment
        IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        self.image_paths = []
        for ext in IMG_EXTENSIONS:
            self.image_paths.extend(sorted(self.root.rglob(f'**/*{ext}')))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            hr_img = Image.open(self.image_paths[idx]).convert("RGB")
            w, h = hr_img.size
            if w < self.crop_size_hr or h < self.crop_size_hr:
                hr_img = hr_img.resize((max(w, self.crop_size_hr), max(h, self.crop_size_hr)))
            i, j, h, w = transforms.RandomCrop.get_params(hr_img, output_size=(self.crop_size_hr, self.crop_size_hr))
            hr_img = transforms.functional.crop(hr_img, i, j, h, w)
            lr_img = hr_img.resize((self.crop_size_lr, self.crop_size_lr), Image.BICUBIC)
            if self.augment and random.random() > 0.5:
                lr_img = transforms.functional.hflip(lr_img)
                hr_img = transforms.functional.hflip(hr_img)
            return self.normalize(self.to_tensor(lr_img)), self.normalize(self.to_tensor(hr_img))
        except: return self.__getitem__((idx + 1) % len(self))

# --- 2. Models ---
class ChannelAttention(nn.Module):
    def __init__(self, c, r=16):
        super().__init__(); self.pool=nn.AdaptiveAvgPool2d(1); self.fc1=nn.Conv2d(c,c//r,1); self.fc2=nn.Conv2d(c//r,c,1)
    def forward(self,x): y=self.pool(x); y=F.relu(self.fc1(y)); return x*torch.sigmoid(self.fc2(y))

class SpatialAttention(nn.Module):
    def __init__(self): super().__init__(); self.conv=nn.Conv2d(2,1,7,padding=3)
    def forward(self,x): y=torch.cat([torch.max(x,1,keepdim=True)[0],torch.mean(x,1,keepdim=True)],1); return x*torch.sigmoid(self.conv(y))

class MIRAMBlock(nn.Module):
    def __init__(self,c): super().__init__(); self.c1=nn.Conv2d(c,c,3,padding=1); self.c2=nn.Conv2d(c,c,3,padding=1); self.ca, self.sa = ChannelAttention(c), SpatialAttention()
    def forward(self,x): out=F.relu(self.c1(x)); out=self.c2(out); out=self.ca(out); out=self.sa(out); return out+x

class UpsampleBlock(nn.Module):
    def __init__(self,c,scale=2): super().__init__(); self.c=nn.Conv2d(c,c*scale**2,3,padding=1); self.ps=nn.PixelShuffle(scale); self.a=nn.PReLU()
    def forward(self,x): return self.a(self.ps(self.c(x)))

class GeneratorViT_SR(nn.Module):
    def __init__(self,in_c=3, embed_dim=768, decoder_dim=256, scale=4):
        super().__init__()
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        self.resize = transforms.Resize((224, 224), antialias=True)
        self.feature_conv = nn.Linear(embed_dim, decoder_dim)
        self.miram = MIRAMBlock(decoder_dim)
        self.tail = nn.Sequential(UpsampleBlock(decoder_dim, 2), UpsampleBlock(decoder_dim, 2), UpsampleBlock(decoder_dim, 2), UpsampleBlock(decoder_dim, 2), nn.Conv2d(decoder_dim, in_c, 9, padding=4))
    def forward(self, x):
        x_resized = self.resize(x)
        features = self.encoder.forward_features(x_resized)[:, 1:, :] 
        features = self.feature_conv(features)
        features = features.permute(0, 2, 1).reshape(x.shape[0], -1, 14, 14)
        features = self.miram(features)
        return self.tail(features)

class Discriminator(nn.Module):
    def __init__(self,in_c=3): super().__init__(); self.m=nn.Sequential(nn.Conv2d(in_c,64,3,1,1), nn.LeakyReLU(0.2), nn.Conv2d(64,64,3,2,1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2), nn.Conv2d(64,128,3,1,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2), nn.Conv2d(128,128,3,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2), nn.Conv2d(128,256,3,1,1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2), nn.Conv2d(256,256,3,2,1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2), nn.Conv2d(256,512,3,1,1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2), nn.Conv2d(512,512,3,2,1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2), nn.Conv2d(512,1,3,1,1))
    def forward(self,x): return self.m(x)

class VGGPerceptual(nn.Module):
    def __init__(self): super().__init__(); v=vgg19(weights="IMAGENET1K_V1").features; self.f=nn.Sequential(*list(v.children())[:35]).eval().requires_grad_(False)
    def forward(self,x,y): return F.l1_loss(self.f(x), self.f(y))

# --- 3. Main ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_ds = OnFlySRDataset(DATA_DIR)
    train_len = int(len(full_ds) * 0.9)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_len, len(full_ds)-train_len])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    print(f"Train images: {len(train_ds)}, Val images: {len(val_ds)}")

    gen = GeneratorViT_SR().to(device)
    disc = Discriminator().to(device)
    vgg = VGGPerceptual().to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE)
    opt_d = torch.optim.Adam(disc.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    # *** BASELINE: NO WEIGHT LOADING ***
    print("INFO: Running baseline training from scratch (no SSL weights).")

    best_psnr = 0.0
    for epoch in range(EPOCHS):
        gen.train(); disc.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for lr, hr in pbar:
            lr, hr = lr.to(device), hr.to(device)
            opt_d.zero_grad()
            with torch.cuda.amp.autocast():
                sr = gen(lr).detach()
                loss_d = (F.binary_cross_entropy_with_logits(disc(hr), torch.ones_like(disc(hr))) + F.binary_cross_entropy_with_logits(disc(sr), torch.zeros_like(disc(sr)))) * 0.5
            scaler.scale(loss_d).backward()
            scaler.step(opt_d)
            opt_g.zero_grad()
            with torch.cuda.amp.autocast():
                sr = gen(lr)
                loss_g = F.l1_loss(sr, hr) + 1e-3 * F.binary_cross_entropy_with_logits(disc(sr), torch.ones_like(disc(sr))) + 6e-3 * vgg(sr, hr)
            scaler.scale(loss_g).backward()
            scaler.step(opt_g)
            scaler.update()
            pbar.set_postfix(G_Loss=f"{loss_g.item():.3f}")

        gen.eval()
        psnr_sum = 0
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            with torch.no_grad():
                sr = gen(lr).clamp(-1, 1)
                mse = F.mse_loss(sr*0.5+0.5, hr*0.5+0.5).item()
                psnr_sum += 20 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else 100
        avg_psnr = psnr_sum / len(val_loader)
        print(f"Epoch {epoch+1} Val PSNR: {avg_psnr:.2f} dB")
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(gen.state_dict(), "best_generator_baseline.pth")

if __name__ == "__main__":
    main()