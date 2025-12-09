#
# pretrain.py (Finalized for Kaggle)
# Phase 1: Self-Supervised Pre-training with MAE+ViT
# Features: On-the-fly loading, AMP (Mixed Precision)
#

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

# --- CONFIGURATION ---
# UPDATE THIS PATH to match your Kaggle Input folder name
DATA_DIR = "/kaggle/input/fyp-medical-images/medical_images_dataset"
SAVE_PATH = "pretrained_encoder.pth" # Saves to Kaggle Output directory
EPOCHS = 100 
BATCH_SIZE = 64 # Increased for T4 GPU (thanks to AMP)
LEARNING_RATE = 1.5e-4
IMG_SIZE = 224

# --- 1. Dataset ---
class UnlabeledMedicalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        self.image_paths = []
        # Recursively find all images in all subfolders
        for ext in IMG_EXTENSIONS:
            self.image_paths.extend(sorted(self.root.rglob(f'**/*{ext}')))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self))

# --- 2. MAE Logic ---
PATCH_SIZE = 16

def patchify(imgs):
    p = PATCH_SIZE
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x

class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, mask_ratio=0.75,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
                 mlp_ratio=4.):
        super().__init__()
        self.mask_ratio = mask_ratio
        
        # Encoder (ViT)
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size)**2 + 1, decoder_embed_dim), requires_grad=False)
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size**2) * 3, bias=True) 

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward(self, imgs):
        # Encoder
        x = self.encoder.patch_embed(imgs)
        x = x + self.encoder.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.encoder.blocks: x = blk(x)
        x = self.encoder.norm(x)
        
        # Decoder
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks: x = blk(x)
        x = self.decoder_norm(x)
        pred = self.decoder_pred(x)
        pred = pred[:, 1:, :]

        # Loss
        target = patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

# --- 3. Main ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Path {DATA_DIR} not found. Check your Kaggle Input path!")
        return

    dataset = UnlabeledMedicalDataset(root_dir=DATA_DIR, transform=transform)
    # Num_workers=2 is optimal for Kaggle
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    print(f"Found {len(dataset)} images.")

    model = MaskedAutoencoderViT(img_size=IMG_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler() # For Mixed Precision Speedup

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs in pbar:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            
            # Mixed Precision Training (Faster!)
            with torch.cuda.amp.autocast():
                loss = model(imgs)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.4f}")

    torch.save(model.encoder.state_dict(), SAVE_PATH)
    print(f"Done! Saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()