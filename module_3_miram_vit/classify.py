# classify.py - TUMOR DIAGNOSIS WITH HEATMAP VISUALIZATION
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
import numpy as np
from config import *
from models import MIRAM
from dataset import load_medical_image

# 1. DATASET
class TumorDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.root = Path(root_dir)
        self.hr_dir = self.root / "HR"
        self.mask_dir = self.root / "MASK"
        self.samples = []
        
        print("🔍 Auto-labeling data based on masks...")
        hr_files = sorted(list(self.hr_dir.glob("*")))
        
        for hr_path in hr_files:
            mask_names = [hr_path.name, f"{hr_path.stem}_mask{hr_path.suffix}"]
            mask_path = None
            for name in mask_names:
                if (self.mask_dir / name).exists():
                    mask_path = self.mask_dir / name
                    break
            
            if mask_path:
                mask = load_medical_image(mask_path)
                label = 1 if np.max(mask) > 0 else 0 
                self.samples.append((hr_path, label, mask_path)) # Store mask path for viz

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.samples[idx]
        img_np = load_medical_image(img_path)
        img_t = torch.from_numpy(img_np).unsqueeze(0)
        img_t = torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE)).squeeze(0)
        return img_t, torch.tensor(label, dtype=torch.float32), str(mask_path), str(img_path)

def get_class_dataloaders():
    full_ds = TumorDataset(DATASET_PATH)
    if len(full_ds) == 0: return None, None
    total = len(full_ds)
    train_sz = int(total * TRAIN_SPLIT)
    test_sz = total - train_sz
    train_ds, test_ds = random_split(full_ds, [train_sz, test_sz])
    return DataLoader(train_ds, BATCH_SIZE, True), DataLoader(test_ds, 1, False) # Test batch=1 for viz

# 2. CLASSIFIER MODEL
class MIRAMClassifier(nn.Module):
    def __init__(self, pretrained_path=BEST_MODEL_PATH):
        super().__init__()
        self.miram = MIRAM()
        if os.path.exists(pretrained_path):
            print(f"✅ Loaded Anatomy Knowledge from {pretrained_path}")
            self.miram.load_state_dict(torch.load(pretrained_path, map_location=DEVICE), strict=False)
        
        # Unfreeze last layer for attention tuning
        for param in self.miram.parameters(): param.requires_grad = False
        for param in self.miram.blocks[-1].parameters(): param.requires_grad = True
            
        self.head = nn.Sequential(
            nn.Linear(EMBED_DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x, return_attn=False):
        # Pass return_last_attention=True to get maps
        if return_attn:
            latent, _, _, attn = self.miram.forward_encoder(x, mask_ratio=0.0, return_last_attention=True)
            cls_token = latent[:, 0, :]
            return torch.sigmoid(self.head(cls_token)), attn
        else:
            latent, _, _ = self.miram.forward_encoder(x, mask_ratio=0.0)
            cls_token = latent[:, 0, :]
            return torch.sigmoid(self.head(cls_token))

# 3. VISUALIZATION FUNCTION
def visualize_tumor_attention(model, test_loader, num_samples=3):
    print("\n🖼️ Generating Tumor Attention Maps...")
    model.eval()
    count = 0
    
    for img, label, mask_path, img_path in test_loader:
        if label.item() == 1: # Only visualize Tumors
            img = img.to(DEVICE)
            
            # Get Prediction and Attention
            pred, attn = model(img, return_attn=True)
            
            # attn shape: [1, NumHeads, SeqLen, SeqLen]
            # We want attention of [CLS] token (index 0) to all other patches (indices 1:)
            # Average over all heads
# NEW (Flexible)
        if attn.dim() == 4:
            # Shape: [Batch, Heads, Tokens, Tokens]
            # We select Batch 0, Average over Heads (:), CLS token (0), All patches (1:)
          cls_attn = attn[0, :, 0, 1:].mean(dim=0)
        else:
            # Shape: [Batch, Tokens, Tokens] (Heads likely already averaged)
            # We select Batch 0, CLS token (0), All patches (1:)
            cls_attn = attn[0, 0, 1:]            
            # Reshape to grid
            grid_size = IMG_SIZE // PATCH_SIZE
            attn_map = cls_attn.reshape(grid_size, grid_size).detach().cpu().numpy()
            
            # Resize heatmap to image size
            attn_map = cv2.resize(attn_map, (IMG_SIZE, IMG_SIZE))
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min()) # Normalize
            
            # Get Original Image
            orig_img = img[0,0].cpu().numpy()
            
            # Get Ground Truth Mask
            gt_mask = load_medical_image(mask_path[0])
            gt_mask = cv2.resize(gt_mask, (IMG_SIZE, IMG_SIZE))
            
            # Plot
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(orig_img, cmap='gray')
            ax[0].set_title(f"MRI (Tumor Positive)\nConf: {pred.item():.2f}")
            
            ax[1].imshow(gt_mask, cmap='gray')
            ax[1].set_title("Ground Truth Mask")
            
            # Overlay Heatmap
            ax[2].imshow(orig_img, cmap='gray')
            ax[2].imshow(attn_map, cmap='jet', alpha=0.5) # Jet heatmap overlay
            ax[2].set_title("Model Attention (Localization)")
            
            save_p = os.path.join(OUTPUT_DIR, f'tumor_viz_{count}.png')
            plt.savefig(save_p)
            plt.close()
            print(f"   Saved visualization to {save_p}")
            
            count += 1
            if count >= num_samples: break

# 4. MAIN LOOP
def train_and_evaluate():
    print(f"🚀 STARTING DIAGNOSIS TRAINING | Device: {DEVICE}")
    train_loader, test_loader = get_class_dataloaders()
    if not train_loader: return

    model = MIRAMClassifier().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Lower LR for fine-tuning
    criterion = nn.BCELoss()
    
    # Train
    for epoch in range(10):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            imgs = batch[0].to(DEVICE)
    # We skip batch[1] because that is likely the mask
            labels = batch[1].to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
            
    torch.save(model.state_dict(), BEST_CLASSIFIER_PATH)
    
    # Generate Visualizations (The Requirement!)
    visualize_tumor_attention(model, test_loader)
    
    # Metrics
    print("\n📊 FINAL DIAGNOSTIC REPORT")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels, _, _ in test_loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            y_pred.extend((out > 0.5).float().cpu().numpy())
            y_true.extend(labels.numpy())
            
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    print(f"Accuracy: {acc*100:.2f}% | Precision: {prec:.2f} | Recall: {rec:.2f}")

if __name__ == "__main__":
    train_and_evaluate()