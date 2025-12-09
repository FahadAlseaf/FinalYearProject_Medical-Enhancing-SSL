import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import timm

# --- CONFIGURATION ---
# 1. Path to ONE test image (Pick any image from your dataset)
IMAGE_PATH = "F:/FYP/medical_images_dataset/nih_xray/00000005_003.png" 

# 2. Paths to your downloaded models
BASELINE_PATH = "F:/FYP/best_generator_baseline.pth"
SSL_PATH = "F:/FYP/best_generator.pth"

# 3. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL ARCHITECTURE (Must match train.py exactly) ---
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

# --- FUNCTIONS ---
def load_model(path):
    print(f"Loading model from {path}...")
    model = GeneratorViT_SR().to(device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def main():
    # 1. Load Data
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Image not found at {IMAGE_PATH}")
        return

    # --- CHANGED SECTION ---
    # Load and Resize the WHOLE image to 224x224 (No Cropping)
    hr_img = Image.open(IMAGE_PATH).convert("RGB")
    hr_img = hr_img.resize((224, 224), Image.BICUBIC)
    # --- END CHANGE ---

    # Create Low Res (Input) by downsampling 4x
    # LR will be 56x56 pixels
    lr_img = hr_img.resize((224//4, 224//4), Image.BICUBIC)

    # Prepare for model
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    lr_tensor = norm(to_tensor(lr_img)).unsqueeze(0).to(device)

    # 2. Load Models
    try:
        model_baseline = load_model(BASELINE_PATH)
        model_ssl = load_model(SSL_PATH)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 3. Run Inference
    print("Generating images...")
    with torch.no_grad():
        # Baseline Output
        out_baseline = model_baseline(lr_tensor)
        out_baseline = out_baseline.clamp(-1, 1).cpu().squeeze()
        out_baseline = (out_baseline * 0.5 + 0.5).permute(1, 2, 0).numpy()

        # SSL Output
        out_ssl = model_ssl(lr_tensor)
        out_ssl = out_ssl.clamp(-1, 1).cpu().squeeze()
        out_ssl = (out_ssl * 0.5 + 0.5).permute(1, 2, 0).numpy()

    # 4. Plot Results
    print("Plotting...")
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    
    titles = ["Low Res Input (56x56)", "Baseline (From Scratch)", "Ours (SSL Pre-trained)", "Ground Truth (High Res)"]
    
    # We resize LR back to 224 just for display purposes so it isn't tiny
    lr_display = lr_img.resize((224, 224), Image.NEAREST)
    images = [lr_display, out_baseline, out_ssl, hr_img]

    for i, ax in enumerate(axs):
        ax.imshow(images[i])
        ax.set_title(titles[i], fontsize=16)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("result_comparison_full.png", dpi=300)
    print("✅ Success! Saved comparison image to 'result_comparison_full.png'")
    plt.show()

if __name__ == "__main__":
    main()