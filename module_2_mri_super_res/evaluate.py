    # evaluate.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from config import *
from models import GeneratorMIRAMSR
from dataset import get_dataloaders
from losses import psnr, ssim

def evaluate():
    _, _, test_dl = get_dataloaders(DATASET_PATH, 1, CROP_SIZE, SCALE_FACTOR)
    if not test_dl: return
    
    gen = GeneratorMIRAMSR(1, scale=SCALE_FACTOR).to(DEVICE)
    try: gen.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    except: return print("❌ No model found.")
    gen.eval()
    
    print("\n🔍 Evaluating...")
    for i, (lr, hr, mask) in enumerate(test_dl):
        if i >= 3: break # Show top 3
        
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)
        with torch.no_grad():
            sr = gen(lr).clamp(0, 1)
            
        p_val = psnr(sr, hr).item()
        
        # Viz
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(lr[0,0].cpu(), cmap='gray'); ax[0].set_title("Input (LR)")
        ax[1].imshow(sr[0,0].cpu(), cmap='gray'); ax[1].set_title(f"SR (PSNR: {p_val:.2f})")
        ax[2].imshow(hr[0,0].cpu(), cmap='gray'); ax[2].set_title("Target (HR)")
        plt.show()

if __name__ == "__main__": evaluate()