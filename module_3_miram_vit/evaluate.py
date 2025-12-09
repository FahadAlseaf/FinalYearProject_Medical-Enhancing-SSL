# evaluate.py - RESTORATION VISUALIZATION
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from config import *
from models import MIRAM
from dataset import get_dataloaders
from losses import psnr, ssim

def unpatchify(x, p_size):
    N, L, D = x.shape
    h = w = int(L**0.5)
    x = x.reshape(shape=(N, h, w, p_size, p_size, 1))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(N, 1, h * p_size, w * p_size))
    return imgs

def evaluate():
    _, _, test_dl = get_dataloaders()
    model = MIRAM().to(DEVICE)
    
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    else:
        print("❌ Model not found. Run train.py first.")
        return
        
    model.eval()
    print("\n🔍 Evaluating MIRAM Reconstruction...")
    
    img_fine, img_coarse = next(iter(test_dl))
    img_fine = img_fine.to(DEVICE)
    
    with torch.no_grad():
        pred_fine, pred_coarse, mask = model(img_fine, MASK_RATIO)
        recon_fine = unpatchify(pred_fine, PATCH_SIZE)
        
        p_val = psnr(recon_fine, img_fine).item()
        s_val = ssim(recon_fine, img_fine).item()
        
        print(f"📊 Metrics -> PSNR: {p_val:.2f} dB | SSIM: {s_val:.4f}")

        # Visualize
        mask_img = mask.unsqueeze(-1).repeat(1, 1, PATCH_SIZE**2 * 1)
        mask_img = unpatchify(mask_img, PATCH_SIZE)
        im_masked = img_fine * (1 - mask_img)
        im_paste = img_fine * (1 - mask_img) + recon_fine * mask_img
        
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(img_fine[0,0].cpu(), cmap='gray'); ax[0].set_title("GT")
    ax[1].imshow(im_masked[0,0].cpu(), cmap='gray'); ax[1].set_title("Masked")
    ax[2].imshow(recon_fine[0,0].cpu(), cmap='gray'); ax[2].set_title("Recon")
    ax[3].imshow(im_paste[0,0].cpu(), cmap='gray'); ax[3].set_title("Result")
    
    save_path = os.path.join(OUTPUT_DIR, 'miram_eval_sample.png')
    plt.savefig(save_path)
    print(f"💾 Saved visualization to: {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate()