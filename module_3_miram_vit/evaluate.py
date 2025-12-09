"""
evaluate.py - MIRAM Reconstruction Visualization

This script evaluates the trained MIRAM model by:
1. Loading a test image
2. Applying random masking
3. Reconstructing the masked patches
4. Visualizing the results

Usage:
    python evaluate.py

Output:
    - miram_eval_sample.png: Side-by-side comparison
"""

import os
import torch
import matplotlib.pyplot as plt

from config import DEVICE, BEST_MODEL_PATH, OUTPUT_DIR, MASK_RATIO, PATCH_SIZE
from models import MIRAM, unpatchify
from dataset import get_dataloaders
from losses import psnr, ssim


def evaluate():
    """Evaluate MIRAM reconstruction quality."""
    
    print("=" * 60)
    print("MIRAM Reconstruction Evaluation")
    print("=" * 60)
    
    # Load data
    _, _, test_dl = get_dataloaders()
    if test_dl is None:
        print("Error: Could not load test data.")
        return
    
    # Load model
    model = MIRAM().to(DEVICE)
    
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(
            torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
        )
        print(f"✅ Loaded model from {BEST_MODEL_PATH}")
    else:
        print(f"❌ Model not found: {BEST_MODEL_PATH}")
        print("   Run train.py first.")
        return
    
    model.eval()
    
    # Get a test sample
    img_fine, img_coarse = next(iter(test_dl))
    img_fine = img_fine.to(DEVICE)
    
    print(f"\n🔍 Evaluating reconstruction (mask ratio: {MASK_RATIO})...")
    
    with torch.no_grad():
        # Forward pass
        pred_fine, pred_coarse, mask = model(img_fine, MASK_RATIO)
        
        # Reconstruct image from patches
        recon_fine = unpatchify(pred_fine, PATCH_SIZE)
        recon_fine = torch.clamp(recon_fine, 0, 1)
        
        # Compute metrics
        p_val = psnr(recon_fine, img_fine).item()
        s_val = ssim(recon_fine, img_fine).item()
        
        print(f"\n📊 Metrics:")
        print(f"   PSNR: {p_val:.2f} dB")
        print(f"   SSIM: {s_val:.4f}")
        
        # Create masked visualization
        # Convert mask to image space
        mask_img = mask.unsqueeze(-1).repeat(1, 1, PATCH_SIZE ** 2)
        mask_img = unpatchify(mask_img, PATCH_SIZE)
        
        # Masked input (visible patches only)
        im_masked = img_fine * (1 - mask_img)
        
        # Paste reconstruction into masked regions
        im_paste = img_fine * (1 - mask_img) + recon_fine * mask_img
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    titles = [
        "Ground Truth",
        f"Masked Input\n({int(MASK_RATIO*100)}% hidden)",
        "Full Reconstruction",
        "Masked Regions Filled"
    ]
    
    images = [
        img_fine[0, 0].cpu().numpy(),
        im_masked[0, 0].cpu().numpy(),
        recon_fine[0, 0].cpu().numpy(),
        im_paste[0, 0].cpu().numpy()
    ]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    # Add metrics text
    fig.suptitle(f"MIRAM Reconstruction | PSNR: {p_val:.2f} dB | SSIM: {s_val:.4f}",
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'miram_eval_sample.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved visualization to: {save_path}")
    
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    evaluate()
