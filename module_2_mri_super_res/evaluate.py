"""
evaluate.py - Evaluation Script for MIRAM MRI Super-Resolution

This script loads the trained model and evaluates it on the test set,
computing PSNR and SSIM metrics and generating visual comparisons.

Usage:
    python evaluate.py

Output:
    - Prints average PSNR/SSIM on test set
    - Displays visual comparisons for sample images
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from config import DATASET_PATH, BEST_MODEL_PATH, DEVICE, SCALE_FACTOR, CROP_SIZE
from models import GeneratorMIRAMSR
from dataset import get_dataloaders
from losses import psnr, ssim


def evaluate():
    """Evaluate the trained model on the test set."""
    
    print("=" * 60)
    print("MIRAM MRI Super-Resolution Evaluation")
    print("=" * 60)
    
    # Load data
    _, _, test_loader = get_dataloaders(DATASET_PATH, batch_size=1)
    
    if test_loader is None or len(test_loader) == 0:
        print("Error: No test data available.")
        return
    
    # Load model
    generator = GeneratorMIRAMSR(in_channels=1, scale=SCALE_FACTOR).to(DEVICE)
    
    try:
        state_dict = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
        generator.load_state_dict(state_dict)
        print(f"✅ Loaded model from {BEST_MODEL_PATH}")
    except FileNotFoundError:
        print(f"❌ Model not found at {BEST_MODEL_PATH}")
        print("   Run train.py first to train the model.")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    generator.eval()
    
    # Evaluate
    print(f"\n🔍 Evaluating on {len(test_loader)} test samples...")
    
    total_psnr = 0.0
    total_ssim = 0.0
    results = []
    
    with torch.no_grad():
        for idx, (lr, hr, mask) in enumerate(test_loader):
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)
            
            # Generate super-resolved image
            sr = generator(lr).clamp(0, 1)
            
            # Compute metrics
            p_val = psnr(sr, hr).item()
            s_val = ssim(sr, hr).item()
            
            total_psnr += p_val
            total_ssim += s_val
            
            # Store for visualization
            if idx < 5:  # Keep first 5 samples
                results.append({
                    'lr': lr.cpu().squeeze().numpy(),
                    'sr': sr.cpu().squeeze().numpy(),
                    'hr': hr.cpu().squeeze().numpy(),
                    'psnr': p_val,
                    'ssim': s_val
                })
    
    # Calculate averages
    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    
    print("\n" + "=" * 40)
    print("📊 Test Results")
    print("=" * 40)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("=" * 40)
    
    # Visualize results
    if len(results) > 0:
        print("\n📷 Generating visual comparisons...")
        visualize_results(results[:3])  # Show top 3


def visualize_results(results: list):
    """
    Create visual comparisons of LR, SR, and HR images.
    
    Args:
        results: List of dictionaries containing lr, sr, hr, psnr, ssim
    """
    n_samples = len(results)
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # Input (LR)
        axes[i, 0].imshow(result['lr'], cmap='gray')
        axes[i, 0].set_title("Input (LR)", fontsize=12)
        axes[i, 0].axis('off')
        
        # Super-resolved
        axes[i, 1].imshow(result['sr'], cmap='gray')
        axes[i, 1].set_title(
            f"Super-Resolved\nPSNR: {result['psnr']:.2f} dB",
            fontsize=12
        )
        axes[i, 1].axis('off')
        
        # Ground truth (HR)
        axes[i, 2].imshow(result['hr'], cmap='gray')
        axes[i, 2].set_title("Ground Truth (HR)", fontsize=12)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
    print("✅ Saved visualization to 'evaluation_results.png'")
    
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    evaluate()
