"""
enhance.py - MIRAM Single Image Inference

This script performs image restoration/denoising using the trained
MIRAM model. In inference mode (mask_ratio=0), the model acts as
a denoising autoencoder.

Usage:
    python enhance.py
    python enhance.py --input image.tif --output restored.tif

Output:
    - Restored 16-bit TIFF image
"""

import os
import argparse
import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

from config import DEVICE, BEST_MODEL_PATH, IMG_SIZE, PATCH_SIZE, OUTPUT_DIR
from models import MIRAM, unpatchify


def load_image(path: str) -> np.ndarray:
    """Load image and normalize to [0, 1]."""
    path = str(path)
    
    try:
        if path.lower().endswith(('.tif', '.tiff')) and HAS_TIFFFILE:
            img = tifffile.imread(path).astype(np.float32)
            if img.max() > 255:
                img /= 65535.0
            else:
                img /= 255.0
        else:
            img = np.array(Image.open(path)).astype(np.float32) / 255.0
        
        # Convert to grayscale if needed
        if img.ndim == 3:
            img = img[0, :, :] if img.shape[0] < 5 else img[:, :, 0]
        
        return img
        
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None


def save_image(img: np.ndarray, path: str, use_16bit: bool = True):
    """Save image as TIFF or PNG."""
    path = str(path)
    
    # Clip to valid range
    img = np.clip(img, 0, 1)
    
    if use_16bit and path.lower().endswith(('.tif', '.tiff')) and HAS_TIFFFILE:
        tifffile.imwrite(path, (img * 65535).astype(np.uint16))
    else:
        Image.fromarray((img * 255).astype(np.uint8)).save(path)
    
    print(f"✅ Saved to: {path}")


def enhance(input_path: str, output_path: str):
    """
    Restore/denoise a single image using MIRAM.
    
    In inference mode (mask_ratio=0), all patches are visible and
    the model acts as a learned denoising filter.
    
    Args:
        input_path: Path to input image
        output_path: Path to save restored image
    """
    print("=" * 60)
    print("MIRAM Image Enhancement")
    print("=" * 60)
    
    # Load model
    print(f"Loading model from {BEST_MODEL_PATH}...")
    
    model = MIRAM().to(DEVICE)
    
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(
            torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
        )
    else:
        print(f"❌ Model not found: {BEST_MODEL_PATH}")
        print("   Run train.py first.")
        return
    
    model.eval()
    
    # Load image
    print(f"Processing: {input_path}")
    img = load_image(input_path)
    
    if img is None:
        return
    
    original_size = img.shape
    print(f"Original size: {original_size}")
    
    # Convert to tensor
    t_in = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    t_in = transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True)(t_in)
    t_in = t_in.to(DEVICE)
    
    # Run inference
    print("Running inference...")
    
    start_time = time.time()
    
    with torch.no_grad():
        # mask_ratio=0 means all patches are visible (denoising mode)
        pred_fine, _, _ = model(t_in, mask_ratio=0.0)
        out_t = unpatchify(pred_fine, PATCH_SIZE)
        out_t = torch.clamp(out_t, 0, 1)
    
    end_time = time.time()
    duration = end_time - start_time
    fps = 1.0 / duration if duration > 0 else 0
    
    print(f"⏱️ Inference Time: {duration*1000:.1f}ms | Speed: {fps:.2f} FPS")
    
    # Save output
    out_np = out_t.squeeze().cpu().numpy()
    
    # Optionally resize back to original size
    if original_size != (IMG_SIZE, IMG_SIZE):
        from PIL import Image as PILImage
        out_pil = PILImage.fromarray((out_np * 255).astype(np.uint8))
        out_pil = out_pil.resize((original_size[1], original_size[0]), PILImage.BICUBIC)
        out_np = np.array(out_pil).astype(np.float32) / 255.0
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    save_image(out_np, output_path)
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Enhance medical image using MIRAM"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Path to input image'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save enhanced image'
    )
    args = parser.parse_args()
    
    # Interactive mode if no arguments
    if args.input is None:
        args.input = input("Enter path to input image: ").strip().strip('"')
    
    if not args.input:
        print("No input provided.")
        return
    
    if args.output is None:
        args.output = os.path.join(OUTPUT_DIR, "restored_output.tif")
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return
    
    enhance(args.input, args.output)


if __name__ == "__main__":
    main()
