"""
enhance.py - Single Image Super-Resolution Inference

This script enhances a single MRI image using the trained MIRAM model.

Usage:
    python enhance.py --input path/to/image.tif --output enhanced.tif
    
    # Or use defaults:
    python enhance.py

Output:
    - Super-resolved image saved to specified path
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

from config import BEST_MODEL_PATH, DEVICE, SCALE_FACTOR, OUTPUT_DIR
from models import GeneratorMIRAMSR


def load_image(path: str) -> torch.Tensor:
    """
    Load an image and convert to tensor.
    
    Args:
        path: Path to input image
        
    Returns:
        Tensor of shape (1, 1, H, W)
    """
    path = str(path)
    
    if path.lower().endswith(('.tif', '.tiff')) and HAS_TIFFFILE:
        img = tifffile.imread(path)
    else:
        img = np.array(Image.open(path))
    
    # Handle multi-channel
    if img.ndim == 3:
        img = img[:, :, 0] if img.shape[2] < img.shape[0] else img[0]
    
    # Normalize
    img = img.astype(np.float32)
    if img.max() > 255:
        img /= 65535.0
    elif img.max() > 1:
        img /= 255.0
    
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


def save_image(tensor: torch.Tensor, path: str, use_16bit: bool = True):
    """
    Save tensor as image.
    
    Args:
        tensor: Image tensor of shape (1, 1, H, W)
        path: Output path
        use_16bit: Save as 16-bit TIFF if True
    """
    img = tensor.squeeze().cpu().numpy()
    img = np.clip(img, 0, 1)
    
    path = str(path)
    
    if use_16bit and path.lower().endswith(('.tif', '.tiff')) and HAS_TIFFFILE:
        tifffile.imwrite(path, (img * 65535).astype(np.uint16))
    else:
        Image.fromarray((img * 255).astype(np.uint8)).save(path)
    
    print(f"✅ Saved to: {path}")


def enhance(input_path: str, output_path: str, model_path: str = None):
    """
    Enhance a single image using the trained model.
    
    Args:
        input_path: Path to input LR image
        output_path: Path to save SR output
        model_path: Path to model weights (default: BEST_MODEL_PATH)
    """
    if model_path is None:
        model_path = BEST_MODEL_PATH
    
    # Load model
    print(f"Loading model from {model_path}...")
    generator = GeneratorMIRAMSR(in_channels=1, scale=SCALE_FACTOR).to(DEVICE)
    
    try:
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        generator.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"❌ Model not found: {model_path}")
        print("   Run train.py first to train the model.")
        return
    
    generator.eval()
    
    # Load and process image
    print(f"Loading image from {input_path}...")
    lr_tensor = load_image(input_path).to(DEVICE)
    
    print(f"Input shape: {lr_tensor.shape[2]}x{lr_tensor.shape[3]}")
    print(f"Enhancing at {SCALE_FACTOR}x scale...")
    
    # Generate SR image
    with torch.no_grad():
        sr_tensor = generator(lr_tensor).clamp(0, 1)
    
    print(f"Output shape: {sr_tensor.shape[2]}x{sr_tensor.shape[3]}")
    
    # Save result
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    save_image(sr_tensor, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Enhance MRI image using MIRAM super-resolution"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Path to input LR image'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save SR output'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=BEST_MODEL_PATH,
        help='Path to model weights'
    )
    args = parser.parse_args()
    
    # Handle defaults
    if args.input is None:
        print("Usage: python enhance.py --input path/to/image.tif --output enhanced.tif")
        print("\nNo input specified. Looking for sample image...")
        
        # Try to find a sample image
        from config import DATASET_PATH
        lr_dir = Path(DATASET_PATH) / "LR"
        
        if lr_dir.exists():
            samples = list(lr_dir.glob("*"))
            if samples:
                args.input = str(samples[0])
                print(f"Found sample: {args.input}")
            else:
                print("❌ No images found in LR directory.")
                return
        else:
            print(f"❌ Dataset not found at {DATASET_PATH}")
            return
    
    if args.output is None:
        input_path = Path(args.input)
        args.output = os.path.join(
            OUTPUT_DIR,
            f"{input_path.stem}_enhanced{input_path.suffix}"
        )
    
    # Run enhancement
    enhance(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
