# enhance.py - MIRAM INFERENCE (RESTORATION & DENOISING)
import torch
import tifffile
import numpy as np
import time
from PIL import Image
from torchvision import transforms
from config import *
from models import MIRAM

def unpatchify(x, p_size):
    """
    Reconstruct image from patches.
    x: (N, L, patch_size**2 * 1) -> imgs: (N, 1, H, W)
    """
    N, L, D = x.shape
    h = w = int(L**0.5)
    x = x.reshape(shape=(N, h, w, p_size, p_size, 1))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(N, 1, h * p_size, w * p_size))
    return imgs

def enhance(img_path, out_path):
    """
    Runs the MIRAM model on a single image to restore/denoise it.
    """
    # 1. Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Loading MIRAM model from {BEST_MODEL_PATH}...")
    
    model = MIRAM().to(device)
    
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    else:
        print(f"❌ Error: Model file not found at {BEST_MODEL_PATH}")
        return

    model.eval()
    
    # 2. Load & Preprocess Image
    print(f"🔍 Processing: {img_path}")
    try:
        if img_path.lower().endswith(('.tif', '.tiff')):
            img = tifffile.imread(img_path).astype(np.float32)
            if img.max() > 255: img /= 65535.0
            else: img /= 255.0
        else:
            img = np.array(Image.open(img_path)).astype(np.float32) / 255.0
            
        # Handle dimensions (ensure 2D)
        if img.ndim == 3: img = img[0, :, :] if img.shape[0] < 5 else img[:, :, 0]
            
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
        # Transform to Tensor [1, 1, H, W]
    t_in = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    
        # Resize to model's expected input size (e.g., 224x224)
    t_in = transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True)(t_in).to(device)
    
    # 3. Run Inference
    # --- FPS TIMER START ---
    start_time = time.time()
    with torch.no_grad():
        # Pass with mask_ratio=0.0 to use all pixels (Denoising Mode)
        pred_fine, _, _ = model(t_in, mask_ratio=0.0)
        out_t = unpatchify(pred_fine, PATCH_SIZE)
        out_t = torch.clamp(out_t, 0, 1)
    
    end_time = time.time()
    duration = end_time - start_time
    fps = 1.0 / duration if duration > 0 else 0
    print(f"⏱️ Inference Time: {duration*1000:.1f}ms | Speed: {fps:.2f} FPS")
    # --- FPS TIMER END ---

    # 4. Save Output
    out_np = out_t.squeeze().cpu().numpy()
    out_uint16 = (out_np * 65535).astype(np.uint16)
    tifffile.imwrite(out_path, out_uint16)
    
    print(f"✅ Enhanced image saved to: {out_path}")

if __name__ == "__main__":
    # Interactive mode
    p = input("Enter path to input image (e.g., test.tif): ").strip().strip('"')
    if p:
        enhance(p, "restored_output.tif")