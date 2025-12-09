# enhance.py
import torch
import tifffile
import numpy as np
from PIL import Image
from config import *
from models import GeneratorMIRAMSR

def enhance(img_path, out_path):
    gen = GeneratorMIRAMSR(1, scale=SCALE_FACTOR).to(DEVICE)
    gen.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    gen.eval()
    
    # Load
    if img_path.endswith('tif'): img = tifffile.imread(img_path).astype(np.float32)/65535.
    else: img = np.array(Image.open(img_path)).astype(np.float32)/255.
    
    if img.ndim==2: img = img[None, None, ...]
    t_in = torch.from_numpy(img).float().to(DEVICE)
    
    with torch.no_grad():
        sr = gen(t_in).clamp(0, 1)
        
    out = sr.squeeze().cpu().numpy()
    tifffile.imwrite(out_path, (out*65535).astype(np.uint16))
    print(f"✅ Saved to {out_path}")

if __name__ == "__main__":
    p = input("Image Path: ")
    enhance(p, "output.tif")