"""
app.py - AI Medical Image Enhancement GUI

A Tkinter-based graphical interface for medical image enhancement using
multiple deep learning models.

Features:
    - Model Selection: Switch between MIRAM+SRGAN (M1) and MRI Enhancer (M2)
    - Live Preview: Real-time enhancement preview with adjustable parameters
    - Batch Processing: Enhance multiple images at once
    - Parameter Control: Sharpness, Contrast, Brightness sliders
    - ViT Visualization: Compare baseline vs SSL-pretrained results

Required Model Files:
    - SR_best.pth: EnhancedMIRAM weights for M1
    - srgan_best.pth: EnhancedGenerator weights for M1
    - mri_enhancer_best.pth: GeneratorMIRAMSR weights for M2
    - best_generator_baseline.pth: Baseline ViT model (optional, for visualization)
    - best_generator.pth: SSL ViT model (optional, for visualization)

Usage:
    python app.py

Dependencies:
    - torch, torchvision, timm
    - tkinter (usually included with Python)
    - PIL/Pillow
    - matplotlib
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageEnhance, ImageTk
import numpy as np
from torchvision import transforms
import timm
import matplotlib.pyplot as plt

# ===========================================
# DEVICE CONFIGURATION
# ===========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===========================================
# IMPORT MODELS
# ===========================================

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.SR_models import EnhancedMIRAM, EnhancedGenerator
    print("✅ Loaded SR_models from app/")
except ImportError:
    from SR_models import EnhancedMIRAM, EnhancedGenerator
    print("✅ Loaded SR_models from current directory")

try:
    from module_2_mri_super_res.models import GeneratorMIRAMSR
    print("✅ Loaded GeneratorMIRAMSR from module_2_mri_super_res/")
except ImportError:
    # Fallback: Define inline if module not found
    print("⚠️ module_2_mri_super_res not found, using inline definition")
    
    class ChannelAttentionMRI(nn.Module):
        def __init__(self, c, r=16):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Conv2d(c, c // r, 1), nn.ReLU(),
                nn.Conv2d(c // r, c, 1), nn.Sigmoid()
            )
        def forward(self, x):
            return x * self.fc(self.pool(x))

    class SpatialAttentionMRI(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, 7, padding=3)
        def forward(self, x):
            avg = torch.mean(x, dim=1, keepdim=True)
            mx = torch.max(x, dim=1, keepdim=True)[0]
            return x * torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))

    class MIRAMBlockMRI(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.conv1 = nn.Conv2d(c, c, 3, padding=1)
            self.conv2 = nn.Conv2d(c, c, 3, padding=1)
            self.ca = ChannelAttentionMRI(c)
            self.sa = SpatialAttentionMRI()
            self.relu = nn.PReLU()
        def forward(self, x):
            res = self.relu(self.conv1(x))
            res = self.conv2(res)
            return x + self.sa(self.ca(res))

    class ResidualBlockMRI(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.PReLU(),
                nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c)
            )
        def forward(self, x):
            return x + self.block(x)

    class UpsampleBlockMRI(nn.Module):
        def __init__(self, c, scale=2):
            super().__init__()
            self.conv = nn.Conv2d(c, c * scale ** 2, 3, padding=1)
            self.ps = nn.PixelShuffle(scale)
            self.act = nn.PReLU()
        def forward(self, x):
            return self.act(self.ps(self.conv(x)))

    class GeneratorMIRAMSR(nn.Module):
        def __init__(self, in_c=1, base=64, num_res=8, scale=4):
            super().__init__()
            self.head = nn.Sequential(nn.Conv2d(in_c, base, 9, padding=4), nn.PReLU())
            body = []
            for i in range(num_res):
                body.append(ResidualBlockMRI(base))
                if (i+1) % 3 == 0:
                    body.append(MIRAMBlockMRI(base))
            self.body = nn.Sequential(*body)
            self.upsampler = nn.Sequential(UpsampleBlockMRI(base, 2), UpsampleBlockMRI(base, 2))
            self.tail = nn.Conv2d(base, in_c, 9, padding=4)
        def forward(self, x):
            f = self.head(x)
            r = self.body(f)
            out = self.upsampler(f + r)
            return torch.clamp(self.tail(out), 0, 1)


# ===========================================
# LOAD MODELS
# ===========================================

# Model paths (can be customized)
MODEL_DIR = os.environ.get("MODEL_DIR", ".")
MIRAM_SR_PATH = os.path.join(MODEL_DIR, "SR_best.pth")
SRGAN_PATH = os.path.join(MODEL_DIR, "srgan_best.pth")
MRI_PATH = os.path.join(MODEL_DIR, "mri_enhancer_best.pth")
BASELINE_PATH = os.path.join(MODEL_DIR, "best_generator_baseline.pth")
SSL_PATH = os.path.join(MODEL_DIR, "best_generator.pth")

# Initialize models
miramSR = EnhancedMIRAM().to(device)
genSR = EnhancedGenerator().to(device)
mri_gen = GeneratorMIRAMSR(in_c=1, base=64, num_res=8, scale=4).to(device)

# Load weights if available
def load_model_safe(model, path, name):
    """Load model weights with error handling."""
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()
            print(f"✅ Loaded {name} from {path}")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load {name}: {e}")
            return False
    else:
        print(f"⚠️ {name} not found: {path}")
        return False

m1_loaded = load_model_safe(miramSR, MIRAM_SR_PATH, "EnhancedMIRAM")
m1_loaded = load_model_safe(genSR, SRGAN_PATH, "EnhancedGenerator") and m1_loaded
m2_loaded = load_model_safe(mri_gen, MRI_PATH, "MRI Enhancer")

# Current model selector
current_model = "srgan2stage" if m1_loaded else ("mri" if m2_loaded else "none")

# Default parameters
DEFAULT_SHARP = 1.0
DEFAULT_CONTRAST = 1.0
DEFAULT_BRIGHT = 1.0
DEFAULT_PREVIEW_SCALE = 1.0


# ===========================================
# IMAGE PROCESSING UTILITIES
# ===========================================

def to_tensor_gray_01(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized tensor [0, 1]."""
    arr = np.array(img, dtype="float32") / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def to_pil_gray_from_01(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor [0, 1] to PIL image."""
    arr = tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(arr.squeeze(), mode="L")


def pad_to_multiple(img: Image.Image, multiple: int = 4) -> Image.Image:
    """Crop image dimensions to be divisible by multiple."""
    w, h = img.size
    new_w = w - (w % multiple)
    new_h = h - (h % multiple)
    if new_w == w and new_h == h:
        return img
    return img.crop((0, 0, new_w, new_h))


# ===========================================
# ENHANCEMENT FUNCTIONS
# ===========================================

def enhance_image_pil_model(img: Image.Image, sharp: float, contrast: float, bright: float) -> Image.Image:
    """
    Enhance image using the selected AI model.
    
    Args:
        img: Input PIL image
        sharp: Sharpness factor
        contrast: Contrast factor
        bright: Brightness factor
        
    Returns:
        Enhanced PIL image
    """
    with torch.no_grad():
        if current_model == "srgan2stage" and m1_loaded:
            # Model 1: EnhancedMIRAM -> EnhancedGenerator
            img_proc = pad_to_multiple(img.convert("L"), multiple=4)
            t = to_tensor_gray_01(img_proc).unsqueeze(0).to(device)
            restored = miramSR(t)
            sr = genSR(restored)
            final_img = to_pil_gray_from_01(sr.squeeze(0))

        elif current_model == "mri" and m2_loaded:
            # Model 2: MRI Enhancer (4x scale)
            img_proc = pad_to_multiple(img.convert("L"), multiple=4)
            t = to_tensor_gray_01(img_proc).unsqueeze(0).to(device)
            sr = mri_gen(t)
            final_img = to_pil_gray_from_01(sr.squeeze(0))

        else:
            # Fallback: return original (converted to grayscale)
            final_img = img.convert("L")
            print("⚠️ No model loaded, returning original image")

    # Apply PIL enhancements
    final_img = ImageEnhance.Sharpness(final_img).enhance(sharp)
    final_img = ImageEnhance.Contrast(final_img).enhance(contrast)
    final_img = ImageEnhance.Brightness(final_img).enhance(bright)
    
    return final_img


def enhance_image_pil_only(img: Image.Image, sharp: float, contrast: float, bright: float) -> Image.Image:
    """Apply PIL-only enhancements (no AI)."""
    out = img.copy()
    out = ImageEnhance.Sharpness(out).enhance(sharp)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    out = ImageEnhance.Brightness(out).enhance(bright)
    return out


# ===========================================
# VIT MODEL FOR VISUALIZATION
# ===========================================

class ChannelAttentionViT(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, c // r, 1)
        self.fc2 = nn.Conv2d(c // r, c, 1)

    def forward(self, x):
        y = self.pool(x)
        y = F.relu(self.fc1(y))
        return x * torch.sigmoid(self.fc2(y))


class SpatialAttentionViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        y = torch.cat([
            torch.max(x, 1, keepdim=True)[0],
            torch.mean(x, 1, keepdim=True)
        ], 1)
        return x * torch.sigmoid(self.conv(y))


class MIRAMBlockViT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.ca = ChannelAttentionViT(c)
        self.sa = SpatialAttentionViT()

    def forward(self, x):
        out = F.relu(self.c1(x))
        out = self.c2(out)
        out = self.ca(out)
        out = self.sa(out)
        return out + x


class UpsampleBlockViT(nn.Module):
    def __init__(self, c, scale=2):
        super().__init__()
        self.c = nn.Conv2d(c, c * scale**2, 3, padding=1)
        self.ps = nn.PixelShuffle(scale)
        self.a = nn.PReLU()

    def forward(self, x):
        return self.a(self.ps(self.c(x)))


class GeneratorViT_SR(nn.Module):
    """ViT-based generator for SSL comparison visualization."""
    
    def __init__(self, in_c=3, embed_dim=768, decoder_dim=256, scale=4):
        super().__init__()
        self.encoder = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0
        )
        self.resize = transforms.Resize((224, 224), antialias=True)
        self.feature_conv = nn.Linear(embed_dim, decoder_dim)
        self.miram = MIRAMBlockViT(decoder_dim)
        self.tail = nn.Sequential(
            UpsampleBlockViT(decoder_dim, 2),
            UpsampleBlockViT(decoder_dim, 2),
            UpsampleBlockViT(decoder_dim, 2),
            UpsampleBlockViT(decoder_dim, 2),
            nn.Conv2d(decoder_dim, in_c, 9, padding=4),
        )

    def forward(self, x):
        x_resized = self.resize(x)
        features = self.encoder.forward_features(x_resized)[:, 1:, :]
        features = self.feature_conv(features)
        features = features.permute(0, 2, 1).reshape(x.shape[0], -1, 14, 14)
        features = self.miram(features)
        return self.tail(features)


def load_vit_model(path: str) -> nn.Module:
    """Load ViT model for visualization."""
    model = GeneratorViT_SR().to(device)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ===========================================
# TKINTER GUI
# ===========================================

root = Tk()
root.title("AI Medical Image Enhancer")
root.geometry("1000x800")
root.resizable(True, True)

# State variables
selected_images = []
output_folder = ""
preview_window = None
preview_canvas = None
preview_tk_image = None
example_image_path = None
example_original_pil = None
preview_job = None


def get_current_params():
    """Get current slider values."""
    return slider_sharp.get(), slider_contrast.get(), slider_bright.get()


def get_preview_scale():
    """Get preview scale factor."""
    return slider_preview_scale.get()


def schedule_preview_update():
    """Debounced preview update."""
    global preview_job
    if preview_job is not None:
        root.after_cancel(preview_job)
    preview_job = root.after(200, update_live_preview)


def ensure_preview_window():
    """Create or show preview window."""
    global preview_window, preview_canvas
    if preview_window is None or not preview_window.winfo_exists():
        preview_window = Toplevel(root)
        preview_window.title("Live Preview")
        preview_window.geometry("600x600")
        preview_canvas = Label(preview_window, bg="gray")
        preview_canvas.pack(fill=BOTH, expand=True)


def get_preview_box_from_window():
    """Get preview window dimensions."""
    if preview_window and preview_window.winfo_exists():
        return preview_window.winfo_width() - 20, preview_window.winfo_height() - 20
    return 500, 500


def get_scaled_size(orig_w, orig_h, max_w, max_h):
    """Calculate scaled size maintaining aspect ratio."""
    scale = min(max_w / orig_w, max_h / orig_h, 1.0)
    return int(orig_w * scale), int(orig_h * scale)


def select_images():
    """Open file dialog to select images."""
    global selected_images
    files = filedialog.askopenfilenames(
        filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp")]
    )
    if files:
        selected_images = list(files)
        img_list.delete(0, END)
        for f in selected_images:
            img_list.insert(END, os.path.basename(f))


def select_output():
    """Open folder dialog for output."""
    global output_folder
    folder = filedialog.askdirectory()
    if folder:
        output_folder = folder
        output_label.config(text=f"Output: {folder}")


def select_example_image():
    """Select image for preview."""
    global example_image_path, example_original_pil
    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp")]
    )
    if path:
        example_image_path = path
        example_original_pil = Image.open(path)
        update_live_preview()


def update_live_preview():
    """Update the live preview with current settings."""
    global example_original_pil, preview_canvas, preview_tk_image
    if example_original_pil is None:
        return

    ensure_preview_window()

    sharp, contrast, bright = get_current_params()
    preview_scale = get_preview_scale()

    orig_w, orig_h = example_original_pil.size
    max_w, max_h = get_preview_box_from_window()
    base_w, base_h = get_scaled_size(orig_w, orig_h, max_w, max_h)

    display_w = max(1, int(base_w * preview_scale))
    display_h = max(1, int(base_h * preview_scale))

    # Limit preview inference size for speed
    PREVIEW_MAX = 512
    scale = min(PREVIEW_MAX / orig_w, PREVIEW_MAX / orig_h, 1.0)
    model_w = int(orig_w * scale)
    model_h = int(orig_h * scale)
    img_for_model = example_original_pil.resize((model_w, model_h), Image.BICUBIC)

    try:
        result = enhance_image_pil_model(img_for_model, sharp, contrast, bright)
    except Exception as ex:
        print(f"Error in AI preview: {ex}")
        result = enhance_image_pil_only(img_for_model, sharp, contrast, bright)

    result_display = result.resize((display_w, display_h), Image.BICUBIC)
    preview_tk_image = ImageTk.PhotoImage(result_display)
    preview_canvas.config(image=preview_tk_image)
    preview_canvas.image = preview_tk_image


def run_enhancement():
    """Run batch enhancement on selected images."""
    if not selected_images:
        messagebox.showerror("Error", "No images selected.")
        return
    if not output_folder:
        messagebox.showerror("Error", "Please select output folder.")
        return

    sharp, contrast, bright = get_current_params()

    model_prefix = {"srgan2stage": "M1", "mri": "M2"}.get(current_model, "MX")

    for img_path in selected_images:
        img = Image.open(img_path)
        enhanced = enhance_image_pil_model(img, sharp, contrast, bright)
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_folder, f"{model_prefix}_enh_{filename}")
        enhanced.save(out_path)

    messagebox.showinfo("Done", f"Enhancement complete! {len(selected_images)} images processed.")


def reset_params():
    """Reset all sliders to defaults."""
    slider_sharp.set(DEFAULT_SHARP)
    slider_contrast.set(DEFAULT_CONTRAST)
    slider_bright.set(DEFAULT_BRIGHT)
    slider_preview_scale.set(DEFAULT_PREVIEW_SCALE)
    update_live_preview()


def set_model_srgan2stage():
    """Switch to MIRAM+SRGAN model."""
    global current_model
    if not m1_loaded:
        messagebox.showwarning("Warning", "M1 model files not found!")
        return
    current_model = "srgan2stage"
    model_label.config(text="Model: EnhancedMIRAM + SRGAN (M1)", fg="blue")
    update_live_preview()


def set_model_mri():
    """Switch to MRI Enhancer model."""
    global current_model
    if not m2_loaded:
        messagebox.showwarning("Warning", "M2 model file not found!")
        return
    current_model = "mri"
    model_label.config(text="Model: MRI Enhancer (M2)", fg="darkgreen")
    update_live_preview()


def visualize_vit_results():
    """Generate ViT baseline vs SSL comparison visualization."""
    if example_original_pil is None:
        messagebox.showerror("Error", "Please select a preview example image first.")
        return

    if not os.path.exists(BASELINE_PATH) or not os.path.exists(SSL_PATH):
        messagebox.showerror("Error", 
            f"ViT model files not found!\nRequired:\n- {BASELINE_PATH}\n- {SSL_PATH}")
        return

    try:
        # Prepare images
        hr_img = example_original_pil.convert("RGB")
        hr_img = hr_img.resize((224, 224), Image.BICUBIC)
        lr_img = hr_img.resize((224 // 4, 224 // 4), Image.BICUBIC)

        to_tensor = transforms.ToTensor()
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        lr_tensor = norm(to_tensor(lr_img)).unsqueeze(0).to(device)

        # Load and run models
        model_baseline = load_vit_model(BASELINE_PATH)
        model_ssl = load_vit_model(SSL_PATH)

        with torch.no_grad():
            out_baseline = model_baseline(lr_tensor)
            out_baseline = out_baseline.clamp(-1, 1).cpu().squeeze()
            out_baseline = (out_baseline * 0.5 + 0.5).permute(1, 2, 0).numpy()

            out_ssl = model_ssl(lr_tensor)
            out_ssl = out_ssl.clamp(-1, 1).cpu().squeeze()
            out_ssl = (out_ssl * 0.5 + 0.5).permute(1, 2, 0).numpy()

        # Plot comparison
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        titles = [
            "Low Res Input (56×56)",
            "Baseline (From Scratch)",
            "Ours (SSL Pre-trained)",
            "Ground Truth (High Res)"
        ]
        lr_display = lr_img.resize((224, 224), Image.NEAREST)
        images = [lr_display, out_baseline, out_ssl, hr_img]

        for ax, img, title in zip(axs, images, titles):
            ax.imshow(img)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis("off")

        plt.tight_layout()
        plt.savefig("result_comparison_vit.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Saved visualization to result_comparison_vit.png")

    except Exception as e:
        messagebox.showerror("Error", f"Visualization failed: {e}")


# ===========================================
# BUILD GUI LAYOUT
# ===========================================

Label(root, text="AI Medical Image Enhancement Tool", font=("Arial", 22, "bold")).pack(pady=10)

# Model selection
model_frame = Frame(root)
model_frame.pack(pady=5)

Button(model_frame, text="M1: MIRAM+SRGAN", width=18, command=set_model_srgan2stage).grid(row=0, column=0, padx=5)
Button(model_frame, text="M2: MRI Enhancer", width=18, command=set_model_mri).grid(row=0, column=1, padx=5)

model_label = Label(root, text=f"Model: {'MIRAM+SRGAN (M1)' if current_model == 'srgan2stage' else 'MRI Enhancer (M2)'}", 
                   font=("Arial", 10), fg="blue" if current_model == "srgan2stage" else "darkgreen")
model_label.pack(pady=5)

# File selection
btn_frame = Frame(root)
btn_frame.pack()

Button(btn_frame, text="Select Images", width=20, command=select_images).grid(row=0, column=0, padx=10)
Button(btn_frame, text="Select Output Folder", width=20, command=select_output).grid(row=0, column=1, padx=10)

output_label = Label(root, text="Output: Not selected", font=("Arial", 10))
output_label.pack(pady=5)

img_list = Listbox(root, width=60, height=8)
img_list.pack(pady=5)

Button(root, text="Select Preview Example", width=25, command=select_example_image).pack(pady=10)

# Parameter sliders
slider_frame = Frame(root)
slider_frame.pack(pady=10)

Label(slider_frame, text="Sharpness").grid(row=0, column=0, sticky="w")
slider_sharp = Scale(slider_frame, from_=0.0, to=50.0, orient=HORIZONTAL, 
                     resolution=0.1, length=250, command=lambda v: schedule_preview_update())
slider_sharp.set(DEFAULT_SHARP)
slider_sharp.grid(row=0, column=1, padx=10)

Label(slider_frame, text="Contrast").grid(row=1, column=0, sticky="w")
slider_contrast = Scale(slider_frame, from_=0.0, to=4.0, orient=HORIZONTAL,
                        resolution=0.01, length=250, command=lambda v: schedule_preview_update())
slider_contrast.set(DEFAULT_CONTRAST)
slider_contrast.grid(row=1, column=1, padx=10)

Label(slider_frame, text="Brightness").grid(row=2, column=0, sticky="w")
slider_bright = Scale(slider_frame, from_=0.0, to=4.0, orient=HORIZONTAL,
                      resolution=0.01, length=250, command=lambda v: schedule_preview_update())
slider_bright.set(DEFAULT_BRIGHT)
slider_bright.grid(row=2, column=1, padx=10)

Label(slider_frame, text="Preview Size (×)").grid(row=3, column=0, sticky="w")
slider_preview_scale = Scale(slider_frame, from_=0.2, to=2.0, orient=HORIZONTAL,
                             resolution=0.05, length=250, command=lambda v: schedule_preview_update())
slider_preview_scale.set(DEFAULT_PREVIEW_SCALE)
slider_preview_scale.grid(row=3, column=1, padx=10)

Button(slider_frame, text="Reset to Default", command=reset_params).grid(row=4, column=0, columnspan=2, pady=5)

# Action buttons
Button(root, text="Start Enhancement", font=("Arial", 14), bg="#4CAF50", fg="white", 
       command=run_enhancement).pack(pady=15)

Button(root, text="Visualize ViT Results (SSL vs Baseline)", font=("Arial", 12), 
       bg="#2196F3", fg="white", command=visualize_vit_results).pack(pady=5)

# Status bar
status_label = Label(root, text=f"M1: {'✅' if m1_loaded else '❌'}  |  M2: {'✅' if m2_loaded else '❌'}", 
                    font=("Arial", 9), fg="gray")
status_label.pack(side=BOTTOM, pady=5)

# Run the application
if __name__ == "__main__":
    root.mainloop()
