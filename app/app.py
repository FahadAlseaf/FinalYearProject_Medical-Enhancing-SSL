import os
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

# =========================
# DEVICE & MODEL LOADING
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from SR import EnhancedMIRAM, EnhancedGenerator
from mri import GeneratorMIRAMSR

# ---- Model 1: MIRAM-SR + SRGAN (two-stage) ----
miramSR = EnhancedMIRAM().to(device)
genSR = EnhancedGenerator().to(device)
miramSR.load_state_dict(torch.load("SR_best.pth", map_location=device))
genSR.load_state_dict(torch.load("srgan_best.pth", map_location=device))
miramSR.eval()
genSR.eval()

# ---- Model 2: MRI Enhancer (GeneratorMIRAMSR) ----
mri_gen = GeneratorMIRAMSR(in_c=1, base=64, num_res=8, scale=4).to(device)
mri_gen.load_state_dict(torch.load("mri_enhancer_best.pth", map_location=device))
mri_gen.eval()

# current model selector: "srgan2stage" or "mri"
current_model = "srgan2stage"

# default parameter values
DEFAULT_SHARP = 1.0
DEFAULT_CONTRAST = 1.0
DEFAULT_BRIGHT = 1.0
DEFAULT_PREVIEW_SCALE = 1.0

# =========================
# BASIC TENSOR ↔ PIL HELPERS
# =========================
def to_tensor_gray_01(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype="float32") / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def to_pil_gray_from_01(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(arr.squeeze(), mode="L")


def pad_to_multiple(img: Image.Image, multiple: int = 4) -> Image.Image:
    w, h = img.size
    new_w = w - (w % multiple)
    new_h = h - (h % multiple)
    if new_w == w and new_h == h:
        return img
    return img.crop((0, 0, new_w, new_h))


# =========================
# ENHANCEMENT FUNCTION (AI)
# =========================
def enhance_image_pil_model(img: Image.Image, sharp: float, contrast: float, bright: float) -> Image.Image:
    """
    img: PIL image (any mode). Uses models according to current_model.
    This is your original MIRAM+SRGAN and MRI logic.
    """
    with torch.no_grad():
        if current_model == "srgan2stage":
            # Model 1: EnhancedMIRAM -> EnhancedGenerator (original behavior)
            img_proc = pad_to_multiple(img.convert("L"), multiple=4)
            t = to_tensor_gray_01(img_proc).unsqueeze(0).to(device)  # (1,1,H,W)
            restored = miramSR(t)                                    # stage 1
            sr = genSR(restored)                                     # stage 2
            final_img = to_pil_gray_from_01(sr.squeeze(0))

        elif current_model == "mri":
            # Model 2: MRI Enhancer (scale 4); enforce multiple-of-4
            img_proc = pad_to_multiple(img.convert("L"), multiple=4)
            t = to_tensor_gray_01(img_proc).unsqueeze(0).to(device)
            sr = mri_gen(t)
            final_img = to_pil_gray_from_01(sr.squeeze(0))

        else:
            # Fallback: use original image if model not recognized
            final_img = img.convert("L")

    # Apply PIL enhancements on whatever size the model produced
    final_img = ImageEnhance.Sharpness(final_img).enhance(sharp)
    final_img = ImageEnhance.Contrast(final_img).enhance(contrast)
    final_img = ImageEnhance.Brightness(final_img).enhance(bright)
    return final_img


def enhance_image(path: str, sharp: float, contrast: float, bright: float) -> Image.Image:
    img = Image.open(path)
    return enhance_image_pil_model(img, sharp, contrast, bright)


def enhance_image_pil_only(img, sharp, contrast, bright):
    out = img.copy()
    out = ImageEnhance.Sharpness(out).enhance(sharp)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    out = ImageEnhance.Brightness(out).enhance(bright)
    return out


# =========================
# VIT-MIRAM MODULE (FOR VISUALIZATION ONLY)
# =========================
BASELINE_PATH = "best_generator_baseline.pth"
SSL_PATH = "best_generator.pth"  # must be different from baseline

class ChannelAttention(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, c // r, 1)
        self.fc2 = nn.Conv2d(c // r, c, 1)

    def forward(self, x):
        y = self.pool(x)
        y = F.relu(self.fc1(y))
        return x * torch.sigmoid(self.fc2(y))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        y = torch.cat(
            [torch.max(x, 1, keepdim=True)[0],
             torch.mean(x, 1, keepdim=True)],
            1,
        )
        return x * torch.sigmoid(self.conv(y))


class MIRAMBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = F.relu(self.c1(x))
        out = self.c2(out)
        out = self.ca(out)
        out = self.sa(out)
        return out + x


class UpsampleBlock(nn.Module):
    def __init__(self, c, scale=2):
        super().__init__()
        self.c = nn.Conv2d(c, c * scale**2, 3, padding=1)
        self.ps = nn.PixelShuffle(scale)
        self.a = nn.PReLU()

    def forward(self, x):
        return self.a(self.ps(self.c(x)))


class GeneratorViT_SR(nn.Module):
    def __init__(self, in_c=3, embed_dim=768, decoder_dim=256, scale=4):
        super().__init__()
        self.encoder = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0
        )
        self.resize = transforms.Resize((224, 224), antialias=True)
        self.feature_conv = nn.Linear(embed_dim, decoder_dim)
        self.miram = MIRAMBlock(decoder_dim)
        self.tail = nn.Sequential(
            UpsampleBlock(decoder_dim, 2),
            UpsampleBlock(decoder_dim, 2),
            UpsampleBlock(decoder_dim, 2),
            UpsampleBlock(decoder_dim, 2),
            nn.Conv2d(decoder_dim, in_c, 9, padding=4),
        )

    def forward(self, x):
        x_resized = self.resize(x)
        # B, 197, 768 => drop class token
        features = self.encoder.forward_features(x_resized)[:, 1:, :]
        features = self.feature_conv(features)
        features = features.permute(0, 2, 1).reshape(x.shape[0], -1, 14, 14)
        features = self.miram(features)
        return self.tail(features)


def load_vit_model(path: str) -> nn.Module:
    model = GeneratorViT_SR().to(device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# =========================
# TKINTER GUI
# =========================
root = Tk()
root.title("AI Image Enhancer")
root.geometry("1000x800")
root.resizable(True, True)

selected_images = []
output_folder = ""

preview_window = None
preview_canvas = None
preview_tk_image = None

example_image_path = None
example_original_pil = None

preview_job = None

def schedule_preview_update():
    global preview_job
    if preview_job is not None:
        root.after_cancel(preview_job)
    preview_job = root.after(200, update_live_preview)  # 200 ms debounce

def ensure_preview_window():
    global preview_window, preview_canvas
    if preview_window is None or not preview_window.winfo_exists():
        preview_window = Toplevel(root)
        preview_window.title("AI Preview")
        preview_window.geometry("600x600")
        preview_window.resizable(True, True)
        preview_window.bind("<Configure>", on_preview_resize)

        preview_canvas = Label(preview_window)
        preview_canvas.pack(expand=True, fill=BOTH)


def get_preview_box_from_window():
    if preview_window is None or not preview_window.winfo_exists():
        return 600, 600
    win_w = preview_window.winfo_width()
    win_h = preview_window.winfo_height()
    if win_w < 200:
        win_w = 600
    if win_h < 200:
        win_h = 600
    return int(win_w * 0.95), int(win_h * 0.9)


def on_preview_resize(event):
    if example_original_pil is not None:
        update_live_preview()


def select_images():
    global selected_images
    selected_images = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp")]
    )
    img_list.delete(0, END)
    for img in selected_images:
        img_list.insert(END, os.path.basename(img))
    if len(selected_images) == 1:
        show_preview(selected_images[0])


def select_output():
    global output_folder
    output_folder = filedialog.askdirectory(title="Select Output Folder")
    output_label.config(text=f"Output: {output_folder}")


def show_preview(path):
    global example_original_pil
    img = Image.open(path)
    example_original_pil = img
    update_live_preview()


def select_example_image():
    global example_image_path, example_original_pil
    path = filedialog.askopenfilename(
        title="Select Example Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp")]
    )
    if path:
        example_image_path = path
        example_original_pil = Image.open(path)
        update_live_preview()


def get_scaled_size(orig_width, orig_height, max_width, max_height):
    scale = min(max_width / orig_width, max_height / orig_height)
    return int(orig_width * scale), int(orig_height * scale)


def get_current_params():
    sharp = slider_sharp.get()
    contrast = slider_contrast.get()
    bright = slider_bright.get()
    return sharp, contrast, bright


def get_preview_scale():
    return slider_preview_scale.get()


def update_live_preview(*args):
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

    # --- NEW: limit preview inference size ---
    # e.g. cap the shorter side to 512 px for the model
    PREVIEW_MAX = 512
    scale = min(PREVIEW_MAX / orig_w, PREVIEW_MAX / orig_h, 1.0)
    model_w = int(orig_w * scale)
    model_h = int(orig_h * scale)
    img_for_model = example_original_pil.resize((model_w, model_h), Image.BICUBIC)
    # ----------------------------------------

    try:
        result = enhance_image_pil_model(img_for_model, sharp, contrast, bright)
    except Exception as ex:
        print("Error in AI preview:", ex)
        result = enhance_image_pil_only(img_for_model, sharp, contrast, bright)

    result_display = result.resize((display_w, display_h), Image.BICUBIC)

    preview_tk_image = ImageTk.PhotoImage(result_display)
    preview_canvas.config(image=preview_tk_image)
    preview_canvas.image = preview_tk_image

def run_enhancement():
    if not selected_images:
        messagebox.showerror("Error", "No images selected.")
        return
    if not output_folder:
        messagebox.showerror("Error", "Please select output folder.")
        return

    sharp, contrast, bright = get_current_params()

    if current_model == "srgan2stage":
        model_prefix = "M1"
    elif current_model == "mri":
        model_prefix = "M2"
    else:
        model_prefix = "MX"

    for img_path in selected_images:
        img = Image.open(img_path)
        enhanced = enhance_image_pil_model(img, sharp, contrast, bright)
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_folder, f"{model_prefix}_enh_{filename}")
        enhanced.save(out_path)

    messagebox.showinfo("Done", "Enhancement complete!")


def reset_params():
    slider_sharp.set(DEFAULT_SHARP)
    slider_contrast.set(DEFAULT_CONTRAST)
    slider_bright.set(DEFAULT_BRIGHT)
    slider_preview_scale.set(DEFAULT_PREVIEW_SCALE)
    update_live_preview()


# =========================
# MODEL SWITCHING
# =========================
def set_model_srgan2stage():
    global current_model
    current_model = "srgan2stage"
    model_label.config(text="Model: EnhancedMIRAM + SRGAN (M1)", fg="blue")
    update_live_preview()


def set_model_mri():
    global current_model
    current_model = "mri"
    model_label.config(text="Model: MRI Enhancer (M2)", fg="darkgreen")
    update_live_preview()


# =========================
# VISUALIZATION BUTTON (ViT PIPELINE)
# =========================
def visualize_vit_results():
    if example_original_pil is None:
        messagebox.showerror("Error", "Please select a preview example image first.")
        return

    try:
        # 1. Prepare HR and LR images (same as visualize_results.py)
        hr_img = example_original_pil.convert("RGB")
        hr_img = hr_img.resize((224, 224), Image.BICUBIC)

        lr_img = hr_img.resize((224 // 4, 224 // 4), Image.BICUBIC)

        to_tensor = transforms.ToTensor()
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
        lr_tensor = norm(to_tensor(lr_img)).unsqueeze(0).to(device)

        # 2. Load baseline and SSL models
        model_baseline = load_vit_model(BASELINE_PATH)
        model_ssl = load_vit_model(SSL_PATH)

        # 3. Inference
        with torch.no_grad():
            out_baseline = model_baseline(lr_tensor)
            out_baseline = out_baseline.clamp(-1, 1).cpu().squeeze()
            out_baseline = (out_baseline * 0.5 + 0.5).permute(1, 2, 0).numpy()

            out_ssl = model_ssl(lr_tensor)
            out_ssl = out_ssl.clamp(-1, 1).cpu().squeeze()
            out_ssl = (out_ssl * 0.5 + 0.5).permute(1, 2, 0).numpy()

        # 4. Plot results (4 panels)
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))

        titles = [
            "Low Res Input (56x56)",
            "Baseline (From Scratch)",
            "Ours (SSL Pre-trained)",
            "Ground Truth (High Res)",
        ]

        lr_display = lr_img.resize((224, 224), Image.NEAREST)
        images = [lr_display, out_baseline, out_ssl, hr_img]

        for i, ax in enumerate(axs):
            ax.imshow(images[i])
            ax.set_title(titles[i], fontsize=16)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig("result_comparison_full.png", dpi=300)
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"Visualization failed: {e}")


# =========================
# BUILD GUI LAYOUT
# =========================
Label(root, text="AI Image Enhancement Tool", font=("Arial", 22, "bold")).pack(pady=10)

model_frame = Frame(root)
model_frame.pack(pady=5)

Button(model_frame, text="M1: MIRAM+SRGAN", width=18, command=set_model_srgan2stage).grid(row=0, column=0, padx=5)
Button(model_frame, text="M2: MRI Enhancer", width=18, command=set_model_mri).grid(row=0, column=1, padx=5)

model_label = Label(root, text="Model: MIRAM+SRGAN (M1)", font=("Arial", 10), fg="blue")
model_label.pack(pady=5)

btn_frame = Frame(root)
btn_frame.pack()

Button(btn_frame, text="Select Images", width=20, command=select_images).grid(row=0, column=0, padx=10)
Button(btn_frame, text="Select Output Folder", width=20, command=select_output).grid(row=0, column=1, padx=10)

output_label = Label(root, text="Output: Not selected", font=("Arial", 10))
output_label.pack(pady=5)

img_list = Listbox(root, width=60, height=8)
img_list.pack(pady=5)

Button(root, text="Select Preview Example", width=25, command=select_example_image).pack(pady=10)

slider_frame = Frame(root)
slider_frame.pack(pady=10)

Label(slider_frame, text="Sharpness").grid(row=0, column=0, sticky="w")
slider_sharp = Scale(
    slider_frame,
    from_=0.0,
    to=50.0,
    orient=HORIZONTAL,
    resolution=0.1,
    length=250,
    command=lambda v: update_live_preview()
)
slider_sharp.set(DEFAULT_SHARP)
slider_sharp.grid(row=0, column=1, padx=10)

Label(slider_frame, text="Contrast").grid(row=1, column=0, sticky="w")
slider_contrast = Scale(
    slider_frame,
    from_=0.0,
    to=4.0,
    orient=HORIZONTAL,
    resolution=0.01,
    length=250,
    command=lambda v: update_live_preview()
)
slider_contrast.set(DEFAULT_CONTRAST)
slider_contrast.grid(row=1, column=1, padx=10)

Label(slider_frame, text="Brightness").grid(row=2, column=0, sticky="w")
slider_bright = Scale(
    slider_frame,
    from_=0.0,
    to=4.0,
    orient=HORIZONTAL,
    resolution=0.01,
    length=250,
    command=lambda v: update_live_preview()
)
slider_bright.set(DEFAULT_BRIGHT)
slider_bright.grid(row=2, column=1, padx=10)

Label(slider_frame, text="Preview Size (×)").grid(row=3, column=0, sticky="w")
slider_preview_scale = Scale(
    slider_frame,
    from_=0.2,
    to=2.0,
    orient=HORIZONTAL,
    resolution=0.05,
    length=250,
    command=lambda v: update_live_preview()
)
slider_preview_scale.set(DEFAULT_PREVIEW_SCALE)
slider_preview_scale.grid(row=3, column=1, padx=10)

Button(slider_frame, text="Reset to Default", command=reset_params).grid(row=4, column=0, columnspan=2, pady=5)

Button(root, text="Start Enhancement", font=("Arial", 14), bg="#4CAF50", fg="white", command=run_enhancement).pack(pady=15)

Button(root,
       text="Visualize ViT Results",
       font=("Arial", 12),
       bg="#2196F3",
       fg="white",
       command=visualize_vit_results).pack(pady=5)

root.mainloop()
