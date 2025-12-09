# losses.py - DUAL SCALE PATCH LOSS + METRICS
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class MIRAMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def patchify(self, imgs, p_size):
        """
        imgs: (N, 1, H, W)
        x: (N, L, p_size**2 * 1)
        """
        N, C, H, W = imgs.shape
        h = w = H // p_size
        x = imgs.reshape(shape=(N, 1, h, p_size, w, p_size))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(N, h * w, p_size**2 * 1))
        return x

    def forward(self, pred_fine, pred_coarse, target_fine, target_coarse, mask):
        # 1. Patchify targets
        target_f = self.patchify(target_fine, PATCH_SIZE)
        
        # Coarse patch size is scaled
        p_coarse = int(PATCH_SIZE * SCALE_COARSE)
        target_c = self.patchify(target_coarse, p_coarse)
        
        # 2. Compute MSE Loss
        loss_f = (pred_fine - target_f) ** 2
        loss_f = loss_f.mean(dim=-1)  # [N, L], mean loss per patch
        
        loss_c = (pred_coarse - target_c) ** 2
        loss_c = loss_c.mean(dim=-1)

        # 3. Apply Mask (Only compute loss on masked patches)
        # sum of loss on masked patches / number of masked patches
        loss_f = (loss_f * mask).sum() / (mask.sum() + 1e-8)
        loss_c = (loss_c * mask).sum() / (mask.sum() + 1e-8)
        
        total_loss = (loss_f * LAMBDA_FINE) + (loss_c * LAMBDA_COARSE)
        
        return total_loss, loss_f, loss_c

# ==========================================
# METRICS (Re-added for Evaluation)
# ==========================================
def psnr(x, y):
    mse = F.mse_loss(x, y)
    if mse < 1e-10: return torch.tensor(100.0).to(x.device)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def ssim(x, y):
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x2 = ((x - mu_x)**2).mean()
    sigma_y2 = ((y - mu_y)**2).mean()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    
    k1, k2 = 0.01, 0.03
    L = 1.0 # Max value
    c1 = (k1 * L)**2
    c2 = (k2 * L)**2
    
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return num / (den + 1e-8)