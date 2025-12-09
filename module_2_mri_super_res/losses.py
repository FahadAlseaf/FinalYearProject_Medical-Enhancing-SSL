# losses.py - MASKED LOSSES & VGG
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights='VGG19_Weights.DEFAULT').features
        self.features = nn.Sequential(*vgg[:35]).eval()
        for p in self.features.parameters(): p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, sr, hr):
        # Repeat 1-channel to 3-channel for VGG
        if sr.size(1) == 1:
            sr = sr.repeat(1, 3, 1, 1)
            hr = hr.repeat(1, 3, 1, 1)
        sr = (sr - self.mean.to(sr.device)) / self.std.to(sr.device)
        hr = (hr - self.mean.to(sr.device)) / self.std.to(sr.device)
        return F.l1_loss(self.features(sr), self.features(hr))

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x, y, mask=None):
        loss = torch.sqrt((x - y)**2 + self.eps**2)
        if mask is not None:
            return torch.sum(loss * mask) / (torch.sum(mask) + 1e-8)
        return torch.mean(loss)

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]).float().view(1,1,3,3)
        self.kernel = k.repeat(1,1,1,1)
    def forward(self, sr, hr, mask=None):
        self.kernel = self.kernel.to(sr.device)
        s_e = F.conv2d(sr, self.kernel, padding=1)
        h_e = F.conv2d(hr, self.kernel, padding=1)
        loss = F.l1_loss(s_e, h_e, reduction='none')
        if mask is not None:
            return torch.sum(loss * mask) / (torch.sum(mask) + 1e-8)
        return torch.mean(loss)

def adversarial_loss(preds, real=True):
    target = torch.ones_like(preds) if real else torch.zeros_like(preds)
    return F.binary_cross_entropy_with_logits(preds, target)

def psnr(x, y):
    mse = F.mse_loss(x, y)
    if mse < 1e-10: return torch.tensor(100.)
    return 20 * torch.log10(1.0/torch.sqrt(mse))

def ssim(x, y):
    mu_x, mu_y = x.mean(), y.mean()
    s_x, s_y = x.std(), y.std()
    cov = torch.mean((x-mu_x)*(y-mu_y))
    return (2*mu_x*mu_y + 0.01)*(2*cov + 0.03) / ((mu_x**2+mu_y**2+0.01)*(s_x**2+s_y**2+0.03))