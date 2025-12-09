# models.py - MIRAM GENERATOR + DISCRIMINATOR
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c // r, 1),
            nn.ReLU(),
            nn.Conv2d(c // r, c, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return x * self.fc(self.pool(x))

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        return x * self.sig(self.conv(torch.cat([avg_out, max_out], dim=1)))

class MIRAMBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention()
        self.relu = nn.PReLU()

    def forward(self, x):
        res = self.relu(self.conv1(x))
        res = self.conv2(res)
        res = self.ca(res)
        res = self.sa(res)
        return x + res

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.PReLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c)
        )
    def forward(self, x): return x + self.block(x)

class UpsampleBlock(nn.Module):
    def __init__(self, c, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(c, c * scale ** 2, 3, padding=1)
        self.ps = nn.PixelShuffle(scale)
        self.act = nn.PReLU()
    def forward(self, x): return self.act(self.ps(self.conv(x)))

class GeneratorMIRAMSR(nn.Module):
    def __init__(self, in_c=1, base=64, num_res=8, scale=4):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(in_c, base, 9, padding=4), nn.PReLU())
        
        body = []
        for i in range(num_res):
            body.append(ResidualBlock(base))
            if (i+1)%3 == 0: body.append(MIRAMBlock(base))
        self.body = nn.Sequential(*body)
        
        self.upsampler = nn.Sequential(
            UpsampleBlock(base, 2),
            UpsampleBlock(base, 2)
        )
        self.tail = nn.Conv2d(base, in_c, 9, padding=4)

    def forward(self, x):
        f = self.head(x)
        r = self.body(f)
        out = self.upsampler(f + r)
        return torch.clamp(self.tail(out), 0, 1)

class Discriminator(nn.Module):
    def __init__(self, in_c=1):
        super().__init__()
        def blk(i, o, s=1, n=True):
            l = [nn.Conv2d(i, o, 3, s, padding=1)]
            if n: l.append(nn.BatchNorm2d(o))
            l.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*l)
            
        self.net = nn.Sequential(
            blk(in_c, 64, n=False),
            blk(64, 64, 2),
            blk(64, 128),
            blk(128, 128, 2),
            blk(128, 256),
            blk(256, 256, 2),
            blk(256, 512),
            blk(512, 512, 2),
            nn.Conv2d(512, 1, 3, padding=1)
        )
    def forward(self, x): return self.net(x)