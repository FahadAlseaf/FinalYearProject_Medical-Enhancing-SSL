# models.py - MIRAM ARCHITECTURE WITH ATTENTION VISUALIZATION
import torch
import torch.nn as nn
import numpy as np
from config import *

# ==========================================
# TRANSFORMER COMPONENTS
# ==========================================
class PatchEmbed(nn.Module):
    """Split image into patches and embed them"""
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Block(nn.Module):
    """Transformer Block with Attention Access"""
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, return_attention=False):
        # We need to capture weights if return_attention is True
        attn_out, attn_weights = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=return_attention)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        
        if return_attention:
            return x, attn_weights
        return x

# ==========================================
# MIRAM MODEL
# ==========================================
class MIRAM(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. ENCODER
        self.patch_embed = PatchEmbed(IMG_SIZE, PATCH_SIZE, IN_CHANNELS, EMBED_DIM)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, EMBED_DIM))
        
        self.blocks = nn.ModuleList([
            Block(EMBED_DIM, NUM_HEADS, MLP_RATIO) for _ in range(DEPTH)
        ])
        self.norm = nn.LayerNorm(EMBED_DIM)
        
        # 2. DUAL DECODERS
        self.decoder_embed = nn.Linear(EMBED_DIM, DECODER_EMBED_DIM, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, DECODER_EMBED_DIM))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, DECODER_EMBED_DIM))
        
        self.decoder_blocks = nn.ModuleList([
            Block(DECODER_EMBED_DIM, DECODER_NUM_HEADS, MLP_RATIO) for _ in range(DECODER_DEPTH)
        ])
        self.decoder_norm = nn.LayerNorm(DECODER_EMBED_DIM)
        
        # Heads
        self.decoder_pred_fine = nn.Linear(DECODER_EMBED_DIM, PATCH_SIZE**2 * IN_CHANNELS, bias=True)
        coarse_p_size = int(PATCH_SIZE * SCALE_COARSE)
        self.decoder_pred_coarse = nn.Linear(DECODER_EMBED_DIM, coarse_p_size**2 * IN_CHANNELS, bias=True)
        
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.xavier_uniform_(self.decoder_pred_fine.weight)
        torch.nn.init.xavier_uniform_(self.decoder_pred_coarse.weight)

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, return_last_attention=False):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        last_attn = None
        for i, blk in enumerate(self.blocks):
            if i == len(self.blocks) - 1 and return_last_attention:
                x, last_attn = blk(x, return_attention=True)
            else:
                x = blk(x)
        x = self.norm(x)
        
        if return_last_attention:
            return x, mask, ids_restore, last_attn
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = x[:, 1:, :] 
        
        pred_fine = self.decoder_pred_fine(x)
        pred_coarse = self.decoder_pred_coarse(x)
        return pred_fine, pred_coarse

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred_fine, pred_coarse = self.forward_decoder(latent, ids_restore)
        return pred_fine, pred_coarse, mask