# train.py - MAIN LOOP
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from config import *
from dataset import get_dataloaders
from models import GeneratorMIRAMSR, Discriminator
from losses import CharbonnierLoss, EdgeLoss, PerceptualLoss, adversarial_loss, psnr

def train():
    print(f"🚀 STARTING 4X SR TRAINING | Device: {DEVICE}")
    train_dl, val_dl, _ = get_dataloaders(DATASET_PATH, BATCH_SIZE, CROP_SIZE, SCALE_FACTOR)
    if not train_dl: return

    gen = GeneratorMIRAMSR(in_c=IMAGE_CHANNELS, scale=SCALE_FACTOR).to(DEVICE)
    disc = Discriminator(in_c=IMAGE_CHANNELS).to(DEVICE)
    
    opt_g = optim.Adam(gen.parameters(), lr=LEARNING_RATE_G, betas=(0.9, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=LEARNING_RATE_D, betas=(0.9, 0.999))
    
    crit_pix = CharbonnierLoss().to(DEVICE)
    crit_edge = EdgeLoss().to(DEVICE)
    crit_vgg = PerceptualLoss().to(DEVICE)
    
    writer = SummaryWriter(LOG_DIR)
    best_psnr = 0.0
    warmup_steps = int(N_EPOCHS * WARMUP_PERCENTAGE)
    
    for epoch in range(N_EPOCHS):
        gen.train()
        loss_log = {'g':0., 'd':0., 'pix':0.}
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        
        for lr, hr, mask in pbar:
            lr, hr, mask = lr.to(DEVICE), hr.to(DEVICE), mask.to(DEVICE)
            
            # Upscale Mask to HR size
            hr_mask = F.interpolate(mask, size=hr.shape[2:], mode='nearest')
            
            # --- Generator ---
            opt_g.zero_grad()
            sr = gen(lr)
            
            l_pix = crit_pix(sr, hr, hr_mask)
            
            if epoch < warmup_steps:
                loss_g = l_pix
            else:
                disc.train()
                l_adv = adversarial_loss(disc(sr), True)
                l_vgg = crit_vgg(sr, hr)
                l_edge = crit_edge(sr, hr, hr_mask)
                loss_g = LAMBDA_PIXEL*l_pix + LAMBDA_ADV*l_adv + LAMBDA_VGG*l_vgg + LAMBDA_EDGE*l_edge
                
            loss_g.backward()
            opt_g.step()
            
            # --- Discriminator ---
            loss_d = torch.tensor(0.)
            if epoch >= warmup_steps:
                opt_d.zero_grad()
                l_real = adversarial_loss(disc(hr), True)
                l_fake = adversarial_loss(disc(sr.detach()), False)
                loss_d = (l_real + l_fake) / 2
                loss_d.backward()
                opt_d.step()

            loss_log['g'] += loss_g.item()
            pbar.set_postfix_str(f"G:{loss_g.item():.4f}")

        # Validation
        gen.eval()
        val_psnr = 0
        with torch.no_grad():
            for lr, hr, mask in val_dl:
                lr, hr = lr.to(DEVICE), hr.to(DEVICE)
                sr = gen(lr).clamp(0, 1)
                val_psnr += psnr(sr, hr).item()
        
        avg_psnr = val_psnr / len(val_dl)
        print(f"   ✅ Val PSNR: {avg_psnr:.2f} dB")
        writer.add_scalar('PSNR', avg_psnr, epoch)
        
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(gen.state_dict(), BEST_MODEL_PATH)
            print("   💾 Best Model Saved")

if __name__ == "__main__": train()