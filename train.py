# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from dataset import PairedDataset
# from models import Generator, Discriminator
# from config import NUM_EPOCHS, BATCH_SIZE, LR_SIZE, HR_SIZE, LEARNING_RATE, DATA_LOW_DIR, DATA_HIGH_DIR, NUM_WORKERS

# # Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("🚀 Using device:", device)

# # Dataset & DataLoader
# dataset = PairedDataset(DATA_LOW_DIR, DATA_HIGH_DIR, lr_size=LR_SIZE, hr_size=HR_SIZE)
# loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

# # Models
# netG = Generator().to(device)
# netD = Discriminator().to(device)

# # netG = Generator(num_blocks=4, upscale_factor=4).to(device)
# # netD = Discriminator().to(device)

# # Load pre-trained models if available
# load_pretrained = input("Do you want to load pre-trained models? (y/n): ").strip().lower()
# if load_pretrained == 'y':
#     try:
#         netG.load_state_dict(torch.load("generator.pth", map_location=device))
#         netD.load_state_dict(torch.load("discriminator.pth", map_location=device))
#         print("✅ Pre-trained models loaded.")
#     except FileNotFoundError:
#         print("⚠️ Pre-trained model files not found. Training from scratch.")

# # Loss & Optimizer
# criterion = nn.BCELoss()
# optimizerG = torch.optim.Adam(netG.parameters(), lr=LEARNING_RATE)
# optimizerD = torch.optim.Adam(netD.parameters(), lr=LEARNING_RATE)

# # Training loop
# for epoch in range(NUM_EPOCHS):
#     for lr_imgs, hr_imgs in tqdm(loader):
#         lr_imgs, hr_imgs = lr_imgs.to(device, non_blocking=True), hr_imgs.to(device, non_blocking=True)

#         # Train Discriminator
#         real = torch.ones(hr_imgs.size(0), 1, device=device)
#         fake = torch.zeros(hr_imgs.size(0), 1, device=device)

#         optimizerD.zero_grad()
#         output_real = netD(hr_imgs)
#         lossD_real = criterion(output_real, real)

#         fake_imgs = netG(lr_imgs)
#         output_fake = netD(fake_imgs.detach())
#         lossD_fake = criterion(output_fake, fake)

#         lossD = (lossD_real + lossD_fake) / 2
#         lossD.backward()
#         optimizerD.step()

#         # Train Generator
#         optimizerG.zero_grad()
#         output_fake = netD(fake_imgs)
#         lossG = criterion(output_fake, real)
#         lossG.backward()
#         optimizerG.step()

#     print(f"Epoch {epoch+1} | LossD: {lossD.item():.4f} | LossG: {lossG.item():.4f}")

# # Save model
# torch.save(netG.state_dict(), "generator.pth")
# torch.save(netD.state_dict(), "discriminator.pth")
# print("✅ Training done, model saved to generator.pth and discriminator.pth")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import PairedDataset
from models import Generator, Discriminator
from config import (
    NUM_EPOCHS, BATCH_SIZE, LR_SIZE, HR_SIZE, 
    LEARNING_RATE, DATA_LOW_DIR, DATA_HIGH_DIR, NUM_WORKERS
)

# ============================
# ⚙️ Setup device
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

# ============================
# 📂 Dataset & DataLoader
# ============================
dataset = PairedDataset(DATA_LOW_DIR, DATA_HIGH_DIR, lr_size=LR_SIZE, hr_size=HR_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

# ============================
# 🧠 Initialize models
# ============================
netG = Generator(num_blocks=4, upscale_factor=HR_SIZE // LR_SIZE).to(device)
netD = Discriminator().to(device)

# ============================
# 📥 Load pretrained weights (optional)
# ============================
load_pretrained = input("Do you want to load pre-trained models? (y/n): ").strip().lower()
if load_pretrained == 'y':
    try:
        netG.load_state_dict(torch.load("generator.pth", map_location=device))
        netD.load_state_dict(torch.load("discriminator.pth", map_location=device))
        print("✅ Pre-trained models loaded.")
    except FileNotFoundError:
        print("⚠️ Pre-trained model files not found. Training from scratch.")

# ============================
# ⚖️ Loss functions
# ============================
adversarial_criterion = nn.BCELoss()
content_criterion = nn.MSELoss()  # dùng MSE thay cho perceptual loss

# ============================
# ⚡ Optimizers
# ============================
optimizerG = torch.optim.Adam(netG.parameters(), lr=LEARNING_RATE)
optimizerD = torch.optim.Adam(netD.parameters(), lr=LEARNING_RATE)

# ============================
# 🧩 Training loop
# ============================
for epoch in range(NUM_EPOCHS):
    netG.train()
    netD.train()

    epoch_lossG, epoch_lossD = 0, 0

    for lr_imgs, hr_imgs in tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

        real_labels = torch.ones(hr_imgs.size(0), 1, device=device)
        fake_labels = torch.zeros(hr_imgs.size(0), 1, device=device)

        # ----------------------------
        # Train Discriminator
        # ----------------------------
        optimizerD.zero_grad()

        real_outputs = netD(hr_imgs)
        fake_imgs = netG(lr_imgs)
        fake_outputs = netD(fake_imgs.detach())

        lossD_real = adversarial_criterion(real_outputs, real_labels)
        lossD_fake = adversarial_criterion(fake_outputs, fake_labels)
        lossD = (lossD_real + lossD_fake) / 2

        lossD.backward()
        optimizerD.step()

        # ----------------------------
        # Train Generator
        # ----------------------------
        optimizerG.zero_grad()

        fake_outputs = netD(fake_imgs)
        adv_loss = adversarial_criterion(fake_outputs, real_labels)
        content_loss = content_criterion(fake_imgs, hr_imgs)

        # tổng hợp loss generator (SRGAN-style)
        lossG = content_loss + 1e-3 * adv_loss

        lossG.backward()
        optimizerG.step()

        epoch_lossD += lossD.item()
        epoch_lossG += lossG.item()

    print(f"✅ Epoch [{epoch+1}/{NUM_EPOCHS}] | LossD: {epoch_lossD/len(loader):.4f} | LossG: {epoch_lossG/len(loader):.4f}")

# ============================
# 💾 Save models
# ============================
torch.save(netG.state_dict(), "generator.pth")
torch.save(netD.state_dict(), "discriminator.pth")
print("🎉 Training complete! Models saved to generator.pth & discriminator.pth.")
