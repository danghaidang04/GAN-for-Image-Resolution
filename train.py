import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import PairedDataset
from models import Generator, Discriminator

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
dataset = PairedDataset("data/low", "data/high", lr_size=64, hr_size=256)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Models
netG = Generator().to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002)
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002)

for epoch in range(5):  # chỉ 5 epoch thử
    for lr, hr in tqdm(loader):
        lr, hr = lr.to(device), hr.to(device)

        # train D
        real = torch.ones(hr.size(0), 1, 1, 1, device=device)
        fake = torch.zeros(hr.size(0), 1, 1, 1, device=device)

        optimizerD.zero_grad()
        output_real = netD(hr)
        lossD_real = criterion(output_real, real)

        fake_img = netG(lr)
        output_fake = netD(fake_img.detach())
        lossD_fake = criterion(output_fake, fake)

        lossD = (lossD_real + lossD_fake) / 2
        lossD.backward()
        optimizerD.step()

        # train G
        optimizerG.zero_grad()
        output_fake = netD(fake_img)
        lossG = criterion(output_fake, real)
        lossG.backward()
        optimizerG.step()

    print(f"Epoch {epoch+1} | LossD: {lossD.item():.4f} | LossG: {lossG.item():.4f}")

torch.save(netG.state_dict(), "generator.pth")
print("✅ Training done, model saved to generator.pth")
