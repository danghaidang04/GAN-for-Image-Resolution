import os
import torch
from torchvision import transforms
from PIL import Image
from models import Generator
from config import LR_SIZE, HR_SIZE
from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
from config import DATA_LOW_VAL, DATA_HIGH_VAL

# --------------------
# Device setup
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

# --------------------
# Load Generator
# --------------------
netG = Generator().to(device)
netG.load_state_dict(torch.load("generator.pth", map_location=device))
netG.eval()

# --------------------
# Prepare transforms
# --------------------
to_tensor_lr = transforms.Compose([
    transforms.Resize((LR_SIZE, LR_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
to_tensor_hr = transforms.Compose([
    transforms.Resize((HR_SIZE, HR_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --------------------
# LPIPS metric
# --------------------
lpips_model = lpips.LPIPS(net='alex').to(device)

# --------------------
# Paths
# --------------------
lr_dir = DATA_LOW_VAL
hr_dir = DATA_HIGH_VAL
lr_files = sorted(os.listdir(lr_dir))
hr_files = sorted(os.listdir(hr_dir))

# --------------------
# Evaluation
# --------------------
psnr_vals, ssim_vals, lpips_vals = [], [], []

for fname in tqdm(lr_files, desc="Evaluating..."):
    lr_path = os.path.join(lr_dir, fname)
    hr_path = os.path.join(hr_dir, fname)
    if not os.path.exists(hr_path):
        continue

    # Load and preprocess
    lr_img = Image.open(lr_path).convert("RGB")
    hr_img = Image.open(hr_path).convert("RGB")

    lr_tensor = to_tensor_lr(lr_img).unsqueeze(0).to(device)
    hr_tensor = to_tensor_hr(hr_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        fake_hr = netG(lr_tensor)

    # Convert to numpy for PSNR/SSIM
    fake_np = fake_hr.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    hr_np = hr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    fake_np = (fake_np + 1) / 2
    hr_np = (hr_np + 1) / 2

    psnr_vals.append(psnr(hr_np, fake_np, data_range=1.0))
    ssim_vals.append(ssim(hr_np, fake_np, channel_axis=2, data_range=1.0))
    lpips_vals.append(lpips_model(hr_tensor, fake_hr).item())

# --------------------
# Compute statistics
# --------------------
def summarize(name, values):
    return {
        "Metric": name,
        "Mean": np.mean(values),
        "Std": np.std(values),
        "Variance": np.var(values)
    }

summary = [
    summarize("PSNR", psnr_vals),
    summarize("SSIM", ssim_vals),
    summarize("LPIPS", lpips_vals)
]

# --------------------
# Print results
# --------------------
print("\n📊 Evaluation Summary:")
print("{:<10} {:>10} {:>10} {:>10}".format("Metric", "Mean", "Std", "Variance"))
for s in summary:
    print("{:<10} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        s["Metric"], s["Mean"], s["Std"], s["Variance"]
    ))

# --------------------
# Save results to CSV
# --------------------
df = pd.DataFrame(summary)
output_path = "metrics_summary.csv"
df.to_csv(output_path, index=False)
print(f"\n💾 Results saved to {output_path}")

print("\n✅ Done.")
