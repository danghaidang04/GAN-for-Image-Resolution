import torch
from PIL import Image
from torchvision import transforms
from models import Generator


print("🚀 Starting inference...")
# --------------------
# Device
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

# --------------------
# Load Generator
# --------------------
netG = Generator().to(device)
netG.load_state_dict(torch.load("generator.pth", map_location=device))
netG.eval()  # chuyển sang eval mode

# --------------------
# Transform cho ảnh LR
# --------------------
transform_lr = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Transform ngược để hiển thị/save ảnh HR
transform_hr = transforms.Compose([
    transforms.Normalize([-1, -1, -1], [2,2,2]),  # undo normalization [-1,1] -> [0,1]
    transforms.ToPILImage()
])

# --------------------
# Load ảnh LR
# --------------------
img_path = "data/low/0.png"  # thay bằng ảnh bạn muốn inference
lr_img = Image.open(img_path).convert("RGB")
lr_tensor = transform_lr(lr_img).unsqueeze(0).to(device)  # thêm batch dim

# --------------------
# Inference
# --------------------
with torch.no_grad():
    hr_tensor = netG(lr_tensor)

# Chuyển tensor về ảnh
hr_img = transform_hr(hr_tensor.squeeze(0).cpu())
hr_img.save("sample_hr.png")
print("✅ Done! HR image saved as sample_hr.png")
