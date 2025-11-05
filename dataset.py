import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from config import LR_SIZE, HR_SIZE

class PairedDataset(Dataset):
    def __init__(self, low_dir, high_dir, lr_size=LR_SIZE, hr_size=HR_SIZE):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.files = sorted(os.listdir(low_dir))

        self.to_tensor_lr = T.Compose([
            T.Resize((lr_size, lr_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        self.to_tensor_hr = T.Compose([
            T.Resize((hr_size, hr_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        lr = Image.open(os.path.join(self.low_dir, name)).convert('RGB')
        hr = Image.open(os.path.join(self.high_dir, name)).convert('RGB')
        return self.to_tensor_lr(lr), self.to_tensor_hr(hr)
