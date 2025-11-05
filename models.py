import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Input: 3x64x64
    Output: 3x256x256
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # input 3x64x64
            nn.Conv2d(3, 64, 3, 1, 1),        # 64x64
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, 4, 2, 1), # 64 -> 128
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1), # 128 -> 256
            nn.ReLU(True),

            nn.Conv2d(32, 3, 3, 1, 1),           # giữ 256x256
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """
    Input: 3x256x256
    Output: scalar real/fake
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),      # 256 -> 128
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, 2, 1),    # 128 -> 64
            nn.LeakyReLU(0.2, True)
        )
        self.fc = nn.Linear(128*64*64, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)





# import torch
# import torch.nn as nn

# # ----------------------------
# # Residual Block (ResNet style)
# # ----------------------------
# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
#         self.bn1 = nn.BatchNorm2d(channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
#         self.bn2 = nn.BatchNorm2d(channels)

#     def forward(self, x):
#         residual = x
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         return out + residual  # skip connection


# # ----------------------------
# # Generator (simple SRGAN style)
# # ----------------------------
# class Generator(nn.Module):
#     def __init__(self, num_blocks=4, upscale_factor=4):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 64, 9, 1, 4)
#         self.relu = nn.ReLU(inplace=True)

#         # residual blocks
#         blocks = [ResidualBlock(64) for _ in range(num_blocks)]
#         self.res_blocks = nn.Sequential(*blocks)

#         # upsample blocks
#         up_blocks = []
#         for _ in range(int(upscale_factor / 2)):
#             up_blocks += [
#                 nn.Conv2d(64, 256, 3, 1, 1),
#                 nn.PixelShuffle(2),
#                 nn.ReLU(inplace=True)
#             ]
#         self.upsample = nn.Sequential(*up_blocks)

#         # final output
#         self.conv_out = nn.Conv2d(64, 3, 9, 1, 4)

#     def forward(self, x):
#         out1 = self.relu(self.conv1(x))
#         out = self.res_blocks(out1)
#         out = self.upsample(out + out1)  # skip từ đầu -> sau residual
#         out = torch.tanh(self.conv_out(out))
#         return out


# # ----------------------------
# # Discriminator (simplified)
# # ----------------------------
# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 64, 3, 1, 1),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(64, 128, 3, 2, 1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(128, 256, 3, 2, 1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(256, 512, 3, 2, 1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(512, 1, 1)
#         )

#     def forward(self, x):
#         return torch.sigmoid(self.model(x).view(x.size(0), -1))
