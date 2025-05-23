import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.MSTCN import MSTCN

class Frame_Feature_Cut(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 feature_dim,
                 num_classes):
        super(Frame_Feature_Cut, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 1x1
        )
        self.linear = nn.Linear(out_channels, feature_dim)
        self.mstcn = MSTCN(in_channels, out_channels, num_classes)
    
    def forward(self, video):
        B, T, C, H, W = video.shape                # x: [B, T, C, H, W]
        x = video.view(B * T, C, H, W)             # [B*T, C, H, W]
        x = self.cnn(x).view(B * T, -1)        # [B*T, 64]
        x = self.fc(x)                         # [B*T, feature_dim]
        x = x.view(B, T, -1).permute(0, 2, 1)  # [B, feature_dim, T]
        x = self.mstcn(x)

        return x
