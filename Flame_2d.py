import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.MSTCN_img import MSTCN

class Frame_Feature_Cut(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_classes):
        super(Frame_Feature_Cut, self).__init__()
        self.MSD = MSTCN(in_channels, out_channels, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(num_classes, num_classes)
    
    def forward(self, x):
        B, T, C, H, W = x.shape                 # x: [B, T, C, H, W]
        x = x.view(B * T, C, H, W)              # → [B*T, C, H, W]
        out = self.MSD(x)
        out = out.view(B, T, -1).permute(0, 2, 1)

        out = self.avg_pool(out)
        out = self.linear(out)
        return out
