import torch
import torch.nn as nn
from Model.ResNet50 import ResNet50

class TS_Net(nn.Module):
    def __init__(self,
                 num_classes:int=9,
                 stride:int=2,
                 basic_channels:int=32):
        super(TS_Net, self).__init__()
        self.resnet50_instrument = ResNet50(in_channels=basic_channels*16, basic_channels=basic_channels*4, num_classes=num_classes)
        self.resnet50_pupil = ResNet50(in_channels=(basic_channels*16 + basic_channels*8)//2, basic_channels=basic_channels*4, num_classes=num_classes)
        self.resnet50_entire = ResNet50(in_channels=((basic_channels*16 + basic_channels*8)//4 + basic_channels*8)//2, basic_channels=basic_channels*2, num_classes=num_classes)
        self.fully_layer = nn.Linear(num_classes*3, num_classes)#2048*3=6144

    def forward(self, x_instrument, x_pupil, x_entire):
        x_instrument = self.resnet50_instrument(x_instrument)
        x_pupil = self.resnet50_pupil(x_pupil)
        x_entire = self.resnet50_entire(x_entire)
        concat = torch.cat([x_instrument, x_pupil, x_entire], dim=1)
        # out = self.fully_layer(concat)

        return concat

