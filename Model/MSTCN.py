import torch
import torch.nn as nn
import torch.nn.functional as F

class Separable_Conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super(Separable_Conv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, dilation=1, padding=(kernel_size-1)//2*1, groups=in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size, dilation=2, padding=(kernel_size-1)//2*2, groups=in_channels)
        self.conv3 = nn.Conv1d(in_channels, in_channels, kernel_size, dilation=4, padding=(kernel_size-1)//2*4, groups=in_channels)
        self.conv4 = nn.Conv1d(in_channels, in_channels, kernel_size, dilation=5, padding=(kernel_size-1)//2*5, groups=in_channels)
        self.conv_second1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv_second2 = nn.Conv1d(in_channels, out_channels, kernel_size=1) 
        self.conv_second3 = nn.Conv1d(in_channels, out_channels, kernel_size=1) 
        self.conv_second4 = nn.Conv1d(in_channels, out_channels, kernel_size=1) 
            #元の画像とサイズを合わせるために、padding=(kernel_size-1)//2*d
    
    def forward(self, x):
        x1 = self.conv1(x)
        out1 = self.conv_second1(x1)
        x2 = self.conv2(x)
        out2 = self.conv_second2(x2)
        x3 = self.conv3(x)
        out3 = self.conv_second3(x3)
        x4 = self.conv4(x)
        out4 = self.conv_second4(x4)
        #これらの出力をconcatすることで、様々な間引き率の結果から全体を把握できる
        x_out = torch.cat([out1, out2, out3, out4], dim=1)

        return x_out

class Conv_Unit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(Conv_Unit, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv_k8 = Separable_Conv(in_channels, out_channels, kernel_size=8)
        self.conv_k16 = Separable_Conv(in_channels, out_channels, kernel_size=16)
        self.conv_k20 = Separable_Conv(in_channels, out_channels, kernel_size=20)
    
    def forward(self, x):
        x = self.conv(x)
        out1 = self.conv_k8(x)
        out2 = self.conv_k16(x)
        out3 = self.conv_k20(x)
        out = torch.cat([out1, out2, out3], dim=1)

        return out

class Residual_Connection(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(Residual_Connection, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)

        return x

class MSD_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(MSD_Block, self).__init__()
        self.CU = Conv_Unit(in_channels, out_channels)
        self.average_pool = nn.AvgPool1d(kernel_size=3, stride=3)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU() 
        self.RC = Residual_Connection(in_channels, out_channels)
    
    def forward(self, x):
        out1 = self.CU(x)

        out2 = self.average_pool(x)
        out2 = self.conv(out2)

        out = torch.cat([out1, out2], dim=1)
        out = self.norm(out)
        out_left = self.relu(out)

        out_right = self.RC(x)


        return out_left + out_right

class MSTCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_classes):
        super(MSTCN, self).__init__()
        self.MSB_blocks = nn.Sequential(
            *[MSD_Block(in_channels if i == 0 else out_channels, out_channels) for i in range(7)]
        )
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(out_channels, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.MSB_blocks(x)
        out = self.global_average_pool(out)
        out = self.classifier(out.squeeze(-1))
        out = self.softmax(out)

        return out


