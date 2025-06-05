import torch
import torch.nn as nn

# https://qiita.com/tchih11/items/377cbf9162e78a639958

class Residual_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 stride:int=1,
                 conv_flag=False):
        super(Residual_Block, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv_flag = conv_flag # conv_flag=Trueのとき、入力xと出力のチャネル数や空間サイズが異なるので、1x1convで合わせる
        self.plus_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_orig = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)

        if self.conv_flag:
            x_plus = self.plus_conv(x_orig)
            x_plus = self.norm(x_plus)
            
        else:
            x_plus = x_orig
        x += x_plus
        x = self.relu(x)

        return x

class Conv_layer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 block_num):
        super(Conv_layer, self).__init__()
        layers = []
        layers.append(Residual_Block(in_channels, out_channels, stride=stride, conv_flag=True))
        for i in range(block_num-1):
            layers.append(Residual_Block(out_channels, out_channels, stride=1, conv_flag=False))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        x = self.blocks(x)

        return  x #これを行うことで、毎回別の重みをもつ新しいブロックを追加できるようになり、学習の自由度が高くなる


class ResNet50(nn.Module):
    def __init__(self,
                 in_channels:int=3,
                 basic_channels:int=64,
                 num_classes:int=2):
        super(ResNet50, self).__init__()
        # convのkernelは本来7だが、画像サイズがとても小さいので仕方なく、kernel_size=3, stride=1, padding=1に。
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=basic_channels, kernel_size=3, stride=1, padding=1)# 3->64
        self.norm = nn.BatchNorm2d(num_features=basic_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.max_pool = nn.Identity()
        self.conv2_x = Conv_layer(in_channels=basic_channels, out_channels=basic_channels*4, stride=1, block_num=3)# 64->256
        self.conv3_x = Conv_layer(in_channels=basic_channels*4, out_channels=basic_channels*8, stride=2, block_num=4)# 256->512
        self.conv4_x = Conv_layer(in_channels=basic_channels*8, out_channels=basic_channels*16, stride=2, block_num=6)# 512->1024
        # self.conv5_x = Conv_layer(in_channels=basic_channels*16, out_channels=basic_channels*32, stride=2, block_num=3)# 1024->2048
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(basic_channels*16, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.avg_pool(x) # (B, C, H, W)->(B, C, 1, 1)
        x = torch.flatten(x, 1)# (B, C, 1, 1)->(B, C*1*1)
        x = self.fc(x) # Linearに入力するときは(B, feature)の型にする必要有なので直前でreshape
        return x

#うまくいくかを確かめ
# if __name__ == "__main__":
#     model = ResNet50(in_channels=3, num_classes=10)
#     dummy_input = torch.randn(4, 3, 224, 224)  # バッチサイズ4、RGB画像（224x224）
#     output = model(dummy_input)
#     print("出力の形状:", output.shape)  # => torch.Size([4, 10])