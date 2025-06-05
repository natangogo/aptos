import torch
import torch.nn as nn

# <参考>
# https://pystyle.info/pytorch-yolov3/
# https://www.mdpi.com/2412-3811/6/9/134
# https://blog.csdn.net/leviopku/article/details/82660381

class Residual_Block(nn.Module):
    def __init__(self, channels, n_blocks, mid_channels=None):
        if mid_channels is None:
            mid_channels = channels // 2
        super(Residual_Block, self).__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(n_blocks)
        ])
    
    def forward(self, x):
        for block in self.blocks:
            h = block(x)
            x = x + h
        return x

class DarkNet53(nn.Module):
    def __init__(self,
                 channels):
        super(DarkNet53, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels*2, kernel_size=3, padding=1, stride=2)
        self.res_block1 = Residual_Block(channels=channels*2, n_blocks=1)
        self.conv3 = nn.Conv2d(in_channels=channels*2, out_channels=channels*4, kernel_size=3, padding=1, stride=2)
        self.res_block2 = Residual_Block(channels=channels*4, n_blocks=4)
        self.conv4 = nn.Conv2d(in_channels=channels*4, out_channels=channels*8, kernel_size=3, padding=1, stride=2)
        self.res_block3 = Residual_Block(channels=channels*8, n_blocks=8)
        self.conv5 = nn.Conv2d(in_channels=channels*8, out_channels=channels*16, kernel_size=3, padding=1, stride=2)
        self.res_block4 = Residual_Block(channels=channels*16, n_blocks=8)
        self.conv6 = nn.Conv2d(in_channels=channels*16, out_channels=channels*32, kernel_size=3, padding=1, stride=2)
        self.res_block5 = Residual_Block(channels=channels*32, n_blocks=4)
        
        # self.conv7 = nn.Conv2d(in_channels=1024, out_channels=1000, kernel_size=1, padding=0, stride=1)
        # self.global_average_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res_block1(x)
        x = self.conv3(x)
        x = self.res_block2(x)
        x = self.conv4(x)
        x1 = self.res_block3(x)
        x = self.conv5(x1)
        x2 = self.res_block4(x)
        x = self.conv6(x2)
        x3 = self.res_block5(x)

        return x1, x2, x3

class Block_A(nn.Module):
    def __init__(self,
                 channels):
        super(Block_A, self).__init__()
        self.block_a = nn.Sequential(
            nn.Conv2d(in_channels=channels,    out_channels=channels//2,   kernel_size=1, padding=0, stride=1), #1024->512 or 512->256 or 256->128
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels//2, out_channels=channels,      kernel_size=3, padding=1, stride=1), #512->1024 or 256->512 or 128->256
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels,    out_channels=channels//2,   kernel_size=1, padding=0, stride=1), #1024->512 or 512->256 or 256->128
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels//2, out_channels=channels,      kernel_size=3, padding=1, stride=1),#512->1024 or 256->512 or 128->256
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels,    out_channels=channels//2,   kernel_size=1, padding=0, stride=1), #1024->512 or 512->256 or 256->128
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True)
        )
        
        
    
    def forward(self, x):
        x = self.block_a(x)

        return x

class Block_B(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels):
        super(Block_B, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1, stride=1)
        self.norm = nn.BatchNorm2d(in_channels*2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels*2, out_channels, kernel_size=1, padding=0, stride=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x

class Block_C(nn.Module):
    def __init__(self, channels):
        super(Block_C, self).__init__()
        self.conv = nn.Conv2d(channels, channels//2, kernel_size=1, padding=0, stride=1)
        self.norm = nn.BatchNorm2d(channels//2)
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.up(x)

        return x

class YOLOv3(nn.Module):
    def __init__(self, 
                 channels:int=32,
                 num_classes:int=2):#YOLO_C = (クラス数 + 5) * Anchor Box の数(YOLO_Layerが必要とするチャンネル数のこと).Anchor Boxは通常3
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        #Block_b1はDarkNet53の最深部、Block_b2は中間層、Block_b3は浅い層
        self.darknet53 = DarkNet53(channels)                                                                                    #3->1024
        self.block_A1 = Block_A(channels*32)                                                                                    #1024->512
        self.block_B1 = Block_B(in_channels=channels*16, out_channels=(num_classes+5)*3)                                        #512->1024->120
        # self.yololayer
        self.block_C1 = Block_C(channels*16)                                                                                    #512->256

        self.block_A2 = Block_A(channels*16 + channels*8)                                                                       # 
        self.block_B2 = Block_B(in_channels=(channels*16 + channels*8)//2, out_channels=(num_classes+5)*3)                      #
        # self.yololayer
        self.block_C2 = Block_C((channels*16 + channels*8)//2)                                                                  # 

        self.block_A3 = Block_A((channels*16 + channels*8)//4 + channels*8)                                                     # 
        self.block_B3 = Block_B(in_channels=((channels*16 + channels*8)//4 + channels*8)//2, out_channels=(num_classes+5)*3)    #
    
    def forward(self, x):
        x_res8, x_res8_2, x_res4 = self.darknet53(x)#x_res8:256, x_res8_2:512, x_res4:1024
        x_a = self.block_A1(x_res4)                 #1024->512
        y1 = self.block_B1(x_a)                     #512->1024->120
        x_c = self.block_C1(x_a)                    #512->256
        x_out1 = torch.cat([x_res8_2, x_c], dim=1)  #256+512=768

        x_a2 = self.block_A2(x_out1)                #768->384
        y2 = self.block_B2(x_a2)                    #384->768->120
        x_c2 = self.block_C2(x_a2)                  #384->192
        x_out2 = torch.cat([x_res8, x_c2], dim=1)   #256 + 192 = 448

        x_a3 = self.block_A3(x_out2)                #448->224
        y3 = self.block_B3(x_a3)                    #224->448->120

        return x_a, x_a2, x_a3