import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.MSTCN import MSTCN
from Model.TS_Net import TS_Net
from Model.YOLOv3_ import YOLOv3

class GL_MSTCN(nn.Module):
    def __init__(self, 
                 out_channels_pretreatment,
                 num_classes,
                 basic_channels:int=32):
        super(GL_MSTCN, self).__init__()
        self.yolo_v3 = YOLOv3(num_classes=num_classes)
        self.ts_net = TS_Net(basic_channels=basic_channels, num_classes=num_classes)
        self.mstcn = MSTCN(in_channels=num_classes*3, out_channels=out_channels_pretreatment, num_classes=num_classes)
        self.fully_layer = nn.Linear(out_channels_pretreatment, num_classes)
        self.conv = nn.Conv2d(out_channels_pretreatment, out_channels_pretreatment, kernel_size=1)
        #mainやtrainingに書くとき

    # forwardの中で、画像をフレームごとに分類し、フレーム数を出す場合と、forwardの引数にフレーム数を入れる場合の2パターンがある 
    def forward(self, frame, timestep, frame_rate):# video -> [B, T, C, H, W]
        if frame.dim() == 4:  # [B, C, H, W] → 単一フレームとして扱う
            frame = frame.unsqueeze(1)  # → [B, 1, C, H, W]

        x_features = []
        # フレーム数 = 秒数 × フレームレート（fps）
        frame_num = min(max(8, int(timestep * frame_rate)), 16)
        actual_frame_count = frame.shape[1]
        loop_count = min(frame_num, actual_frame_count)

        for t in range(loop_count):  # Tフレーム分ループ
            frame_t = frame[:, t, :, :, :]  #[B, C, H, W]
            x_instrument, x_pupil, x_entire = self.yolo_v3(frame_t)#[B, C, H, W]
            x_tsnet = self.ts_net(x_instrument, x_pupil, x_entire) # [B, C]
            x_features.append(x_tsnet)#[[B, C], [B, C]....]

        #時間ごとにtsnetの出力を保存する方法を考慮する必要あり↓
        x_features = torch.stack(x_features, dim=1)  # [[B, C], [B, C]....] → [B, T, C]
        x_mstcn = self.mstcn(x_features.transpose(1, 2))# [B, T, C] → [B, C, T]

        return x_mstcn



        