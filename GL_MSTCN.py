import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.MSTCN import MSTCN


class TS_Net(nn.Module):
    def __init__(self, 
                 video_frame):
        super(TS_Net, self).__init__()
        