import os, math, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
from tqdm import tqdm

# ----------- Config -----------
IMG_SIZE = 640
GRID_SIZE = 20
NUM_CLASSES = 2
DATA_DIR = "/kaggle/input/bird-vs-drone/Dataset"
OUT_DIR = "/kaggle/working"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 10

# ----------- Simple YOLO Nano Model -----------
class DWConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))

class YoloNano(nn.Module):  # Updated for output 20x20
    def __init__(self, num_classes=NUM_CLASSES, input_ch=1, base_filters=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_ch, base_filters, 3, stride=2, padding=1, bias=False),  # 640->320
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.stage1 = DWConv(base_filters, base_filters*2, stride=2)   # 320->160
        self.stage2 = DWConv(base_filters*2, base_filters*4, stride=2) # 160->80
        self.stage3 = DWConv(base_filters*4, base_filters*8, stride=2) # 80->40
        self.stage4 = DWConv(base_filters*8, base_filters*16, stride=2) # 40->20
        # New final stage for exact 20x20 output:
        self.stage5 = DWConv(base_filters*16, base_filters*16, stride=1) # keep 20x20, no downsample
        out_ch = 1 + 4 + num_classes
        self.head = nn.Conv2d(base_filters*16, out_ch, 1)
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return self.head(x)