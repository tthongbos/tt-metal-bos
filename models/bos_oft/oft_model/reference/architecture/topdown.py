import torch
import torch.nn as nn

from . import resnet


class TopDown(nn.Sequential):
    def __init__(self, channels=256, layers=8, dtype=torch.float32):
        super().__init__(*[resnet.BasicBlock(channels, channels, dtype=dtype) for _ in range(layers)])
        self.to(dtype)
