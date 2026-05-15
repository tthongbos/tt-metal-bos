import torch
import torch.nn as nn

from . import resnet


class TopDown(nn.Sequential):
    def __init__(self, channels=256, layers=8, dtype=torch.float32):
        super().__init__(*[resnet.BasicBlock(channels, channels, dtype=dtype) for _ in range(layers)])
        self.to(dtype)

    def forward(self, x, return_intermediates=False):
        if not return_intermediates:
            for block in self:
                x = block(x)
            return x

        outs = {}
        for block_index, block in enumerate(self, start=1):
            x, block_outs = block(x, return_intermediates=True)
            outs[f"block{block_index}"] = block_outs
        outs["out"] = x
        return x, outs
