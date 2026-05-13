import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import resnet
    from .oft import OFT
    from .topdown import TopDown
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[4]))
    from model_dev.oft_model.reference.architecture import resnet
    from model_dev.oft_model.reference.architecture.oft import OFT
    from model_dev.oft_model.reference.architecture.topdown import TopDown


class OftNet(nn.Module):
    def __init__(
        self,
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=0.5,
        grid_height=6.0,
        frontend_pretrained=False,
        dtype=torch.float32,
    ):
        super().__init__()

        assert frontend in ["resnet18", "resnet34"], "unrecognised frontend"
        self.frontend = getattr(resnet, frontend)(pretrained=frontend_pretrained, dtype=dtype)

        self.lat8 = nn.Conv2d(128, 256, 1, dtype=dtype)
        self.lat16 = nn.Conv2d(256, 256, 1, dtype=dtype)
        self.lat32 = nn.Conv2d(512, 256, 1, dtype=dtype)
        self.bn8 = nn.GroupNorm(16, 256, dtype=dtype)
        self.bn16 = nn.GroupNorm(16, 256, dtype=dtype)
        self.bn32 = nn.GroupNorm(16, 256, dtype=dtype)

        self.oft8 = OFT(256, grid_res, grid_height, 1 / 8.0, dtype=dtype)
        self.oft16 = OFT(256, grid_res, grid_height, 1 / 16.0, dtype=dtype)
        self.oft32 = OFT(256, grid_res, grid_height, 1 / 32.0, dtype=dtype)

        self.topdown = TopDown(channels=256, layers=topdown_layers, dtype=dtype)
        self.head = nn.Conv2d(256, num_classes * 9, kernel_size=3, padding=1, dtype=dtype)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=dtype))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=dtype))
        self.dtype = dtype
        self.to(dtype)

    def forward(self, image, calib, grid):
        if image.dtype != self.dtype:
            image = image.to(self.dtype)
        if calib.dtype != self.dtype:
            calib = calib.to(self.dtype)
        if grid.dtype != self.dtype:
            grid = grid.to(self.dtype)

        image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        feats8, feats16, feats32, _ = self.frontend.forward_feature_pyramid(image)

        lat8 = F.relu(self.bn8(self.lat8(feats8)))
        lat16 = F.relu(self.bn16(self.lat16(feats16)))
        lat32 = F.relu(self.bn32(self.lat32(feats32)))

        ortho = self.oft8(lat8, calib, grid) + self.oft16(lat16, calib, grid) + self.oft32(lat32, calib, grid)
        topdown = self.topdown(ortho)

        batch, _, depth, width = topdown.size()
        outputs = self.head(topdown).view(batch, -1, 9, depth, width)
        scores, pos_offsets, dim_offsets, ang_offsets = torch.split(outputs, [1, 3, 3, 2], dim=2)
        return scores.squeeze(2), pos_offsets, dim_offsets, ang_offsets
