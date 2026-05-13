import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model_dev.oft_model.utils.helper import perspective
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4]))
    from model_dev.oft_model.utils.helper import perspective


EPSILON = 1e-6
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_IMAGE_PATH = Path(__file__).resolve().parents[2] / "utils" / "huggingface_cat_image.jpg"


# Fast average pooling with integral images
def integral_image(features):
    return torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)


class OFT(nn.Module):
    def __init__(self, channel, cell_size, grid_height, scale, dtype=torch.float32):
        super().__init__()
        y_corners = torch.arange(0, grid_height, cell_size, dtype=dtype) - grid_height / 2.0
        y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])  # [0, y, 0]
        self.register_buffer("y_corners", y_corners)
        self.conv3d = nn.Linear((len(y_corners) - 1) * channel, channel, dtype=dtype)
        self.scale = scale
        self.dtype = dtype
        self.to(dtype)

    def forward(self, features, calib, grid):
        if features.dtype != self.dtype:
            features = features.to(self.dtype)
        if calib.dtype != self.dtype:
            calib = calib.to(self.dtype)
        if grid.dtype != self.dtype:
            grid = grid.to(self.dtype)

        # expand grid in y dimension: (B, 1, D, W, 3) + (Y, 1, 1, 3) -> (B, Y, D, W, 3)
        corners = grid.unsqueeze(1) + self.y_corners.view(-1, 1, 1, 3)
        img_corners = perspective(calib.view(-1, 1, 1, 1, 3, 4), corners)

        img_height, img_width = features.size()[2:]
        img_size = torch.tensor([img_width, img_height], dtype=self.dtype, device=features.device) / self.scale
        norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)
        bbox_corners = torch.cat(
            [
                torch.min(norm_corners[:, :-1, :-1, :-1], norm_corners[:, :-1, 1:, :-1]),
                torch.max(norm_corners[:, 1:, 1:, 1:], norm_corners[:, 1:, :-1, 1:]),
            ],
            dim=-1,
        )
        batch, _, depth, width, _ = bbox_corners.size()
        bbox_corners = bbox_corners.flatten(2, 3)

        # Compute the area of each bounding box
        epsilon = torch.tensor(EPSILON, dtype=self.dtype, device=features.device)
        area = (
            (bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) * img_height * img_width * 0.25 + epsilon
        ).unsqueeze(1)
        visible = area > epsilon

        # Sample integral image at bounding box locations
        integral_img = integral_image(features)
        top_left = F.grid_sample(integral_img, bbox_corners[..., [0, 1]], align_corners=False)
        btm_right = F.grid_sample(integral_img, bbox_corners[..., [2, 3]], align_corners=False)
        top_right = F.grid_sample(integral_img, bbox_corners[..., [2, 1]], align_corners=False)
        btm_left = F.grid_sample(integral_img, bbox_corners[..., [0, 3]], align_corners=False)

        # Compute voxel features (ignore features which are not visible)
        vox_feats = (top_left + btm_right - top_right - btm_left) / area
        vox_feats = vox_feats * visible.to(dtype=self.dtype)
        # vox_feats = vox_feats.view(batch, -1, depth, width)
        vox_feats = vox_feats.permute(0, 3, 1, 2).flatten(0, 1).flatten(1, 2)

        # Flatten to orthographic feature map
        ortho_feats = self.conv3d(vox_feats).view(batch, depth, width, -1)
        ortho_feats = F.relu(ortho_feats.permute(0, 3, 1, 2), inplace=True)
        # ortho_feats = F.relu(self.conv3d(vox_feats))

        # Block gradients to pixels which are not visible in the image

        return ortho_feats


def _make_test_grid(batch, depth_corners, width_corners, dtype=torch.float32):
    xcoords = torch.linspace(-2.0, 2.0, steps=width_corners, dtype=dtype)
    zcoords = torch.linspace(5.0, 9.0, steps=depth_corners, dtype=dtype)
    xx = xcoords.view(1, -1).expand(depth_corners, width_corners)
    zz = zcoords.view(-1, 1).expand(depth_corners, width_corners)
    yy = torch.zeros_like(xx)
    grid = torch.stack([xx, yy, zz], dim=-1)
    return grid.unsqueeze(0).repeat(batch, 1, 1, 1)


def _make_test_calib(image_height, image_width, dtype=torch.float32):
    focal = min(image_height, image_width) * 0.55
    return torch.tensor(
        [
            [focal, 0.0, image_width / 2.0, 0.0],
            [0.0, focal, image_height / 2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=dtype,
    ).unsqueeze(0)


def _load_resnet_helpers():
    try:
        from model_dev.oft_model.reference.architecture.resnet import _preprocess_image, resnet18
    except ModuleNotFoundError:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.append(str(PROJECT_ROOT))
        from model_dev.oft_model.reference.architecture.resnet import _preprocess_image, resnet18

    return resnet18, _preprocess_image


def _test_integral_image(dtype=torch.float32):
    features = torch.ones(1, 1, 2, 3, dtype=dtype)
    expected = torch.tensor([[[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]]], dtype=dtype)
    output = integral_image(features)
    assert torch.allclose(output, expected), f"integral_image failed: {output}"


def _test_oft_forward():
    torch.manual_seed(0)
    dtype = torch.float32
    batch = 1
    channels = 4
    image_height = 32
    image_width = 32
    depth_corners = 5
    width_corners = 6

    model = OFT(channels, cell_size=1.0, grid_height=2.0, scale=1.0, dtype=dtype)
    model.eval()

    features = torch.randn(batch, channels, image_height, image_width, dtype=dtype)
    calib = torch.tensor(
        [
            [20.0, 0.0, image_width / 2.0, 0.0],
            [0.0, 20.0, image_height / 2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=dtype,
    ).unsqueeze(0)
    grid = _make_test_grid(batch, depth_corners, width_corners, dtype=dtype)

    with torch.no_grad():
        output = model(features, calib, grid)

    expected_shape = (batch, channels, depth_corners - 1, width_corners - 1)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {tuple(output.shape)}"
    assert output.dtype == dtype, f"Expected dtype {dtype}, got {output.dtype}"
    assert not torch.isnan(output).any(), "OFT output has NaN values"

    print("OFT forward test passed")
    print(f"features shape: {tuple(features.shape)}")
    print(f"calib shape:    {tuple(calib.shape)}")
    print(f"grid shape:     {tuple(grid.shape)}")
    print(f"output shape:   {tuple(output.shape)}")


def _test_resnet_to_oft_image(image_path=DEFAULT_IMAGE_PATH, pretrained=False):
    torch.manual_seed(0)
    dtype = torch.float32
    resnet18, preprocess_image = _load_resnet_helpers()

    frontend = resnet18(pretrained=pretrained, dtype=dtype)
    frontend.eval()

    image = preprocess_image(image_path, dtype=dtype)
    with torch.no_grad():
        features = frontend.forward_features(image)

    batch, channels, feat_height, feat_width = features.shape
    image_height, image_width = image.shape[2:]
    scale = feat_width / image_width

    grid = _make_test_grid(batch, depth_corners=6, width_corners=8, dtype=dtype)
    calib = _make_test_calib(image_height, image_width, dtype=dtype)
    oft = OFT(channels, cell_size=1.0, grid_height=2.0, scale=scale, dtype=dtype)
    oft.eval()

    with torch.no_grad():
        ortho_features = oft(features, calib, grid)

    expected_shape = (batch, channels, grid.shape[1] - 1, grid.shape[2] - 1)
    assert (
        ortho_features.shape == expected_shape
    ), f"Expected ResNet->OFT output shape {expected_shape}, got {tuple(ortho_features.shape)}"
    assert not torch.isnan(features).any(), "ResNet features have NaN values"
    assert not torch.isnan(ortho_features).any(), "ResNet->OFT output has NaN values"

    print("ResNet -> OFT image test passed")
    print(f"image source:        {image_path}")
    print(f"image tensor shape:  {tuple(image.shape)}")
    print(f"resnet feature:      {tuple(features.shape)}")
    print(f"feature scale:       {scale:.6f}")
    print(f"calib shape:         {tuple(calib.shape)}")
    print(f"grid shape:          {tuple(grid.shape)}")
    print(f"oft output shape:    {tuple(ortho_features.shape)}")


def _oft_test(image_path=DEFAULT_IMAGE_PATH, pretrained=False):
    _test_integral_image()
    _test_oft_forward()
    _test_resnet_to_oft_image(image_path=image_path, pretrained=pretrained)


def _main():
    parser = argparse.ArgumentParser(description="Run OFT smoke tests.")
    parser.add_argument(
        "--image", default=str(DEFAULT_IMAGE_PATH), help="Path, file:// URL, or http(s) URL to an RGB image."
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained ImageNet weights for the ResNet frontend."
    )
    args = parser.parse_args()

    _oft_test(image_path=args.image, pretrained=args.pretrained)


if __name__ == "__main__":
    _main()
