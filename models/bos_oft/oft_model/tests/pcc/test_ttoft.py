import atexit
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

_TT_METAL_CACHE_DIR = Path(tempfile.mkdtemp(prefix="ttoft_tt_metal_cache_", dir="/tmp"))
os.environ["TT_METAL_CACHE"] = str(_TT_METAL_CACHE_DIR)


def _cleanup_tt_metal_cache():
    shutil.rmtree(_TT_METAL_CACHE_DIR, ignore_errors=True)


atexit.register(_cleanup_tt_metal_cache)

import ttnn

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_dev.oft_model.reference.architecture.oft import OFT
from model_dev.oft_model.tt.tt_oft import ttnn_OFT


def pcc(a, b, eps=1e-12):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.square().sum().sqrt() * b.square().sum().sqrt()
    if denom <= eps:
        return 1.0 if torch.allclose(a, b, atol=eps, rtol=0.0) else float("nan")
    return ((a * b).sum() / (denom + eps)).item()


def to_torch_float(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float()
    return ttnn.to_torch(x).float()


def tt_to_nchw(tt_x, ref_x):
    x = to_torch_float(tt_x)
    b, c, h, w = ref_x.shape

    if tuple(x.shape) == tuple(ref_x.shape):
        return x

    if x.ndim == 4 and x.shape[0] == b and x.shape[1] >= h and x.shape[2] >= w and x.shape[3] >= c:
        x = x[:, :h, :w, :c]
        return x.permute(0, 3, 1, 2).contiguous()

    if x.ndim == 4 and x.shape[0] == b and x.shape[-2] >= h * w and x.shape[-1] >= c:
        x = x[:, :, : h * w, :c]
        return x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    raise RuntimeError(f"Cannot convert TT shape {tuple(x.shape)} to Torch shape {tuple(ref_x.shape)}")


def align_tt_to_ref(tt_tensor, ref_tensor):
    tt_x = to_torch_float(tt_tensor)
    ref_x = to_torch_float(ref_tensor)

    if tuple(tt_x.shape) == tuple(ref_x.shape):
        return tt_x, ref_x

    if tt_x.ndim == 4 and ref_x.ndim == 4:
        b, c, h, w = ref_x.shape
        if tt_x.shape[0] == b and tt_x.shape[1] >= h and tt_x.shape[2] >= w and tt_x.shape[3] >= c:
            tt_x = tt_x[:, :h, :w, :c].permute(0, 3, 1, 2).contiguous()
            return tt_x, ref_x

    return None, ref_x


def make_test_grid(batch, depth_corners, width_corners, dtype=torch.float32):
    xcoords = torch.linspace(-2.0, 2.0, steps=width_corners, dtype=dtype)
    zcoords = torch.linspace(6.0, 10.0, steps=depth_corners, dtype=dtype)
    xx = xcoords.view(1, -1).expand(depth_corners, width_corners)
    zz = zcoords.view(-1, 1).expand(depth_corners, width_corners)
    yy = torch.zeros_like(xx)
    grid = torch.stack([xx, yy, zz], dim=-1)
    return grid.unsqueeze(0).repeat(batch, 1, 1, 1)


def make_test_calib(batch, image_height, image_width, dtype=torch.float32):
    focal = min(image_height, image_width) * 0.55
    calib = torch.tensor(
        [
            [focal, 0.0, image_width / 2.0, 0.0],
            [0.0, focal, image_height / 2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=dtype,
    )
    return calib.unsqueeze(0).repeat(batch, 1, 1)


def make_tt_oft(torch_model, channels, cell_size, grid_height, scale):
    layer_params = {
        "channels": channels,
        "cell_size": cell_size,
        "grid_height": grid_height,
        "scale": scale,
    }
    model_parameters = {
        "y_corners": torch_model.y_corners.detach().clone(),
        "conv3d": {
            "weight": torch_model.conv3d.weight.detach().clone(),
            "bias": torch_model.conv3d.bias.detach().clone() if torch_model.conv3d.bias is not None else None,
        },
    }
    return ttnn_OFT(layer_params, model_parameters)


def print_pcc(name, tt_tensor, ref_tensor):
    tt_x = tt_to_nchw(tt_tensor, ref_tensor)
    print(f"{name:<24} pcc: {pcc(tt_x, ref_tensor):.8f}  tt: {tuple(tt_x.shape)}  torch: {tuple(ref_tensor.shape)}")
    return tt_x


def print_block_pcc(outs, outs_ref):
    print("\nper-layer outputs")
    for name in [
        "norm_image",
        "feats8",
        "feats16",
        "feats32",
        "lat8",
        "lat16",
        "lat32",
        "ortho",
        "topdown",
        "scores",
        "pos_offsets",
        "dim_offsets",
        "ang_offsets",
    ]:
        print_pcc(name, outs[name], outs_ref[name])
    exit()
    print("\nper-basic-block outputs")
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        tt_blocks = outs.get(f"{layer_name}_blocks", {})
        ref_blocks = outs_ref.get(f"{layer_name}_blocks", {})

        for block_name, ref_block in ref_blocks.items():
            tt_block = tt_blocks.get(block_name)
            if tt_block is None:
                print(f"{layer_name}.{block_name:<18} missing in TTNN outputs")
                continue

            for op_name in block_ops:
                if op_name not in ref_block or op_name not in tt_block:
                    continue
                print_pcc(f"{layer_name}.{block_name}.{op_name}", tt_block[op_name], ref_block[op_name])


def print_intermediate_pcc(outs, outs_ref):
    names = [
        "features",
        "corners",
        "norm_corners",
        "bbox_corners_pre_flatten",
        "bbox_corners",
        "area",
        "integral_img",
        "top_left",
        "btm_right",
        "top_right",
        "btm_left",
        "vox_feats_sum",
        "vox_feats_area",
        "visible_float",
        "vox_feats_visible",
        "vox_feats_permuted",
        "vox_feats_flatten0",
        "vox_feats_flat",
        "conv3d",
        "conv3d_reshaped",
        "out",
    ]

    print("\nper-layer outputs")
    for name in names:
        if name not in outs or name not in outs_ref:
            print(f"{name:<28} missing")
            continue

        tt_x, ref_x = align_tt_to_ref(outs[name], outs_ref[name])
        if tt_x is None:
            print(
                f"{name:<28} shape mismatch  tt: {tuple(to_torch_float(outs[name]).shape)}  torch: {tuple(ref_x.shape)}"
            )
            continue

        diff = (tt_x - ref_x).abs()
        print(
            f"{name:<28} pcc: {pcc(tt_x, ref_x):.8f}  "
            f"max_abs: {diff.max().item():.6f}  mean_abs: {diff.mean().item():.6f}  "
            f"tt: {tuple(tt_x.shape)}  torch: {tuple(ref_x.shape)}"
        )


def main():
    torch.manual_seed(0)
    batch = 1
    channels = 32
    image_height = 32
    image_width = 64
    depth_corners = 5
    width_corners = 7
    cell_size = 1.0
    grid_height = 2.0
    scale = 1.0
    dtype = torch.float32

    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    print("device arch:", device.arch())

    try:
        torch_model = OFT(channels, cell_size=cell_size, grid_height=grid_height, scale=scale, dtype=dtype).eval()
        tt_model = make_tt_oft(torch_model, channels, cell_size, grid_height, scale)

        torch.manual_seed(123)
        features_nchw = torch.randn(batch, channels, image_height, image_width, dtype=dtype)
        features_nhwc = features_nchw.permute(0, 2, 3, 1).contiguous()
        calib = make_test_calib(batch, image_height, image_width, dtype=dtype)
        grid = make_test_grid(batch, depth_corners, width_corners, dtype=dtype)

        with torch.no_grad():
            ref_out, outs_ref = run_torch_oft_with_outs(torch_model, features_nchw, calib, grid)

        ttnn.synchronize_device(device)
        start = time.perf_counter()
        tt_out, outs = tt_model(features_nhwc, calib, grid, device, return_intermediates=True)
        ttnn.synchronize_device(device)
        elapsed_s = time.perf_counter() - start

        print("\noft output")
        tt_out = print_pcc("out", tt_out, ref_out)
        print_intermediate_pcc(outs, outs_ref)

        if torch.isnan(tt_out).any():
            raise RuntimeError("TT OFT output has NaN values")
        if torch.isnan(ref_out).any():
            raise RuntimeError("Torch OFT output has NaN values")

        print(f"\nDebug latency: {elapsed_s:.4f} seconds")
        print(f"Debug FPS: {batch / elapsed_s:.4f}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
