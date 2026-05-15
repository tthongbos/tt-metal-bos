import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import torch

import ttnn

THIS_FILE = Path(__file__).resolve()
OFT_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
TT_METAL_ROOT = Path(os.environ.get("TT_METAL_HOME", WORKSPACE_ROOT / "tt-metal")).resolve()

for path in (PROJECT_ROOT, TT_METAL_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_dev.oft_model.reference.architecture.bbox_visualize import save_bbox_visualization
from model_dev.oft_model.reference.architecture.encoder import ObjectEncoder
from model_dev.oft_model.reference.architecture.oftnet import OftNet
from model_dev.oft_model.tt.oftnet import TTOFTNET
from model_dev.oft_model.utils.pipeline_utils import (
    load_calib,
    load_oft_checkpoint,
    load_padded_image_tensor,
    make_grid,
    print_objects,
)

from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters

DEFAULT_IMAGE_PATH = OFT_ROOT / "utils" / "000013.jpg"
DEFAULT_CALIB_PATH = OFT_ROOT / "utils" / "000013.txt"
DEFAULT_CHECKPOINT_PATH = OFT_ROOT / "utils" / "checkpoint" / "oft_checkpoint-0600.pth"

GRID_RES = 0.5
GRID_SIZE = (80.0, 80.0)
GRID_HEIGHT = 4.0
Y_OFFSET = 1.74
H_PADDED = 384
W_PADDED = 1280
NMS_THRESH = 0.0


import shutil


def pcc(a, b, eps=1e-12):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    return ((a * b).sum() / (a.square().sum().sqrt() * b.square().sum().sqrt() + eps)).item()


def disable_graphviz_render_if_needed():
    if shutil.which("dot") is not None:
        return

    import ttnn.model_preprocessing as ttnn_model_preprocessing

    ttnn_model_preprocessing.visualize = lambda *args, **kwargs: None


def decode_oftnet_outputs(scores, pos_offsets, dim_offsets, ang_offsets, grid, dtype, nms_thresh):
    encoder = ObjectEncoder(dtype=dtype, nms_thresh=nms_thresh)
    encoder.eval()

    decoded, _ = encoder.decode(
        scores.squeeze(0),
        pos_offsets.squeeze(0),
        dim_offsets.squeeze(0),
        ang_offsets.squeeze(0),
        grid.squeeze(0),
    )
    return encoder.create_objects(*decoded), decoded


def tt_to_nchw(tt_x, ref_x):
    x = tt_x.float() if isinstance(tt_x, torch.Tensor) else ttnn.to_torch(tt_x).float()
    b, c, h, w = ref_x.shape

    if tuple(x.shape) == tuple(ref_x.shape):
        return x

    # TTNN thường là [B, 1, H*W, C]
    if x.ndim == 4 and x.shape[0] == b and x.shape[-2] >= h * w and x.shape[-1] >= c:
        x = x[:, :, : h * w, :c]
        return x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    # Or [B, H, W, C].
    if x.ndim == 4 and x.shape[0] == b and x.shape[1] >= h and x.shape[2] >= w and x.shape[3] >= c:
        x = x[:, :h, :w, :c]
        return x.permute(0, 3, 1, 2).contiguous()

    raise RuntimeError(f"Cannot convert TT shape {tuple(x.shape)} to Torch shape {tuple(ref_x.shape)}")


def print_pcc(name, tt_tensor, ref_tensor):
    tt_x = tt_to_nchw(tt_tensor, ref_tensor)
    print(f"{name:<24} pcc: {pcc(tt_x, ref_tensor):.8f}  tt: {tuple(tt_x.shape)}  torch: {tuple(ref_tensor.shape)}")
    return tt_x


def print_generic_pcc(name, tt_tensor, ref_tensor):
    tt_x = tt_tensor.float() if isinstance(tt_tensor, torch.Tensor) else ttnn.to_torch(tt_tensor).float()
    ref_x = ref_tensor.float()

    if tuple(tt_x.shape) != tuple(ref_x.shape):
        if (
            tt_x.ndim == 4
            and ref_x.ndim == 4
            and tt_x.shape[0] == ref_x.shape[0]
            and tt_x.shape[-1] == ref_x.shape[1]
            and tt_x.shape[1] == ref_x.shape[2]
            and tt_x.shape[2] == ref_x.shape[3]
        ):
            tt_x = tt_x.permute(0, 3, 1, 2).contiguous()
        elif (
            tt_x.ndim == 5
            and ref_x.ndim == 5
            and tt_x.shape[0] == ref_x.shape[0]
            and tt_x.shape[-1] == ref_x.shape[1]
            and tt_x.shape[1] == ref_x.shape[2]
            and tt_x.shape[2] == ref_x.shape[3]
            and tt_x.shape[3] == ref_x.shape[4]
        ):
            tt_x = tt_x.permute(0, 4, 1, 2, 3).contiguous()

    if tuple(tt_x.shape) != tuple(ref_x.shape):
        print(
            f"{name:<24} shape-mismatch  tt: {tuple(tt_x.shape)} dtype: {tt_x.dtype}  torch: {tuple(ref_x.shape)} dtype: {ref_x.dtype}"
        )
        return None

    print(
        f"{name:<24} pcc: {pcc(tt_x, ref_x):.8f}  tt: {tuple(tt_x.shape)} dtype: {tt_x.dtype}  torch: {tuple(ref_x.shape)} dtype: {ref_x.dtype}"
    )
    return tt_x


def print_oft_pcc(module_name, tt_outs, ref_outs):
    common_order = [
        "features",
        "corners",
        "norm_corners",
        "bbox_corners_pre_flatten",
        "bbox_corners",
        "area",
        "area_for_div",
        "integral_img",
        "top_left",
        "btm_right",
        "top_right",
        "btm_left",
        "rect_sum",
        "vox_avg",
        "vox_feats_flat",
        "conv3d",
        "out",
    ]

    printed_any = False
    for key in common_order:
        if key not in tt_outs or key not in ref_outs:
            continue
        print_generic_pcc(f"{module_name}.{key}", tt_outs[key], ref_outs[key])
        printed_any = True

    if not printed_any:
        print(f"{module_name:<24} no-common-intermediates")


def print_block_pcc(outs, outs_ref):
    block_ops = [
        "conv1",
        "grnorm1",
        "relu1",
        "conv2",
        "grnorm2",
        "downsample",
        "add",
        "relu2",
        "out",
    ]

    print("\nper-layer outputs")

    for name in ["norm_image", "feats8", "feats16", "feats32", "lat8", "lat16", "lat32", "ortho", "topdown"]:
        print_pcc(name, outs[name], outs_ref[name])
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

    print("\nper-oft outputs")
    for oft_name in ["oft8", "oft16", "oft32"]:
        tt_oft = outs.get(oft_name, {})
        ref_oft = outs_ref.get(oft_name, {})
        print_oft_pcc(oft_name, tt_oft, ref_oft)

    print("\nper-oft ortho outputs")
    for ortho_name in ["ortho8", "ortho16", "ortho32"]:
        if ortho_name in outs and ortho_name in outs_ref:
            print_pcc(ortho_name, outs[ortho_name], outs_ref[ortho_name])

    print("\nper-topdown outputs")
    tt_blocks = outs.get("topdown_blocks", {})
    ref_blocks = outs_ref.get("topdown_blocks", {})
    for block_name, ref_block in ref_blocks.items():
        tt_block = tt_blocks.get(block_name)
        if tt_block is None:
            print(f"topdown.{block_name:<16} missing in TTNN outputs")
            continue
        for op_name in block_ops:
            if op_name not in ref_block or op_name not in tt_block:
                continue
            print_pcc(f"topdown.{block_name}.{op_name}", tt_block[op_name], ref_block[op_name])


def tt_tensor_to_torch(tt_tensor, shape):
    return ttnn.to_torch(tt_tensor).reshape(shape).float()


def open_device(device_id, l1_small_size):
    num_devices = ttnn.GetNumPCIeDevices()
    if num_devices <= device_id:
        raise RuntimeError(f"TT device {device_id} is not available. Found {num_devices} PCIe device(s).")

    device = ttnn.open_device(device_id=device_id, l1_small_size=l1_small_size)
    if hasattr(device, "disable_and_clear_program_cache"):
        device.disable_and_clear_program_cache()
    return device


def build_tt_oftnet(
    device,
    ref_model,
    image,
    calib,
    grid,
    topdown_layers,
    grid_res,
    grid_height,
):
    disable_graphviz_render_if_needed()
    model_parameters = create_OFT_model_parameters(ref_model, (image, calib, grid), device=device)

    tt_model = TTOFTNET(
        device=device,
        parameters=model_parameters,
        layer_args=model_parameters.layer_args,
        mean=ref_model.mean,
        std=ref_model.std,
        input_shape_hw=image.shape[2:],  # NCHW -> H, W
        torch_frontend=ref_model.frontend,
        batch_size=image.shape[0],
        grid_res=grid_res,
        grid_height=grid_height,
        dtype=ttnn.bfloat16,
    )
    # install_clean_forward_overrides(tt_model)
    return tt_model


def run_ttnn_pipeline(
    image_path=DEFAULT_IMAGE_PATH,
    calib_path=DEFAULT_CALIB_PATH,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    device_id=0,
    l1_small_size=32768,
    topdown_layers=8,
    grid_res=GRID_RES,
    grid_size=GRID_SIZE,
    grid_height=GRID_HEIGHT,
    y_offset=Y_OFFSET,
    pad_hw=(H_PADDED, W_PADDED),
    nms_thresh=NMS_THRESH,
    output_path=None,
    host_postprocess=False,
    collect_intermediates=False,
):
    torch.manual_seed(0)
    model_dtype = torch.float32

    image, image_hw = load_padded_image_tensor(image_path, pad_hw=pad_hw, dtype=model_dtype)
    calib = load_calib(calib_path, dtype=model_dtype)
    grid = make_grid(
        grid_size=grid_size,
        grid_offset=(-grid_size[0] / 2.0, y_offset, 0.0),
        grid_res=grid_res,
        dtype=model_dtype,
    )

    ref_model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=topdown_layers,
        grid_res=grid_res,
        grid_height=grid_height,
        dtype=model_dtype,
    )

    ref_model = load_oft_checkpoint(ref_model, checkpoint_path)
    ref_model.eval()
    with torch.no_grad():
        scores, pos_offsets, dim_offsets, ang_offsets, ref_out = ref_model(image, calib, grid)
        objects, decoded = decode_oftnet_outputs(
            scores,
            pos_offsets,
            dim_offsets,
            ang_offsets,
            grid,
            dtype=model_dtype,
            nms_thresh=nms_thresh,
        )
    device = open_device(device_id=device_id, l1_small_size=l1_small_size)
    try:
        tt_model = build_tt_oftnet(
            device,
            ref_model,
            image,
            calib,
            grid,
            topdown_layers=topdown_layers,
            grid_res=grid_res,
            grid_height=grid_height,
        )

        tt_image = image.permute(0, 2, 3, 1).contiguous()
        tt_image = ttnn.from_torch(tt_image, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        tt_calib = ttnn.from_torch(calib, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_grid = ttnn.from_torch(grid, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn.synchronize_device(device)
        start = time.perf_counter()
        tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets, out = tt_model.forward(
            device,
            tt_image,
            tt_calib,
            tt_grid,
            collect_intermediates=collect_intermediates,
        )
        ttnn.synchronize_device(device)
        if collect_intermediates:
            print_block_pcc(out, ref_out)
        elapsed_s = time.perf_counter() - start
        batch = image.shape[0]
        depth = grid.shape[1] - 1
        width = grid.shape[2] - 1
        scores = ttnn.to_torch(tt_scores).float()
        pos_offsets = ttnn.to_torch(tt_pos_offsets).float()
        dim_offsets = ttnn.to_torch(tt_dim_offsets).float()
        ang_offsets = ttnn.to_torch(tt_ang_offsets).float()

        scores = scores.reshape(batch, 1, depth, width)

        pos_offsets = pos_offsets.permute(0, 1, 3, 2).reshape(batch, 1, 3, depth, width)
        dim_offsets = dim_offsets.permute(0, 1, 3, 2).reshape(batch, 1, 3, depth, width)
        ang_offsets = ang_offsets.permute(0, 1, 3, 2).reshape(batch, 1, 2, depth, width)

        print("TTNN OftNet pipeline finished")
        print(f"image source:       {image_path}")
        print(f"calib source:       {calib_path}")
        print(f"checkpoint:         {checkpoint_path}")
        print(f"image hw:           {image_hw}")
        print(f"grid cells:         {depth}x{width}")
        print(f"latency:            {elapsed_s:.4f} seconds")
        print(f"fps:                {batch / elapsed_s:.4f}")
        print(f"scores shape:       {tuple(scores.shape)}")
        print(f"pos_offsets shape:  {tuple(pos_offsets.shape)}")
        print(f"dim_offsets shape:  {tuple(dim_offsets.shape)}")
        print(f"ang_offsets shape:  {tuple(ang_offsets.shape)}")

        if not host_postprocess:
            return None, None, (tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets)

        # scores = tt_tensor_to_torch(tt_scores, (batch, 1, depth, width))
        # pos_offsets = tt_tensor_to_torch(pos_offsets, (batch, 1, 3, depth, width))
        # dim_offsets = tt_tensor_to_torch(dim_offsets, (batch, 1, 3, depth, width))
        # ang_offsets = tt_tensor_to_torch(ang_offsets, (batch, 1, 2, depth, width))

        for name, tensor in (
            ("scores", scores),
            ("pos_offsets", pos_offsets),
            ("dim_offsets", dim_offsets),
            ("ang_offsets", ang_offsets),
        ):
            if torch.isnan(tensor).any():
                raise RuntimeError(f"{name} has NaN values")

        objects, decoded = decode_oftnet_outputs(
            scores,
            pos_offsets,
            dim_offsets,
            ang_offsets,
            grid,
            dtype=model_dtype,
            nms_thresh=nms_thresh,
        )
        print_objects(objects)

        if output_path:
            save_bbox_visualization(image, image_hw, calib, objects, output_path)
            print(f"saved visualization: {output_path}")

        return objects, decoded, (scores, pos_offsets, dim_offsets, ang_offsets)
    finally:
        ttnn.close_device(device)


def main():
    parser = argparse.ArgumentParser(description="Run TTNN OFTNet: image + calib + weights -> decoded objects.")
    parser.add_argument(
        "--image", default=str(DEFAULT_IMAGE_PATH), help="Path, file:// URL, or http(s) URL to a KITTI RGB image."
    )
    parser.add_argument("--calib", default=str(DEFAULT_CALIB_PATH), help="Path to KITTI calib txt file with P2 matrix.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH), help="Path to trained OFT checkpoint.")
    parser.add_argument("--device-id", type=int, default=0, help="TT PCIe device id.")
    parser.add_argument("--l1-small-size", type=int, default=16 * 1024, help="TTNN open_device l1_small_size.")
    parser.add_argument("--topdown-layers", type=int, default=8, help="Number of TopDown BasicBlocks.")
    parser.add_argument("--grid-res", type=float, default=GRID_RES, help="Grid resolution.")
    parser.add_argument("--grid-size", type=float, nargs=2, default=list(GRID_SIZE), metavar=("DEPTH", "WIDTH"))
    parser.add_argument("--grid-height", type=float, default=GRID_HEIGHT, help="OFT vertical grid height.")
    parser.add_argument("--y-offset", type=float, default=Y_OFFSET, help="Grid y offset.")
    parser.add_argument("--pad-height", type=int, default=H_PADDED, help="Padded input image height.")
    parser.add_argument("--pad-width", type=int, default=W_PADDED, help="Padded input image width.")
    parser.add_argument("--nms-thresh", type=float, default=NMS_THRESH, help="Decoder NMS threshold.")
    parser.add_argument(
        "--host-postprocess",
        action="store_true",
        help="Convert TTNN outputs to Torch only after model inference for object decode.",
    )
    parser.add_argument(
        "--collect-intermediates", action="store_true", help="Capture intermediate tensors and print PCC debug output."
    )
    parser.add_argument(
        "--save-vis",
        default=None,
        help="Optional path to save projected 3D bounding boxes; implies --host-postprocess.",
    )
    args = parser.parse_args()

    run_ttnn_pipeline(
        image_path=args.image,
        calib_path=args.calib,
        checkpoint_path=args.checkpoint,
        device_id=args.device_id,
        l1_small_size=args.l1_small_size,
        topdown_layers=args.topdown_layers,
        grid_res=args.grid_res,
        grid_size=tuple(args.grid_size),
        grid_height=args.grid_height,
        y_offset=args.y_offset,
        pad_hw=(args.pad_height, args.pad_width),
        nms_thresh=args.nms_thresh,
        output_path=args.save_vis,
        host_postprocess=args.host_postprocess or args.save_vis is not None,
        collect_intermediates=args.collect_intermediates,
    )


if __name__ == "__main__":
    main()
