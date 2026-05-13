import argparse
import sys
from pathlib import Path

import torch

try:
    from ...utils.pipeline_utils import (
        load_calib,
        load_oft_checkpoint,
        load_padded_image_tensor,
        make_grid,
        print_objects,
    )
    from .bbox_visualize import save_bbox_visualization
    from .encoder import ObjectEncoder
    from .oftnet import OftNet
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[4]))
    from model_dev.oft_model.reference.architecture.bbox_visualize import save_bbox_visualization
    from model_dev.oft_model.reference.architecture.encoder import ObjectEncoder
    from model_dev.oft_model.reference.architecture.oftnet import OftNet
    from model_dev.oft_model.utils.pipeline_utils import (
        load_calib,
        load_oft_checkpoint,
        load_padded_image_tensor,
        make_grid,
        print_objects,
    )


DEFAULT_IMAGE_PATH = Path(__file__).resolve().parents[2] / "utils" / "000013.jpg"
DEFAULT_CALIB_PATH = Path(__file__).resolve().parents[2] / "utils" / "000013.txt"
GRID_RES = 0.5
GRID_SIZE = (80.0, 80.0)
GRID_HEIGHT = 4.0
Y_OFFSET = 1.74
H_PADDED = 384
W_PADDED = 1280
NMS_THRESH = 0.4
DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "utils" / "checkpoint" / "oft_checkpoint-0600.pth"


def decode_oftnet_outputs(
    scores,
    pos_offsets,
    dim_offsets,
    ang_offsets,
    grid,
    dtype=torch.float32,
    nms_thresh=NMS_THRESH,
):
    encoder = ObjectEncoder(dtype=dtype, nms_thresh=nms_thresh)
    encoder.eval()

    decoded, _ = encoder.decode(
        scores.squeeze(0),
        pos_offsets.squeeze(0),
        dim_offsets.squeeze(0),
        ang_offsets.squeeze(0),
        grid.squeeze(0),
    )
    objects = encoder.create_objects(*decoded)
    return objects, decoded


def run_full_pipeline(
    image_path=DEFAULT_IMAGE_PATH,
    calib_path=DEFAULT_CALIB_PATH,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    frontend_pretrained=False,
    topdown_layers=8,
    grid_res=GRID_RES,
    grid_size=GRID_SIZE,
    grid_height=GRID_HEIGHT,
    y_offset=Y_OFFSET,
    pad_hw=(H_PADDED, W_PADDED),
    nms_thresh=NMS_THRESH,
    output_path=None,
):
    torch.manual_seed(0)
    dtype = torch.float32

    image, image_hw = load_padded_image_tensor(image_path, pad_hw=pad_hw, dtype=dtype)
    calib = load_calib(calib_path, dtype=dtype)
    grid = make_grid(
        grid_size=grid_size,
        grid_offset=(-grid_size[0] / 2.0, y_offset, 0.0),
        grid_res=grid_res,
        dtype=dtype,
    )

    model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=topdown_layers,
        grid_res=grid_res,
        grid_height=grid_height,
        frontend_pretrained=frontend_pretrained,
        dtype=dtype,
    )
    model = load_oft_checkpoint(model, checkpoint_path)
    model.eval()

    with torch.no_grad():
        scores, pos_offsets, dim_offsets, ang_offsets = model(image, calib, grid)
        objects, decoded = decode_oftnet_outputs(
            scores,
            pos_offsets,
            dim_offsets,
            ang_offsets,
            grid,
            dtype=dtype,
            nms_thresh=nms_thresh,
        )

    assert not torch.isnan(scores).any(), "scores has NaN values"
    assert not torch.isnan(pos_offsets).any(), "pos_offsets has NaN values"
    assert not torch.isnan(dim_offsets).any(), "dim_offsets has NaN values"
    assert not torch.isnan(ang_offsets).any(), "ang_offsets has NaN values"

    print("OftNet full pipeline finished")
    print(f"image source:       {image_path}")
    print(f"calib source:       {calib_path}")
    print(f"checkpoint:         {checkpoint_path or 'none'}")
    print(f"image hw:           {image_hw}")
    print_objects(objects)

    if output_path:
        save_bbox_visualization(image, image_hw, calib, objects, output_path)

    return objects, decoded, (scores, pos_offsets, dim_offsets, ang_offsets)


def main():
    parser = argparse.ArgumentParser(
        description="Run full OFT pipeline: image + calib -> OftNet -> decoder -> objects."
    )
    parser.add_argument(
        "--image", default=str(DEFAULT_IMAGE_PATH), help="Path, file:// URL, or http(s) URL to a KITTI RGB image."
    )
    parser.add_argument("--calib", default=str(DEFAULT_CALIB_PATH), help="Path to KITTI calib txt file with P2 matrix.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH), help="Path to trained OFT checkpoint.")
    parser.add_argument(
        "--frontend-pretrained",
        action="store_true",
        help="Use torchvision/ImageNet pretrained ResNet frontend weights.",
    )
    parser.add_argument("--topdown-layers", type=int, default=8, help="Number of TopDown BasicBlocks.")
    parser.add_argument("--grid-res", type=float, default=GRID_RES, help="Grid resolution.")
    parser.add_argument("--grid-size", type=float, nargs=2, default=list(GRID_SIZE), metavar=("DEPTH", "WIDTH"))
    parser.add_argument("--grid-height", type=float, default=GRID_HEIGHT, help="OFT vertical grid height.")
    parser.add_argument("--y-offset", type=float, default=Y_OFFSET, help="Grid y offset.")
    parser.add_argument("--pad-height", type=int, default=H_PADDED, help="Padded input image height.")
    parser.add_argument("--pad-width", type=int, default=W_PADDED, help="Padded input image width.")
    parser.add_argument("--nms-thresh", type=float, default=NMS_THRESH, help="Decoder NMS threshold.")
    parser.add_argument("--save-vis", default=None, help="Optional path to save projected 3D bounding boxes.")
    args = parser.parse_args()

    run_full_pipeline(
        image_path=args.image,
        calib_path=args.calib,
        checkpoint_path=args.checkpoint,
        frontend_pretrained=args.frontend_pretrained,
        topdown_layers=args.topdown_layers,
        grid_res=args.grid_res,
        grid_size=tuple(args.grid_size),
        grid_height=args.grid_height,
        y_offset=args.y_offset,
        pad_hw=(args.pad_height, args.pad_width),
        nms_thresh=args.nms_thresh,
        output_path=args.save_vis,
    )


if __name__ == "__main__":
    main()
