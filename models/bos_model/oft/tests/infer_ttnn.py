import json
import time
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from oft import KittiObjectDataset, ObjectEncoder, OFTNet, visualize_objects
from torchvision.transforms.functional import to_tensor


# AOI TODO: Change the code from using torch to TTNN later on
def parse_args():
    # First parse an optional --config to load defaults from JSON
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="path to JSON config file")
    pre_args, _ = pre_parser.parse_known_args()

    # default config path is `config/default_infer_torch.json` next to this file
    if pre_args.config:
        cfg_path = Path(pre_args.config)
    else:
        cfg_path = Path(__file__).resolve().parent / "config" / "default_infer_torch.json"

    config = {}
    if cfg_path.exists():
        with cfg_path.open("r") as f:
            try:
                config = json.load(f)
            except Exception:
                config = {}

    def cfg(key, default=None):
        return config.get(key, default)

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default=str(cfg_path), help="path to JSON config file")

    # positional model path: allow omitted if provided in config
    parser.add_argument(
        "--model-path",
        type=str,
        nargs="?",
        default=cfg("model_path", None),
        help="path to checkpoint file containing trained model",
    )
    parser.add_argument("-g", "--gpu", type=int, default=cfg("gpu", -1), help="gpu to use for inference (-1 for cpu)")

    # Data options
    parser.add_argument(
        "--root", type=str, default=cfg("root", "data/kitti"), help="root directory of the KITTI dataset"
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        nargs=2,
        default=tuple(cfg("grid_size", (80.0, 80.0))),
        help="width and depth of validation grid, in meters",
    )
    parser.add_argument(
        "--yoffset", type=float, default=cfg("yoffset", 1.74), help="vertical offset of the grid from the camera axis"
    )
    parser.add_argument(
        "--nms-thresh", type=float, default=cfg("nms_thresh", 0.5), help="minimum score for a positive detection"
    )

    # Model options
    parser.add_argument(
        "--grid-height", type=float, default=cfg("grid_height", 4.0), help="size of grid cells, in meters"
    )
    parser.add_argument(
        "-r", "--grid-res", type=float, default=cfg("grid_res", 0.5), help="size of grid cells, in meters"
    )
    parser.add_argument(
        "--frontend",
        type=str,
        default=cfg("frontend", "resnet18"),
        choices=["resnet18", "resnet34"],
        help="name of frontend ResNet architecture",
    )
    parser.add_argument(
        "--topdown", type=int, default=cfg("topdown", 8), help="number of residual blocks in topdown network"
    )

    args = parser.parse_args()

    # # enforce model-path presence if not provided by config
    # if args.model_path is None:
    #     parser.error('the following arguments are required: model-path')

    return args


def main():
    # Parse command line arguments
    args = parse_args()
    matplotlib.use("Agg")

    # Load validation dataset to visualise
    dataset = KittiObjectDataset(args.root, "small", args.grid_size, args.grid_res, args.yoffset)

    # Build model
    model = OFTNet(
        num_classes=1,
        frontend=args.frontend,
        topdown_layers=args.topdown,
        grid_res=args.grid_res,
        grid_height=args.grid_height,
    )
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    # Load checkpoint
    ckpt = torch.load(args.model_path, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["model"])

    # Create encoder
    encoder = ObjectEncoder(nms_thresh=args.nms_thresh)
    # TODO: ttnn_OFT model and preprocessing
    # Set up plots
    _, (ax1, ax2) = plt.subplots(nrows=2)
    plt.ion()

    # Iterate over validation images
    for _, image, calib, objects, grid in dataset:
        # Move tensors to gpu
        image = to_tensor(image)
        if args.gpu >= 0:
            image, calib, grid = image.cuda(), calib.cuda(), grid.cuda()

        # Run model forwards
        pred_encoded = model(image[None], calib[None], grid[None])

        # Decode predictions
        pred_encoded = [t[0].cpu() for t in pred_encoded]
        detections = encoder.decode(*pred_encoded, grid.cpu())

        # Visualize predictions
        visualize_objects(image, calib, detections, ax=ax1)
        ax1.set_title("Detections")
        visualize_objects(image, calib, objects, ax=ax2)
        ax2.set_title("Ground truth")

        # Ensure plots directory exists and save figure
        plots_dir = Path(__file__).resolve().parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = plots_dir / f"frame_{_:04d}.png"
        plt.savefig(save_path)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
