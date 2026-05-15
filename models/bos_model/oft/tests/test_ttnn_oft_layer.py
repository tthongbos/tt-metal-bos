# For testing if the module is working correctly
# Run the test from /workspace/oft please
# with python oft/model/ttnn/ttnn_oft.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


import json

import torch
from bos_model.oft.reference.oft.data.kitti import KittiObjectDataset
from bos_model.oft.reference.oft.model import OFT
from bos_model.oft.reference.oft.model import utils as utils
from bos_model.oft.reference.oft.model.utils import make_grid
from bos_model.oft.tt.ttnn_oft import ttnn_OFT

import ttnn


def main():
    # Load validation dataset to visualise
    # need to be specified with the real path to the dataset

    kitti_root = "/workspace/oft/data/kitti"
    split = "single"
    split_file = os.path.join(os.path.dirname(__file__), f"../reference/oft/data/splits/{split}.txt")

    assert os.path.exists(kitti_root), f"Dataset path '{kitti_root}' does not exist."
    assert os.path.exists(split_file), f"Split text file '{split_file}' does not exist."

    dataset = KittiObjectDataset(kitti_root, split, (80.0, 80.0), 0.5, 1.74)

    with open("models/bos_model/oft/tests/test_configs/oft_test_params.json", "r") as f:
        test_parameters = json.load(f)

    # Initialize the first Tenstorrent device (device_id=0)
    device = ttnn.open_device(device_id=0)

    # Randomized params and weights
    torch_oft = OFT(**test_parameters["layer_params"])

    model_parameters = {
        "y_corners": torch_oft.y_corners,
        "conv3d": {"weight": torch_oft.conv3d.weight, "bias": torch_oft.conv3d.bias},
    }

    ttnn_oft = ttnn_OFT(layer_params=test_parameters["layer_params"], model_parameters=model_parameters)

    for _, image, calib, objects, grid in dataset:
        #     with trace():
        #         features = torch.rand(1, 47, 156, 256) # [B, H, W, C] ttnn type
        #         calib = calib
        #         grid = make_grid((79.0, 80.0), (-40., -1.74, 0.0), 0.5).unsqueeze(0)

        #         # Calculate oft in tnnn
        #         ortho_feats_ttnn = ttnn_oft(features, calib, grid, device=device)
        #         ortho_feats_torch = torch_oft(features.permute(0, 3, 1, 2), calib, grid) # permute features to [B, C, H, W] for PyTorch implementation
        #         print(ortho_feats_torch.shape)
        #         print(ortho_feats_ttnn.shape)
        #         utils.comparing_torch_ttnn(ortho_feats_torch.permute(0, 2, 3, 1), ortho_feats_ttnn)

        # return visualize(ortho_feats_ttnn,  file_name="./tracer_demo/graph_oft.svg")
        features = torch.rand(1, 47, 156, 256)  # [B, H, W, C] ttnn type
        calib = calib
        grid = make_grid((80.0, 80.0), (-40.0, -1.74, 0.0), 0.5).unsqueeze(0)

        # Calculate oft in tnnn
        ortho_feats_ttnn = ttnn_oft(features, calib, grid, device=device)
        ortho_feats_torch = torch_oft(
            features.permute(0, 3, 1, 2), calib, grid
        )  # permute features to [B, C, H, W] for PyTorch implementation
        print(ortho_feats_torch.shape)
        print(ortho_feats_ttnn.shape)
        utils.comparing_torch_ttnn(ortho_feats_torch.permute(0, 2, 3, 1), ortho_feats_ttnn)


if __name__ == "__main__":
    main()
