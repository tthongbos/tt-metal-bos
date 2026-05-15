import os
import sys

import torch
import torch.nn.functional as F

import ttnn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


EPSILON = 1e-6


def perspective_and_normalize(matrix, vector, img_size, batch_size=1):
    """
    matrix: [B, 1, 1, 1, 3, 4]
    vector: [B, Y, D, W, 3]

    Returns:
        [B, Y, D, W, 2]
    """

    # if memory_config is None:
    #     memory_config = ttnn.create_sharded_memory_config()
    vector = ttnn.unsqueeze(vector, -1)  # [B, Y, D, W, 3, 1]

    vector_t = ttnn.transpose(vector, -2, -1)  # [B, Y, D, W, 1, 3]

    vector_shape = vector.shape

    vector_t = ttnn.reshape(
        vector_t, (vector_t.shape[0] * vector_t.shape[1] * vector_t.shape[2] * vector_t.shape[3] * 1, 3)
    )  # [B, Y*D*W, 3, 1]

    matrix_t = ttnn.transpose(matrix[..., :-1], -2, -1)

    # Step 2 - Shard in0 into L1 with HEIGHT sharding
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})

    height_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            grid=shard_grid, shard_shape=[204800 // 64, 32], shard_orientation=ttnn.ShardOrientation.ROW_MAJOR
        ),
    )

    vector_sharded = ttnn.to_memory_config(vector_t, memory_config=height_sharded_mem_config)

    vector_sharded = ttnn.to_layout(vector_sharded, layout=ttnn.TILE_LAYOUT)
    matrix_t = ttnn.to_layout(matrix_t, layout=ttnn.TILE_LAYOUT)
    # Step 3 - Matmul Config
    matmul_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,  # K tiles = ceil(3/32) = 1
        out_subblock_h=4,
        out_subblock_w=1,
        per_core_M=100,  # (204800 * 1) / (32 * 64) = 100 tiles
        per_core_N=1,  # N tiles = ceil(3/32) = 1
        fuse_batch=True,  # fold batch dim into M
        fused_activation=None,
        mcast_in0=False,  # in0 HEIGHT-sharded, in1 multicast to all cores
    )

    matrix_t = ttnn.to_memory_config(matrix_t, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Step 4: Define output memory config and call matmul
    out_shard_spec = ttnn.ShardSpec(shard_grid, [3200, 32], ttnn.ShardOrientation.ROW_MAJOR)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard_spec)

    homogeneous = ttnn.matmul(
        vector_sharded,
        matrix_t,  # stays in DRAM (multicast source)
        program_config=matmul_config,
        # memory_config=out_mem_config,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(vector_sharded)
    ttnn.deallocate(matrix_t)

    homogeneous = ttnn.to_layout(homogeneous, layout=ttnn.ROW_MAJOR_LAYOUT)
    homogeneous = ttnn.reshape(homogeneous, (vector_shape[0], vector_shape[1], vector_shape[2], vector_shape[3], 1, 3))
    homogeneous = ttnn.transpose(homogeneous, -2, -1)

    # [B, Y, D, W, 3, 1]

    homogeneous = ttnn.add(homogeneous, matrix[..., -1:])  # [..., 1]
    homogeneous = ttnn.squeeze(homogeneous, -1)  # [..., 1]

    homogeneous = ttnn.divide(ttnn.divide(homogeneous[..., :-1], img_size), homogeneous[..., -1:])

    homogeneous = ttnn.multiply(homogeneous, 2.0)
    homogeneous = ttnn.subtract(homogeneous, 1.0)
    homogeneous = ttnn.clamp(homogeneous, -1.0, 1.0)

    return homogeneous


# TODO: Later on need to unifying the moving to device of the parameters
# to be the work of model_preprocessing. For now, to keep it simple,
# we will move the parameters to device in the forward pass.


class ttnn_OFT:
    def __init__(self, layer_params, model_parameters):
        channels = layer_params["channels"]
        cell_size = layer_params["cell_size"]
        grid_height = layer_params["grid_height"]
        scale = layer_params["scale"]
        self.y_corners = ttnn.from_torch(model_parameters["y_corners"], dtype=ttnn.bfloat8_b)
        self.conv3d_weight = ttnn.from_torch(
            torch.permute(model_parameters["conv3d"]["weight"], (1, 0)), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b
        )
        self.conv3d_bias = (
            ttnn.from_torch(model_parameters["conv3d"]["bias"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
            if model_parameters["conv3d"]["bias"] is not None
            else None
        )
        self.scale = scale

    def __call__(self, features, calib, grid, device):
        features = ttnn.from_torch(features, device=device)
        calib = ttnn.from_torch(calib, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        grid = ttnn.from_torch(grid, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

        grid = ttnn.unsqueeze(grid, 1)

        # print(device.compute_with_storage_grid_size())
        # print(dir(device))
        # Already set as ttnn tensors in init
        self.y_corners = ttnn.to_device(self.y_corners, device=device)
        self.conv3d_weight = ttnn.to_device(self.conv3d_weight, device=device)
        if self.conv3d_bias is not None:
            self.conv3d_bias = ttnn.to_device(self.conv3d_bias, device=device)

        # Expand the grid in the y dimension
        corners = ttnn.add(grid, self.y_corners)

        # # Project grid corners to image plane and normalize to [-1, 1]

        img_height, img_width = features.shape[1], features.shape[2]
        img_size = ttnn.from_torch(torch.tensor([img_width, img_height]) / self.scale, device=device)

        norm_corners = perspective_and_normalize(
            ttnn.to_layout(ttnn.view(calib, (-1, 1, 3, 4)), layout=ttnn.ROW_MAJOR_LAYOUT), corners, img_size
        )

        # Get top-left and bottom-right coordinates of voxel bounding boxes
        bbox_corners = ttnn.concat(
            [
                ttnn.minimum(norm_corners[:, :-1, :-1, :-1], norm_corners[:, :-1, 1:, :-1]),
                ttnn.maximum(norm_corners[:, 1:, 1:, 1:], norm_corners[:, 1:, :-1, 1:]),
            ],
            dim=-1,
        )

        ttnn.deallocate(norm_corners)
        bbox_corners = ttnn.to_layout(bbox_corners, layout=ttnn.ROW_MAJOR_LAYOUT)
        batch, _, depth, width, _ = bbox_corners.shape
        bbox_corners_shape = bbox_corners.shape
        # print("bbox_corners.shape before reshape = ", bbox_corners.shape)
        bbox_corners = ttnn.reshape(
            bbox_corners,
            (
                bbox_corners_shape[0],
                bbox_corners_shape[1],
                bbox_corners_shape[2] * bbox_corners_shape[3],
                bbox_corners_shape[4],
            ),
        )

        # Compute the area of each bounding box
        bbox_corners_area = ttnn.subtract(bbox_corners[..., 2:], bbox_corners[..., :2], dtype=ttnn.bfloat16)
        bbox_corners_area = ttnn.to_memory_config(bbox_corners_area, memory_config=ttnn.L1_MEMORY_CONFIG)

        area = ttnn.prod(bbox_corners_area, dim=-1)
        area = ttnn.multiply(area, img_height * img_width * 0.25)
        area = ttnn.add(area, EPSILON)
        area = ttnn.unsqueeze(area, 1)
        visible = area > EPSILON

        ttnn.deallocate(area)

        # Sample integral image at bounding box locations
        integral_img = integral_image(features)

        bbox_corners = ttnn.to_torch(
            bbox_corners, dtype=torch.float
        )  # grid_sample only works with float, not blocked float

        integral_img = ttnn.to_torch(
            ttnn.permute(integral_img, (0, 3, 1, 2)), dtype=torch.float
        )  # permute to [B, C, H, W] for grid_sample in torch

        top_left = F.grid_sample(integral_img, bbox_corners[..., [0, 1]], align_corners=True)
        btm_right = F.grid_sample(integral_img, bbox_corners[..., [2, 3]], align_corners=True)
        top_right = F.grid_sample(integral_img, bbox_corners[..., [2, 1]], align_corners=True)
        btm_left = F.grid_sample(integral_img, bbox_corners[..., [0, 3]], align_corners=True)

        top_left = ttnn.from_torch(top_left, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
        btm_right = ttnn.from_torch(btm_right, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
        top_right = ttnn.from_torch(top_right, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
        btm_left = ttnn.from_torch(btm_left, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

        # Compute voxel features (ignore features which are not visible)
        vox_feats = top_left + btm_right - top_right - btm_left
        visible_float = ttnn.typecast(visible, ttnn.bfloat16)
        vox_feats = ttnn.multiply(vox_feats, visible_float, dtype=ttnn.bfloat16)
        vox_size = vox_feats.shape
        vox_feats = ttnn.reshape(vox_feats, (vox_size[0], vox_size[1] * vox_size[2], vox_size[3]))
        vox_feats = ttnn.permute(vox_feats, (0, 2, 1))
        # cannot cheat permute performance with data type, error message

        # permute and then reshape then permute back just because of 20ms deduction of device runtime
        vox_feats = ttnn.reshape(vox_feats, (vox_size[0] * vox_size[3], vox_size[1] * vox_size[2]))
        self.conv3d_weight = ttnn.to_memory_config(self.conv3d_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
        self.conv3d_bias = ttnn.to_memory_config(self.conv3d_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
        vox_feats = ttnn.to_layout(vox_feats, layout=ttnn.TILE_LAYOUT)
        ortho_feats = ttnn.linear(
            vox_feats,
            self.conv3d_weight,
            bias=self.conv3d_bias,
            dtype=ttnn.bfloat8_b,
        )

        ortho_feats = ttnn.reshape(ortho_feats, (batch, depth, width, -1))
        ortho_feats = ttnn.relu(ortho_feats)

        return ortho_feats


def integral_image(features):
    features_tile = ttnn.to_layout(features, ttnn.TILE_LAYOUT)
    res = ttnn.cumsum(ttnn.cumsum(features_tile, dim=2), dim=1)
    return ttnn.to_layout(res, ttnn.ROW_MAJOR_LAYOUT)


def torch_integral_image(features):
    return torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)
