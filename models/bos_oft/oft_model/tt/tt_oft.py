import torch
import torch.nn.functional as F

import ttnn

EPSILON = 1e-6


def _get_arg(layer_params, name, default=None):
    if isinstance(layer_params, dict):
        return layer_params.get(name, default)
    return getattr(layer_params, name, default)


def perspective(matrix, vector):
    """
    matrix: [B, 1, 1, 1, 3, 4]
    vector: [B, Y, D, W, 3]

    Returns:
        [B, Y, D, W, 2]
    """
    vector = ttnn.unsqueeze(vector, -1)  # [B, Y, D, W, 3, 1]
    vector_shape = vector.shape
    vector = ttnn.reshape(
        vector, (vector.shape[0], vector.shape[1] * vector.shape[2] * vector.shape[3], 3, 1)
    )  # [B, Y*D*W, 3, 1]
    # vector = ttnn.permute(vector, (1, 0, 2, 3)) # [Y*D*W, B, 3, 1]

    matrix = ttnn.reshape(matrix, (matrix.shape[0], 1, 3, 4))  # [B, 1, 3, 4]

    # homogeneous = ttnn.matmul(ttnn.transpose(vector, -2, -1), ttnn.transpose(matrix[..., :-1], -2, -1), dtype=ttnn.bfloat16) # [B, Y, D, W, 1, 3]
    homogeneous = ttnn.matmul(vector, matrix[..., :-1], transpose_a=True, transpose_b=True)  # [B, Y, D, W, 1, 3]
    homogeneous = ttnn.permute(homogeneous, (1, 0, 3, 2))
    homogeneous = ttnn.reshape(
        homogeneous, (vector_shape[0], vector_shape[1], vector_shape[2], vector_shape[3], 3, 1)
    )  # [B, Y, D, W, 3, 1]

    # torch_homogeneous = utils.perspective(matrix_torch, vector_torch)
    homogeneous = ttnn.add(homogeneous, matrix[..., -1:])  # [..., 1]

    homogeneous = ttnn.squeeze(homogeneous, -1)  # [..., 1]
    ttnn_homogeneous = homogeneous[..., :-1] / homogeneous[..., -1:]

    return ttnn_homogeneous


def perspective_and_normalize(matrix, vector, img_size):
    """
    matrix: [B, 1, 1, 1, 3, 4]
    vector: [B, Y, D, W, 3]

    Returns:
        [B, Y, D, W, 2]
    """
    vector = ttnn.unsqueeze(vector, -1)  # [B, Y, D, W, 3, 1]

    vector_shape = vector.shape
    vector = ttnn.reshape(
        vector, (vector.shape[0], vector.shape[1] * vector.shape[2] * vector.shape[3], 3, 1)
    )  # [B, Y*D*W, 3, 1]
    vector = ttnn.permute(vector, (1, 0, 2, 3))  # [B, Y*D*W, 3, 1]

    matrix = ttnn.reshape(matrix, (matrix.shape[0], 1, 3, 4))  # [B, 1, 3, 4]

    homogeneous = ttnn.matmul(
        ttnn.transpose(vector, -2, -1), ttnn.transpose(matrix[..., :-1], -2, -1), dtype=ttnn.bfloat16
    )  # [B, Y, D, W, 1, 3]
    homogeneous = ttnn.permute(homogeneous, (1, 0, 2, 3))
    homogeneous = ttnn.transpose(homogeneous, -2, -1)  # [B, Y, D, W, 3, 1]
    homogeneous = ttnn.reshape(
        homogeneous, (vector_shape[0], vector_shape[1], vector_shape[2], vector_shape[3], 3, 1)
    )  # [B, Y, D, W, 3, 1]

    # torch_homogeneous = utils.perspective(matrix_torch, vector_torch)
    homogeneous = ttnn.add(homogeneous, matrix[..., -1:])  # [..., 1]

    homogeneous = ttnn.squeeze(homogeneous, -1)  # [..., 1]
    ttnn_homogeneous = ttnn.divide(homogeneous[..., :-1], img_size) / homogeneous[..., -1:]
    ttnn_homogeneous = ttnn.multiply(ttnn_homogeneous, 2.0)
    ttnn_homogeneous = ttnn.subtract(ttnn_homogeneous, 1.0)
    ttnn_homogeneous = ttnn.clamp(ttnn_homogeneous, -1.0, 1.0)

    return ttnn_homogeneous

    # # Split projection matrix into A and b
    # A = matrix[:, :, :, :, :, :-1]      # [B, 1, 1, 1, 3, 3]
    # b = matrix[:, :, :, :, :, -1:]      # [B, 1, 1, 1, 3, 1]

    # # Convert vector from column vector to row vector
    # v_col = ttnn.unsqueeze(vector, -1)  # [B, Y, D, W, 3, 1]
    # v_row = ttnn.transpose(v_col, -2, -1)  # [B, Y, D, W, 1, 3]

    # # Transpose matrix and bias
    # A_t = ttnn.transpose(A, -2, -1)     # [B, 1, 1, 1, 3, 3]
    # b_t = ttnn.transpose(b, -2, -1)     # [B, 1, 1, 1, 1, 3]

    # # Compute v^T @ A^T + b^T
    # homogenous_row = ttnn.matmul(matrix[], A_t)  # [B, Y, D, W, 1, 3]
    # homogenous_row = ttnn.add(homogenous_row, b_t)

    # # return homogenous_row
    # # Remove the row dimension
    # homogenous = ttnn.squeeze(homogenous_row, -2)  # [B, Y, D, W, 3]
    # return homogenous


# TODO: Later on need to unifying the moving to device of the parameters
# to be the work of model_preprocessing. For now, to keep it simple,
# we will move the parameters to device in the forward pass.


class ttnn_OFT:
    def __init__(
        self,
        layer_params,
        model_parameters,
        channels=None,
        cell_size=None,
        grid_height=None,
        scale=None,
        feature_shape_hw=None,
    ):
        channels = channels if channels is not None else _get_arg(layer_params, "channels")
        cell_size = cell_size if cell_size is not None else _get_arg(layer_params, "cell_size")
        grid_height = grid_height if grid_height is not None else _get_arg(layer_params, "grid_height")
        scale = scale if scale is not None else _get_arg(layer_params, "scale")
        self.channels = channels
        self.feature_shape_hw = feature_shape_hw
        y_corners = torch.arange(0, grid_height, cell_size) - grid_height / 2.0
        y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])
        self.y_corners = ttnn.from_torch(y_corners, dtype=ttnn.bfloat16)
        conv3d_weight = model_parameters["conv3d"]["weight"]
        if not isinstance(conv3d_weight, torch.Tensor):
            conv3d_weight = ttnn.to_torch(conv3d_weight)
        self.conv3d_weight = ttnn.from_torch(conv3d_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        self.conv3d_bias = (
            ttnn.from_torch(
                model_parameters["conv3d"]["bias"]
                if isinstance(model_parameters["conv3d"]["bias"], torch.Tensor)
                else ttnn.to_torch(model_parameters["conv3d"]["bias"]),
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
            )
            if model_parameters["conv3d"]["bias"] is not None
            else None
        )
        self.scale = scale

    def __call__(self, features, calib, grid, device, return_intermediates=False):
        outs = {} if return_intermediates else None

        def record(name, value):
            if outs is not None:
                outs[name] = value

        if not isinstance(features, ttnn.Tensor):
            features = ttnn.from_torch(features)
        if self.feature_shape_hw is not None:
            batch_size = int(features.shape[0])
            feature_height, feature_width = self.feature_shape_hw
            if features.is_sharded():
                features = ttnn.sharded_to_interleaved(features, ttnn.DRAM_MEMORY_CONFIG)
            features = ttnn.reshape(features, (batch_size, feature_height, feature_width, self.channels))
        calib = ttnn.from_torch(calib, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        grid_unsqueeze = grid.unsqueeze(1)
        grid = ttnn.from_torch(grid_unsqueeze, dtype=ttnn.bfloat16)
        features = ttnn.to_device(features, device=device)

        calib = ttnn.to_device(calib, device=device)
        grid = ttnn.to_device(grid, device=device)
        record("features", ttnn.to_torch(ttnn.permute(features, (0, 3, 1, 2))).float())

        # Already set as ttnn tensors in init
        self.y_corners = ttnn.to_device(self.y_corners, device=device)
        self.conv3d_weight = ttnn.to_device(self.conv3d_weight, device=device)
        if self.conv3d_bias is not None:
            self.conv3d_bias = ttnn.to_device(self.conv3d_bias, device=device)

        # Expand the grid in the y dimension
        corners = ttnn.add(grid, self.y_corners)
        corners = ttnn.to_layout(corners, ttnn.TILE_LAYOUT)

        # # Project grid corners to image plane and normalize to [-1, 1]
        # img_corners = perspective(ttnn.to_layout(ttnn.view(calib, (-1, 1, 1, 1, 3, 4)), layout=ttnn.TILE_LAYOUT), corners)
        # img_corners = ttnn.typecast(img_corners, ttnn.bfloat16)

        # # Normalize to [-1, 1]
        # img_height, img_width = features.shape[1], features.shape[2]
        # img_size = torch.tensor([img_width, img_height]) / self.scale

        # torch_img_corners = ttnn.to_torch(img_corners)

        # norm_corners = ttnn.from_torch(torch_img_corners / img_size, device=device)

        # norm_corners = ttnn.clamp(ttnn.subtract(ttnn.multiply(norm_corners, 2.0), 1.0), -1, 1)
        img_height, img_width = features.shape[1], features.shape[2]
        img_size = ttnn.from_torch(torch.tensor([img_width, img_height]) / self.scale, device=device)

        norm_corners = perspective_and_normalize(
            ttnn.to_layout(ttnn.view(calib, (-1, 1, 1, 1, 3, 4)), layout=ttnn.TILE_LAYOUT), corners, img_size
        )
        record("corners", ttnn.to_torch(corners).float())
        record("norm_corners", ttnn.to_torch(norm_corners).float())

        # Get top-left and bottom-right coordinates of voxel bounding boxes
        bbox_corners = ttnn.concat(
            [
                ttnn.minimum(norm_corners[:, :-1, :-1, :-1], norm_corners[:, :-1, 1:, :-1]),
                ttnn.maximum(norm_corners[:, 1:, 1:, 1:], norm_corners[:, 1:, :-1, 1:]),
            ],
            dim=-1,
        )

        batch, _, depth, width, _ = bbox_corners.shape
        bbox_corners_shape = bbox_corners.shape
        bbox_corners = ttnn.reshape(
            bbox_corners,
            (
                bbox_corners_shape[0],
                bbox_corners_shape[1],
                bbox_corners_shape[2] * bbox_corners_shape[3],
                bbox_corners_shape[4],
            ),
        )
        record(
            "bbox_corners_pre_flatten",
            ttnn.to_torch(
                ttnn.reshape(
                    bbox_corners,
                    (
                        bbox_corners_shape[0],
                        bbox_corners_shape[1],
                        bbox_corners_shape[2],
                        bbox_corners_shape[3],
                        bbox_corners_shape[4],
                    ),
                )
            ).float(),
        )
        record("bbox_corners", ttnn.to_torch(bbox_corners).float())
        # grid_sample only works with float16 or float32, and bfloat16 is more efficient on ttnn, so we use that. The loss of precision should be acceptable since the coordinates are normalized to [-1, 1]
        # bbox_corners = ttnn.typecast(bbox_corners, ttnn.float8) # grid_sample only works with float32

        # Compute the area of each bounding box
        bbox_corners_area = ttnn.subtract(bbox_corners[..., 2:], bbox_corners[..., :2], dtype=ttnn.bfloat16)

        area = ttnn.prod(bbox_corners_area, dim=-1)

        area = ttnn.multiply(area, img_height * img_width * 0.25)
        area = ttnn.add(area, EPSILON)
        area = ttnn.unsqueeze(area, 1)
        visible = area > EPSILON
        record("area", ttnn.to_torch(area).float())

        features = ttnn.typecast(features, ttnn.float32)
        # Sample integral image at bounding box locations
        integral_img = integral_image(features)
        record("integral_img", ttnn.to_torch(ttnn.permute(integral_img, (0, 3, 1, 2))).float())

        bbox_corners = ttnn.to_torch(
            bbox_corners, dtype=torch.float
        )  # grid_sample only works with float, not blocked float
        integral_img = ttnn.to_torch(
            ttnn.permute(integral_img, (0, 3, 1, 2)), dtype=torch.float
        )  # permute to [B, C, H, W] for grid_sample in torch

        top_left = F.grid_sample(integral_img, bbox_corners[..., [0, 1]], align_corners=False)
        btm_right = F.grid_sample(integral_img, bbox_corners[..., [2, 3]], align_corners=False)
        top_right = F.grid_sample(integral_img, bbox_corners[..., [2, 1]], align_corners=False)
        btm_left = F.grid_sample(integral_img, bbox_corners[..., [0, 3]], align_corners=False)

        top_left = ttnn.permute(ttnn.from_torch(top_left, device=device), (0, 2, 3, 1))  # permute back to [B, H, W, C]
        btm_right = ttnn.permute(ttnn.from_torch(btm_right, device=device), (0, 2, 3, 1))
        top_right = ttnn.permute(ttnn.from_torch(top_right, device=device), (0, 2, 3, 1))
        btm_left = ttnn.permute(ttnn.from_torch(btm_left, device=device), (0, 2, 3, 1))
        print("top_left.dtype", top_left.dtype)
        print("btm_right.dtype", btm_right.dtype)
        print("top_right.dtype", top_right.dtype)
        print("btm_left.dtype", btm_left.dtype)
        record("top_left", top_left)
        record("btm_right", btm_right)
        record("top_right", top_right)
        record("btm_left", btm_left)
        # integral_img = ttnn.to_layout(integral_img, layout=ttnn.ROW_MAJOR_LAYOUT)

        # bbox_corners = ttnn.to_layout(bbox_corners, layout=ttnn.ROW_MAJOR_LAYOUT)
        # integral_img = ttnn.typecast(integral_img, ttnn.float32)
        # bbox_corners = ttnn.typecast(bbox_corners, ttnn.float32)

        # top_left = ttnn.grid_sample(integral_img, bbox_corners[:, :, :, 0:2], align_corners=True)
        # btm_right = ttnn.grid_sample(integral_img, bbox_corners[:, :, :, 2:4], align_corners=True)

        # x1 = bbox_corners[:, :, :, 0:1]
        # y1 = bbox_corners[:, :, :, 1:2]
        # x2 = bbox_corners[:, :, :, 2:3]
        # y2 = bbox_corners[:, :, :, 3:4]
        # top_right_coords = ttnn.concat([x2, y1], dim=-1)
        # btm_left_coords = ttnn.concat([x1, y2], dim=-1)

        # top_right = ttnn.grid_sample(integral_img, top_right_coords, align_corners=True)
        # btm_left = ttnn.grid_sample(integral_img, btm_left_coords, align_corners=True)

        # Compute voxel features (ignore features which are not visible)
        rect_sum = top_left + btm_right - top_right - btm_left
        record("rect_sum", rect_sum)
        area_for_div = ttnn.permute(area, (0, 2, 3, 1))  # [B, Y, D*W, 1]
        record("area_for_div", area_for_div)
        vox_feats = ttnn.divide(rect_sum, area_for_div)
        record("vox_avg", vox_feats)
        visible_float = ttnn.typecast(visible, ttnn.float32)
        visible_float = ttnn.permute(visible_float, (0, 2, 3, 1))  # [B, Y, D*W, 1]

        vox_feats = ttnn.multiply(vox_feats, visible_float, dtype=ttnn.float32)
        vox_feats = ttnn.permute(vox_feats, (0, 2, 3, 1))  # [B, D*W, C, Y]
        vox_size = vox_feats.shape
        vox_feats = ttnn.reshape(vox_feats, (vox_size[0] * vox_size[1], vox_size[2], vox_size[3]))
        vox_size = vox_feats.shape
        vox_feats = ttnn.reshape(vox_feats, (vox_size[0], vox_size[1] * vox_size[2]))
        record("vox_feats_flat", ttnn.to_torch(vox_feats).float())
        if vox_feats.layout != ttnn.TILE_LAYOUT:
            vox_feats = ttnn.to_layout(vox_feats, ttnn.TILE_LAYOUT)
        # Flatten to orthographic feature map
        ortho_feats = ttnn.linear(vox_feats, self.conv3d_weight, bias=self.conv3d_bias, dtype=ttnn.bfloat16)
        ortho_feats = ttnn.reshape(ortho_feats, (batch, depth, width, -1))
        record("conv3d", ttnn.to_torch(ortho_feats).float())
        ortho_feats = ttnn.relu(ortho_feats)
        record("out", ttnn.to_torch(ttnn.permute(ortho_feats, (0, 3, 1, 2))).float())
        ortho_feats = ttnn.reshape(ortho_feats, (batch, 1, depth * width, self.channels))

        if return_intermediates:
            return ortho_feats, outs
        return ortho_feats


def integral_image(features):
    features_tile = ttnn.to_layout(features, ttnn.TILE_LAYOUT)

    res = ttnn.cumsum(ttnn.cumsum(features_tile, dim=2), dim=1)

    return ttnn.to_layout(res, ttnn.ROW_MAJOR_LAYOUT)


def torch_integral_image(features):
    return torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)
