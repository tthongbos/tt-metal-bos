from model_dev.oft_model.tt.operations import Conv, GroupNormL1, Relu
from model_dev.oft_model.tt.resnet_test import tt_resnet18
from model_dev.oft_model.tt.topdown import TTTopDown
from model_dev.oft_model.tt.tt_oft import ttnn_OFT

import ttnn


class TTOFTNET:
    def __init__(
        self,
        device,
        parameters,
        layer_args,
        mean,
        std,
        input_shape_hw,
        torch_frontend,
        batch_size=1,
        grid_res=0.5,
        grid_height=6.0,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.dtype = dtype

        input_height, input_width = input_shape_hw

        self.frontend = tt_resnet18(
            device,
            torch_frontend,
            input_height=input_height,
            input_width=input_width,
            batch_size=batch_size,
            dtype=dtype,
        )

        self.lat8 = Conv(parameters.lat8, layer_args.lat8, weight_dtype=dtype)
        self.lat16 = Conv(parameters.lat16, layer_args.lat16, weight_dtype=dtype)
        self.lat32 = Conv(parameters.lat32, layer_args.lat32, weight_dtype=dtype)

        self.bn8 = GroupNormL1(parameters.bn8, layer_args.bn8, dtype=dtype)
        self.bn16 = GroupNormL1(parameters.bn16, layer_args.bn16, dtype=dtype)
        self.bn32 = GroupNormL1(parameters.bn32, layer_args.bn32, dtype=dtype)

        self.topdown = TTTopDown(parameters.topdown, layer_args.topdown, dtype=dtype)

        self.oft8 = ttnn_OFT(
            layer_args.oft8,
            parameters.oft8,
            256,
            grid_res,
            grid_height,
            1 / 8.0,
            feature_shape_hw=(layer_args.lat8.input_height, layer_args.lat8.input_width),
        )
        self.oft16 = ttnn_OFT(
            layer_args.oft16,
            parameters.oft16,
            256,
            grid_res,
            grid_height,
            1 / 16.0,
            feature_shape_hw=(layer_args.lat16.input_height, layer_args.lat16.input_width),
        )
        self.oft32 = ttnn_OFT(
            layer_args.oft32,
            parameters.oft32,
            256,
            grid_res,
            grid_height,
            1 / 32.0,
            feature_shape_hw=(layer_args.lat32.input_height, layer_args.lat32.input_width),
        )

        self.head = Conv(parameters.head, layer_args.head, weight_dtype=dtype)

        self.mean = ttnn.from_torch(
            mean,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.std = ttnn.from_torch(
            std,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _capture(self, x):
        return ttnn.to_torch(x).float()

    def forward_normalize(self, input_tensor):
        return ttnn.divide(
            ttnn.sub(input_tensor, self.mean),
            self.std,
        )

    def forward(self, device, image, calib, grid, *, collect_intermediates=True):
        out = {} if collect_intermediates else None
        image = self.forward_normalize(image)
        if collect_intermediates:
            out["norm_image"] = self._capture(image)
        feats8, feats16, feats32, outs = self.frontend.forward_feature_pyramid(
            device,
            image,
            shard="HS",
            collect_intermediates=collect_intermediates,
        )
        if collect_intermediates:
            out["feats8"] = self._capture(feats8)
            out["feats16"] = self._capture(feats16)
            out["feats32"] = self._capture(feats32)
            out["layer1_blocks"] = outs.get("layer1_blocks", {})
            out["layer2_blocks"] = outs.get("layer2_blocks", {})
            out["layer3_blocks"] = outs.get("layer3_blocks", {})
            out["layer4_blocks"] = outs.get("layer4_blocks", {})
        lat8, _, _ = self.lat8(device, feats8, shard="BS")
        lat8 = self.bn8(device, "HS", lat8)
        relu = Relu()
        lat8 = relu(lat8)
        if collect_intermediates:
            out["lat8"] = self._capture(lat8)
        lat16, _, _ = self.lat16(device, feats16, shard="BS")
        lat16 = self.bn16(device, "HS", lat16)

        lat16 = relu(lat16)
        if collect_intermediates:
            out["lat16"] = self._capture(lat16)
        lat32, _, _ = self.lat32(device, feats32, shard="BS")
        lat32 = self.bn32(device, "HS", lat32)
        lat32 = relu(lat32)
        if collect_intermediates:
            out["lat32"] = self._capture(lat32)
        calib_torch = ttnn.to_torch(calib).float()
        grid_torch = ttnn.to_torch(grid).float()

        if collect_intermediates:  # Not clean :vvv
            ortho1, out_or1 = self.oft8(lat8, calib_torch, grid_torch, device, return_intermediates=True)
            ortho2, out_or2 = self.oft16(lat16, calib_torch, grid_torch, device, return_intermediates=True)
            ortho3, out_or3 = self.oft32(lat32, calib_torch, grid_torch, device, return_intermediates=True)
        else:
            ortho1 = self.oft8(lat8, calib_torch, grid_torch, device, return_intermediates=False)
            ortho2 = self.oft16(lat16, calib_torch, grid_torch, device, return_intermediates=False)
            ortho3 = self.oft32(lat32, calib_torch, grid_torch, device, return_intermediates=False)
            out_or1 = out_or2 = out_or3 = None
        if collect_intermediates:
            out["oft8"] = out_or1
            out["oft16"] = out_or2
            out["oft32"] = out_or3
            out["ortho8"] = self._capture(ortho1)
            out["ortho16"] = self._capture(ortho2)
            out["ortho32"] = self._capture(ortho3)
        ortho = ortho1 + ortho2 + ortho3
        if collect_intermediates:
            out["ortho"] = self._capture(ortho)
        topdown, topdown_outs = self.topdown(self.device, ortho, collect_intermediates=collect_intermediates)
        if collect_intermediates:
            out["topdown"] = self._capture(topdown)
            out["topdown_blocks"] = topdown_outs
        output, _, _ = self.head(self.device, topdown, shard="BS")
        if collect_intermediates:
            out["output"] = self._capture(output)
        scores = output[:, :, :, 0:1]
        pos_offsets = output[:, :, :, 1:4]
        dim_offsets = output[:, :, :, 4:7]
        ang_offsets = output[:, :, :, 7:9]
        if collect_intermediates:
            out["scores"] = self._capture(scores)
            out["pos_offsets"] = self._capture(pos_offsets)
            out["dim_offsets"] = self._capture(dim_offsets)
            out["ang_offsets"] = self._capture(ang_offsets)
        return scores, pos_offsets, dim_offsets, ang_offsets, out
