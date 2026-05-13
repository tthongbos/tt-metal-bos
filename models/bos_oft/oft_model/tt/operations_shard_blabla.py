import argparse
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.nn as nn
from loguru import logger

import ttnn


def _nearest_32(value):
    return ((value + 31) // 32) * 32


class Conv:
    def __init__(
        self,
        parameters,
        conv_args,
        *,
        act_block_h=32,
        activation=None,
        deallocate=False,
        width_sharding=False,
        height_sharding=False,
        block_sharding=False,
        weight_dtype=ttnn.bfloat8_b,
        output_layout=ttnn.TILE_LAYOUT,
        is_sliced=False,
    ):
        self.weights = self._to_tt_weight(parameters.weight, weight_dtype)
        self.bias = (
            self._to_tt_bias(parameters.bias, weight_dtype) if getattr(parameters, "bias", None) is not None else None
        )
        self.conv_args = conv_args
        self.kernel_size = tuple(parameters.weight.shape[2:])
        self.act_block_h = act_block_h
        self.activation = activation
        self.deallocate = deallocate
        self.weight_dtype = weight_dtype
        self.output_layout = output_layout
        self.shard_layout = self._make_shard_layout(width_sharding, height_sharding, block_sharding)
        self.slice_config = self._make_slice_config(is_sliced)

    @staticmethod
    def _to_tt_weight(weight, dtype):
        if isinstance(weight, ttnn.Tensor):
            return weight
        return ttnn.from_torch(weight, dtype=dtype)

    @staticmethod
    def _to_tt_bias(bias, dtype):
        if isinstance(bias, ttnn.Tensor):
            return bias
        return ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    @staticmethod
    def _make_shard_layout(width_sharding, height_sharding, block_sharding):
        if width_sharding:
            return ttnn.TensorMemoryLayout.WIDTH_SHARDED
        if height_sharding:
            return ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        if block_sharding:
            return ttnn.TensorMemoryLayout.BLOCK_SHARDED
        return None

    @staticmethod
    def _make_slice_config(is_sliced):
        if not is_sliced:
            return ttnn.Conv2dL1FullSliceConfig
        return ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceHeight,
            num_slices=2,
        )

    def __call__(self, device, input_tensor):
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.weight_dtype,
            shard_layout=self.shard_layout,
            output_layout=self.output_layout,
            deallocate_activation=self.deallocate,
            activation=self.activation,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        output, (out_h, out_w), (self.weights, self.bias) = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=self.conv_args.in_channels,
            out_channels=self.conv_args.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=self.conv_args.stride,
            padding=self.conv_args.padding,
            batch_size=self.conv_args.batch_size,
            input_height=self.conv_args.input_height,
            input_width=self.conv_args.input_width,
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            slice_config=self.slice_config,
        )
        return output, out_h, out_w


class GroupNormL1:
    def __init__(
        self,
        parameters,
        layer_args,
        dtype=ttnn.float32,
        is_sliced=False,
    ):
        self.weight = parameters.weight
        self.bias = parameters.bias
        self.num_groups = layer_args.num_groups
        self.channels = layer_args.num_channels
        self.eps = layer_args.eps
        self.dtype = dtype
        self.is_sliced = is_sliced
        self.input_height = layer_args.input_height
        self.input_width = layer_args.input_width

    @contextmanager
    def _split_view(self, num_splits):
        original_num_groups = self.num_groups
        original_channels = self.channels
        self.num_groups //= num_splits
        self.channels //= num_splits
        try:
            yield
        finally:
            self.num_groups = original_num_groups
            self.channels = original_channels

    def _get_core_grid(self, device, sharding):
        compute_grid = device.compute_with_storage_grid_size()
        if sharding == "HS":
            max_cores = max(1, (self.input_height * self.input_width) // 32)
            grid_x = min(compute_grid.x * compute_grid.y, max_cores)
            return ttnn.CoreGrid(y=1, x=max(1, grid_x))
        elif sharding == "BS":
            return ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)
        return ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)

    def _build_masks(self, device, grid_size, use_negative_mask):
        input_mask = ttnn.create_group_norm_input_mask(self.channels, self.num_groups, grid_size.y, ttnn.bfloat16)
        input_mask = ttnn.to_device(input_mask, device)
        negative_mask = None
        if use_negative_mask:
            negative_mask = ttnn.create_group_norm_input_negative_mask(
                self.channels,
                self.num_groups,
                grid_size.y,
                ttnn.bfloat16,
            )
            negative_mask = ttnn.to_device(negative_mask, device)
        return input_mask, negative_mask

    def _build_affine_params(self, device, grid_size):
        gamma = ttnn.create_group_norm_weight_bias_rm(self.weight, self.channels, grid_size.y)
        beta = ttnn.create_group_norm_weight_bias_rm(self.bias, self.channels, grid_size.y)
        common_kwargs = {
            "dtype": ttnn.bfloat16,
            "layout": ttnn.ROW_MAJOR_LAYOUT,
            "device": device,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        }
        return ttnn.from_torch(gamma, **common_kwargs), ttnn.from_torch(beta, **common_kwargs)

    @staticmethod
    def _build_shard_grid(grid_size):
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})

    def _build_sharded_memory_config(self, grid_size, sharding):
        shard_grid = self._build_shard_grid(grid_size)
        if sharding == "HS":
            shard_shape = ((self.input_height * self.input_width) // (grid_size.x * grid_size.y), self.channels)
            shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
            return ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec)
        if sharding == "BS":
            shard_shape = ((self.input_height * self.input_width) // grid_size.x, self.channels // grid_size.y)
            shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
            return ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec)
        raise ValueError(f"Unsupported sharding mode: {sharding}")

    @staticmethod
    def _prepare_input(input_tensor, memory_config, use_negative_mask):
        input_tensor = ttnn.to_memory_config(input_tensor, memory_config=memory_config)
        if use_negative_mask and input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
            input_tensor = ttnn.move(input_tensor)
        return input_tensor

    def __call__(self, device, input_tensor, sharding="HS", negative_mask=False, num_splits=1):
        with self._split_view(num_splits):
            grid_size = self._get_core_grid(device, sharding)

            logger.info(f"GroupNormL1 using core grid: {grid_size}")

            input_mask, input_nmask = self._build_masks(device, grid_size, negative_mask)
            gamma_t, beta_t = self._build_affine_params(device, grid_size)
            sharded_mem_config = self._build_sharded_memory_config(grid_size, sharding)
            input_tensor = self._prepare_input(input_tensor, sharded_mem_config, negative_mask)
            return ttnn.group_norm(
                input_tensor,
                num_groups=self.num_groups,
                input_mask=input_mask,
                negative_mask=input_nmask,
                weight=gamma_t,
                bias=beta_t,
                memory_config=sharded_mem_config,
                core_grid=grid_size,
                epsilon=self.eps,
                inplace=input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT,
            )


class GroupNormDram:
    def __init__(
        self,
        parameters,
        layer_args,
        dtype=ttnn.float32,
        is_sliced=False,
    ):
        self.weight = parameters.weight
        self.bias = parameters.bias
        self.num_groups = layer_args.num_groups
        self.channels = layer_args.num_channels
        self.eps = layer_args.eps
        self.dtype = dtype
        self.is_sliced = is_sliced

    @staticmethod
    def _get_core_grid(device, num_splits):
        compute_grid = device.compute_with_storage_grid_size()
        grid_y = 2 if num_splits > 4 else compute_grid.y
        return ttnn.CoreGrid(y=grid_y, x=4)

    @staticmethod
    def _ensure_tile_layout(input_tensor, grid_size):
        if input_tensor.layout == ttnn.TILE_LAYOUT:
            return input_tensor
        unpadded_shape = input_tensor.shape
        padded_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            _nearest_32(unpadded_shape[2] // grid_size.x) * grid_size.x,
            _nearest_32(unpadded_shape[3] // grid_size.y) * grid_size.y,
        ]
        return ttnn.tilize_with_val_padding(
            input_tensor,
            output_tensor_shape=padded_shape,
            pad_value=0,
            use_multicore=True,
        )

    def _build_dram_params(self, device, grid_size):
        return ttnn.dram_group_norm_params_from_torch(
            [self.weight, self.bias],
            self.channels,
            self.num_groups,
            device,
            core_grid=grid_size,
            return_mask=True,
        )

    def __call__(self, device, input_tensor, num_splits=1):
        grid_size = self._get_core_grid(device, num_splits)
        input_tensor = self._ensure_tile_layout(input_tensor, grid_size)
        (gamma_t, beta_t), input_mask = self._build_dram_params(device, grid_size)
        return ttnn.group_norm(
            input_tensor,
            num_groups=self.num_groups,
            input_mask=input_mask,
            weight=gamma_t,  # y = gamma * x + beta
            bias=beta_t,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_layout=ttnn.TILE_LAYOUT,
            core_grid=grid_size,
            inplace=False,
            num_out_blocks=num_splits,
            epsilon=self.eps,
        )


class Relu:
    def __call__(self, input_tensor):
        return ttnn.relu(input_tensor)


####################
#                  #
#    TEST CODE     #
#                  #
####################


def _open_device(device_id):
    num_devices = ttnn.GetNumPCIeDevices()
    logger.info(f"Found {num_devices} PCIe device(s)")
    if num_devices <= device_id:
        raise RuntimeError(f"Device {device_id} is not available. Found {num_devices} PCIe device(s).")

    print(f"Opening TT device {device_id}")
    device = ttnn.open_device(device_id=device_id)
    print(f"Device arch: {device.arch()}")
    return device


def _basic_conv_test(device_id=0):
    device = _open_device(device_id)
    try:
        torch.manual_seed(0)
        batch_size = 1
        input_height = 8
        input_width = 8
        in_channels = 3
        out_channels = 4

        torch_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        conv_args = SimpleNamespace(
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            stride=(1, 1),
            padding=(0, 0),
        )

        torch_input = torch.randn(batch_size, input_height, input_width, in_channels)
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

        conv = Conv(torch_conv, conv_args, weight_dtype=ttnn.float32)
        output, out_h, out_w = conv(device, tt_input)
        torch_output = ttnn.to_torch(output)
        print("Conv smoke test passed")
        print(f"output shape: {tuple(torch_output.shape)}")
        print(f"out_h/out_w: {out_h}/{out_w}")
    finally:
        print(f"Closing TT device {device_id}")
        ttnn.close_device(device)


def _basic_groupnorm_l1_test(device_id=0):
    device = _open_device(device_id)
    try:
        torch.manual_seed(0)
        batch_size = 1
        input_height = 64
        input_width = 4
        in_channels = 4
        num_channels = 256
        num_groups = 16

        torch_conv = nn.Conv2d(in_channels, num_channels, kernel_size=1, stride=1, padding=0)
        torch_groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6)
        torch_relu = nn.ReLU()
        conv_args = SimpleNamespace(
            in_channels=in_channels,
            out_channels=num_channels,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            stride=(1, 1),
            padding=(0, 0),
        )
        layer_args = SimpleNamespace(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=torch_groupnorm.eps,
            input_height=input_height,
            input_width=input_width,
            height_sharding=True,
        )

        torch_input = torch.randn(batch_size, input_height, input_width, in_channels)
        torch_reference = torch_conv(torch_input.permute(0, 3, 1, 2))
        torch_reference = torch_groupnorm(torch_reference)
        torch_reference = torch_relu(torch_reference)
        torch_reference = torch_reference.permute(0, 2, 3, 1).reshape(
            batch_size, 1, input_height * input_width, num_channels
        )

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        conv = Conv(torch_conv, conv_args, weight_dtype=ttnn.bfloat16)
        logger.info("Initialized Conv operation for GroupNormL1 test")
        groupnorm = GroupNormL1(torch_groupnorm, layer_args, dtype=ttnn.bfloat16)
        relu = Relu()

        tt_output, out_h, out_w = conv(device, tt_input)
        tt_output = groupnorm(device, tt_output, sharding="BS", negative_mask=True, num_splits=1)
        tt_output = relu(tt_output)
        torch_output = ttnn.to_torch(tt_output)

        print("Conv -> GroupNormL1 -> ReLU smoke test passed")
        print(f"input shape: {tuple(torch_input.shape)}")
        print(f"output shape: {tuple(torch_output.shape)}")
        print(f"reference shape: {tuple(torch_reference.shape)}")
        print(f"out_h/out_w: {out_h}/{out_w}")
    finally:
        print(f"Closing TT device {device_id}")
        ttnn.close_device(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run basic TTNN operation smoke tests.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--test",
        choices=("conv", "groupnorm-l1"),
        default="groupnorm-l1",
        help="Pick which smoke test to run.",
    )
    args = parser.parse_args()
    if args.test == "conv":
        _basic_conv_test(device_id=args.device_id)
    else:
        _basic_groupnorm_l1_test(device_id=args.device_id)
