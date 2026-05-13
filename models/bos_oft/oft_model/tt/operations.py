import argparse
from types import SimpleNamespace

import torch
import torch.nn as nn
from loguru import logger

import ttnn


def make_hs_mem_config_tile_aligned(device, input_nhw, channels, shard):
    compute_grid = device.compute_with_storage_grid_size()
    max_x = compute_grid.x  # 5
    max_y = compute_grid.y  # 4

    best_x = None
    best_y = None
    best_num_cores = None
    if shard == "HS":
        for y in range(max_y, 0, -1):
            for x in range(max_x, 0, -1):
                num_cores = x * y
                if input_nhw % num_cores != 0:
                    continue

                shard_h = input_nhw // num_cores

                if shard_h % 32 != 0:
                    continue

                best_x = x
                best_y = y
                best_num_cores = num_cores
                break

            if best_num_cores is not None:
                break

        if best_num_cores is None:
            raise RuntimeError(
                f"Cannot find tile-aligned HS grid for input_nhw={input_nhw}, "
                f"channels={channels}, physical_grid=({max_x},{max_y})"
            )
        shard_shape = (
            input_nhw // best_num_cores,
            channels,
        )
        grid_size = ttnn.CoreGrid(x=best_x, y=best_y)

        shard_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}
        )

        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)

        sharded_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.types.BufferType.L1,
            shard_spec,
        )
    elif shard == "BS":
        for y in range(max_y, 0, -1):
            for x in range(max_x, 0, -1):
                num_cores = x * y
                if input_nhw % num_cores != 0:
                    continue

                shard_h = input_nhw // x
                shard_w = channels // y

                if shard_h % 32 != 0:
                    continue

                if shard_w % 32 != 0:
                    continue

                best_x = x
                best_y = y
                best_num_cores = num_cores
                shard_shape = (shard_h, shard_w)
                break

            if best_num_cores is not None:
                break

        if best_num_cores is None:
            raise RuntimeError(
                f"Cannot find tile-aligned BS grid for input_nhw={input_nhw}, "
                f"channels={channels}, physical_grid=({max_x},{max_y})"
            )
        grid_size = ttnn.CoreGrid(x=best_x, y=best_y)
        shard_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}
        )
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
        sharded_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.types.BufferType.L1,
            shard_spec,
        )
    return sharded_mem_config, grid_size

    return sharded_mem_config, grid_size


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
        weight_dtype=ttnn.bfloat8_b,
        output_layout=ttnn.TILE_LAYOUT,
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

    @staticmethod
    def _shard_config(shard):
        if shard == "HS":
            return ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        elif shard == "BS":
            return ttnn.TensorMemoryLayout.BLOCK_SHARDED
        elif shard == "WS":
            return ttnn.TensorMemoryLayout.WIDTH_SHARDED

    @staticmethod
    def _to_tt_weight(weight, dtype):
        if isinstance(weight, ttnn.Tensor):
            return weight
        storage_dtype = ttnn.bfloat16 if dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b) else dtype
        return ttnn.from_torch(weight, dtype=storage_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    @staticmethod
    def _to_tt_bias(bias, dtype):
        if isinstance(bias, ttnn.Tensor):
            return bias
        storage_dtype = ttnn.bfloat16 if dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b) else dtype
        return ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=storage_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    def __call__(self, device, input_tensor, shard):
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.weight_dtype,
            output_layout=self.output_layout,
            deallocate_activation=self.deallocate,
            activation=self.activation,
            shard_layout=self._shard_config(shard),
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
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
        )
        return output, out_h, out_w


class GroupNormL1:
    def __init__(self, parameters, layer_args, dtype=ttnn.bfloat16, is_sliced=False):
        self.weight = parameters.weight
        self.bias = parameters.bias
        self.num_groups = layer_args.num_groups
        self.channels = layer_args.num_channels
        self.eps = layer_args.eps
        self.dtype = dtype
        self.input_height = layer_args.input_height
        self.input_width = layer_args.input_width
        self.batch_size = getattr(layer_args, "batch_size", 1)

    def __call__(self, device, shard, input_tensor, num_splits=1):
        compute_grid = device.compute_with_storage_grid_size()
        grid_size = grid_size = ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)
        grid_y = grid_size.y
        grid_x = grid_size.x
        if shard == "HS":
            grid_y = 1
            grid_x *= grid_size.y
        channels = self.channels
        self.num_groups = self.num_groups // num_splits
        self.channels = self.channels // num_splits

        input_nhw = input_tensor.shape[2]
        channels = input_tensor.shape[3]

        sharded_mem_config, grid_size = make_hs_mem_config_tile_aligned(
            device=device, input_nhw=input_nhw, channels=channels, shard=shard
        )

        num_cores_across_channel = ttnn.get_group_norm_cores_across_channel(
            sharded_mem_config.memory_layout,
            grid_size,
            sharded_mem_config.shard_spec.orientation,
        )
        input_mask_tensor = ttnn.create_group_norm_input_mask(
            self.channels, self.num_groups, num_cores_across_channel, ttnn.bfloat16
        )
        input_mask_tensor = ttnn.to_device(input_mask_tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG)
        negative_input_mask_tensor = ttnn.create_group_norm_input_negative_mask(
            self.channels, self.num_groups, num_cores_across_channel, ttnn.bfloat16
        )
        negative_input_mask_tensor = ttnn.to_device(
            negative_input_mask_tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        gamma = ttnn.create_group_norm_weight_bias_rm(self.weight, self.channels, num_cores_across_channel)
        beta = ttnn.create_group_norm_weight_bias_rm(self.bias, self.channels, num_cores_across_channel)
        gamma_t = ttnn.from_torch(
            gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        beta_t = ttnn.from_torch(
            beta,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        input_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_mem_config)
        if negative_input_mask_tensor and input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
            input_tensor = ttnn.move(input_tensor)

        out = ttnn.group_norm(
            input_tensor,
            num_groups=self.num_groups,
            input_mask=input_mask_tensor,
            negative_mask=negative_input_mask_tensor,
            weight=gamma_t,  # y = gamma * x + beta
            bias=beta_t,
            core_grid=grid_size,
            inplace=True if input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT else False,
            epsilon=self.eps,
        )
        self.num_groups = self.num_groups * num_splits
        self.channels = self.channels * num_splits
        return out


class GroupNormDram:
    def __init__(
        self,
        parameters,
        layer_args,
        dtype=ttnn.bfloat8_b,
    ):
        self.weight = parameters.weight
        self.bias = parameters.bias
        self.num_groups = layer_args.num_groups
        self.channels = layer_args.num_channels
        self.eps = layer_args.eps
        self.input_height = layer_args.input_height
        self.input_width = layer_args.input_width
        self.batch_size = getattr(layer_args, "batch_size", 1)
        self.dtype = dtype

    def _get_grid_from_tensor(self, device, input_tensor, num_groups):
        shape = input_tensor.shape
        batch_size = int(shape[0])
        input_nhw = int(shape[0]) * int(shape[1]) * int(shape[2])
        channels = int(shape[3])

        return ttnn.determine_expected_group_norm_dram_grid_size(
            device=device,
            num_channels=channels,
            num_groups=num_groups,
            input_nhw=input_nhw,
            num_batches=batch_size,
        )

    def _get_num_out_blocks(self, grid_size):
        block_h = _nearest_32(self.input_height * self.input_width) // 32
        block_h = block_h // grid_size.y
        if block_h >= 64:
            return 32
        if block_h >= 32:
            return 16
        if block_h >= 8:
            return 8
        if block_h >= 4:
            return 4
        return 1

    def __call__(self, device, input_tensor, num_splits=1):
        grid_size = self._get_grid_from_tensor(
            device,
            input_tensor,
            self.num_groups,
        )

        # input_tensor = ttnn.typecast(input_tensor, self.dtype)
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
        if input_tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            input_tensor = ttnn.to_memory_config(
                input_tensor,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        if input_tensor.dtype != ttnn.bfloat16:
            input_tensor = ttnn.typecast(input_tensor, ttnn.bfloat16)
        [gamma_t, beta_t], input_mask = ttnn.dram_group_norm_params_from_torch(
            [self.weight, self.bias],
            self.channels,
            self.num_groups,
            device,
            core_grid=grid_size,
            return_mask=True,
            dtype=ttnn.bfloat16,
        )

        out = ttnn.group_norm(
            input_tensor,
            num_groups=self.num_groups,
            input_mask=input_mask,
            weight=gamma_t,  # y = gamma * x + beta
            bias=beta_t,
            # memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_layout=ttnn.TILE_LAYOUT,
            core_grid=grid_size,
            inplace=False,
            num_out_blocks=self._get_num_out_blocks(grid_size),
            epsilon=self.eps,
        )
        return out


class Relu:
    def __call__(self, input_tensor):
        return ttnn.relu(input_tensor, memory_config=input_tensor.memory_config())


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
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

        conv = Conv(torch_conv, conv_args, weight_dtype=ttnn.bfloat8_b)
        output, out_h, out_w = conv(device, tt_input)
        torch_output = ttnn.to_torch(output)
        print("Conv smoke test passed")
        print(f"output shape: {tuple(torch_output.shape)}")
        print(f"out_h/out_w: {out_h}/{out_w}")
    finally:
        print(f"Closing TT device {device_id}")
        ttnn.close_device(device)


def _basic_groupnorm_test(device_id=0):
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
        )

        torch_input = torch.randn(batch_size, input_height, input_width, in_channels)
        torch_reference = torch_conv(torch_input.permute(0, 3, 1, 2))
        torch_reference = torch_groupnorm(torch_reference)
        torch_reference = torch_relu(torch_reference)
        torch_reference = torch_reference.permute(0, 2, 3, 1).reshape(
            batch_size, 1, input_height * input_width, num_channels
        )

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        conv = Conv(torch_conv, conv_args, weight_dtype=ttnn.bfloat8_b)
        logger.info("Initialized Conv operation for GroupNorm test")
        groupnorm = GroupNormDram(torch_groupnorm, layer_args, dtype=ttnn.bfloat8)
        relu = Relu()

        tt_output, out_h, out_w = conv(device, tt_input)
        tt_output = groupnorm(device, tt_output, num_splits=1)
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
        _basic_groupnorm_test(device_id=args.device_id)
