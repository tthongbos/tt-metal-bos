import sys
from pathlib import Path

import ttnn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TT_METAL_ROOT = Path("/home/phutruong/tt-metal")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if TT_METAL_ROOT.exists() and str(TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TT_METAL_ROOT))

from model_dev.oft_model.tt.operations import Conv, GroupNormL1, Relu

from models.tt_cnn.tt.builder import MaxPool2dConfiguration, TtMaxPool2d

MODEL_URLS = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
}


def _load_pretrained(model, pretrained_state_dict):
    model_dict = model.state_dict()
    dtype = next(model.parameters()).dtype
    pretrained_state_dict = {
        key: value.to(dtype) if value.is_floating_point() else value
        for key, value in pretrained_state_dict.items()
        if key in model_dict
    }
    model_dict.update(pretrained_state_dict)
    model.load_state_dict(model_dict)


class TTBasicBlock:
    expansion = 1
    block_id = 1

    def __init__(self, parameters, layer_args, *, dtype=ttnn.bfloat8_b):
        self.block_id = TTBasicBlock.block_id
        TTBasicBlock.block_id += 1
        self.conv1 = Conv(parameters.conv1, layer_args.conv1, weight_dtype=dtype)
        self.conv2 = Conv(parameters.conv2, layer_args.conv2, weight_dtype=dtype)
        self.bn1 = GroupNormL1(parameters.bn1, layer_args.bn1, dtype=dtype)
        self.bn2 = GroupNormL1(parameters.bn2, layer_args.bn2, dtype=dtype)
        self.relu = Relu()

        self.downsample = hasattr(parameters, "downsample") and getattr(parameters, "downsample") is not None
        if self.downsample:
            self.downsample_conv = Conv(parameters.downsample[0], layer_args.downsample[0], weight_dtype=dtype)
            self.downsample_bn = GroupNormL1(parameters.downsample[1], layer_args.downsample[1], dtype=dtype)

    @staticmethod
    def _capture(x):
        return ttnn.to_torch(x).float()

    @staticmethod
    def _match_identity_to_out(identity, out):
        if identity.memory_config() != out.memory_config():
            identity = ttnn.to_memory_config(identity, memory_config=out.memory_config())
        if identity.layout != out.layout:
            identity = ttnn.to_layout(identity, out.layout)
        return identity

    # logger.info(f"Block id {block.block_id} with output in: {x.memory_config()} layout: {x.layout} shape: {x.shape}")
    def forward(self, device, x, *, num_splits=1, collect_intermediates=True):
        outs = {}
        identity = x

        # logger.info(f"Block id {self.block_id}: conv1 start")
        out, _, _ = self.conv1(device, x)
        # out = ttnn.move(out)
        if collect_intermediates:
            outs["conv1"] = self._capture(out)
        # logger.info(f"Block id {self.block_id} with output in: {out.memory_config()} layout: {out.layout} shape: {out.shape}")
        # logger.info(f"Block id {self.block_id}: bn1 start")
        out = self.bn1(device, out, num_splits=num_splits)
        if collect_intermediates:
            outs["grnorm1"] = self._capture(out)
            outs["grnorm"] = outs["grnorm1"]
        # logger.info(f"Block id {self.block_id} with output in: {out.memory_config()} layout: {out.layout} shape: {out.shape}")

        # logger.info(f"Block id {self.block_id}: relu1 start")
        out = self.relu(out)
        if collect_intermediates:
            outs["relu1"] = self._capture(out)
        # logger.info(f"Block id {self.block_id} with output in: {out.memory_config()} layout: {out.layout} shape: {out.shape}")
        # if out.layout != ttnn.ROW_MAJOR_LAYOUT:
        # out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)

        # logger.info(f"Block id {self.block_id}: conv2 start")
        out, _, _ = self.conv2(device, out)
        # logger.info(f"Block id {self.block_id} with output in: {out.memory_config()} layout: {out.layout} shape: {out.shape}")
        # out = ttnn.move(out)
        if collect_intermediates:
            outs["conv2"] = self._capture(out)

        # logger.info(f"Block id {self.block_id}: bn2 start")
        out = self.bn2(device, out, num_splits=num_splits)
        if collect_intermediates:
            outs["grnorm2"] = self._capture(out)
        # logger.info(f"Block id {self.block_id} with output in: {out.memory_config()} layout: {out.layout} shape: {out.shape}")
        if self.downsample:
            # logger.info(f"Block id {self.block_id}: downsample start")
            identity, _, _ = self.downsample_conv(device, identity)
            identity = self.downsample_bn(device, identity, num_splits=num_splits)
            if collect_intermediates:
                outs["downsample"] = self._capture(identity)
            # logger.info(f"Block id {self.block_id} with output in: {identity.memory_config()} layout: {identity.layout} shape: {identity.shape}")
        # logger.info(f"Block id {self.block_id}: matching identity to out start (store identity in L1)")
        if identity.memory_config() != ttnn.L1_MEMORY_CONFIG:
            identity = ttnn.to_memory_config(identity, ttnn.L1_MEMORY_CONFIG)
        if out.memory_config() != ttnn.L1_MEMORY_CONFIG:
            out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)

        if identity.layout != out.layout:
            identity = ttnn.to_layout(identity, out.layout)

        # logger.info(f"Block id {self.block_id}: add start")
        if tuple(out.shape) != tuple(identity.shape):
            identity = ttnn.reshape(identity, out.shape)
        out = ttnn.add(out, identity)
        # logger.info(f"[ADD] Block id {self.block_id} with output in: {out.memory_config()} layout: {out.layout} shape: {out.shape}")
        if out.memory_config() != ttnn.L1_MEMORY_CONFIG:
            out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)
        if collect_intermediates:
            outs["add"] = self._capture(out)

        # logger.info(f"Block id {self.block_id}: relu2 start")
        out = self.relu(out)
        if collect_intermediates:
            outs["relu2"] = self._capture(out)
            outs["out"] = outs["relu2"]
        # logger.info(f"Block id {self.block_id} with output in: {out.memory_config()} layout: {out.layout} shape: {out.shape}")
        return out, outs


class TTResNet:
    def __init__(self, device, parameters, layer_args, *, dtype=ttnn.bfloat8_b):
        self.conv1 = Conv(parameters.conv1, layer_args.conv1, weight_dtype=dtype)
        self.bn1 = GroupNormL1(parameters.bn1, layer_args.bn1, dtype=dtype)
        self.relu = Relu()

        self.maxpool = TtMaxPool2d(
            configuration=MaxPool2dConfiguration(
                input_height=layer_args.maxpool.input_height,
                input_width=layer_args.maxpool.input_width,
                channels=layer_args.maxpool.input_channels,
                batch_size=layer_args.maxpool.batch_size,
                kernel_size=(layer_args.maxpool.kernel_size, layer_args.maxpool.kernel_size),
                stride=(layer_args.maxpool.stride, layer_args.maxpool.stride),
                padding=(layer_args.maxpool.padding, layer_args.maxpool.padding),
                dilation=(layer_args.maxpool.dilation, layer_args.maxpool.dilation),
                deallocate_input=False,
                dtype=dtype,
                output_layout=ttnn.TILE_LAYOUT,
            ),
            device=device,
        )

        self.layer1 = self._make_layer(parameters.layer1, layer_args.layer1, dtype=ttnn.float32)
        self.layer2 = self._make_layer(parameters.layer2, layer_args.layer2, dtype=ttnn.float32)
        self.layer3 = self._make_layer(parameters.layer3, layer_args.layer3, dtype=ttnn.float32)
        self.layer4 = self._make_layer(parameters.layer4, layer_args.layer4, dtype=ttnn.float32)

    @staticmethod
    def _make_layer(parameters, layer_args, *, dtype):
        return [
            TTBasicBlock(block_parameters, block_args, dtype=dtype)
            for block_parameters, block_args in zip(parameters, layer_args)
        ]

    @staticmethod
    def _run_layer(layer, device, x, *, num_splits=1, collect_intermediates=False):
        layer_outs = {}
        for block_index, block in enumerate(layer, start=1):
            x, outs = block.forward(
                device,
                x,
                num_splits=num_splits,
                collect_intermediates=collect_intermediates,
            )
            if collect_intermediates:
                layer_outs[f"block{block_index}"] = outs
            # logger.info(f"Block id {block.block_id} with output in: {x.memory_config()} layout: {x.layout} shape: {x.shape}")
        return x, layer_outs

    @staticmethod
    def _to_dram(x):
        if x.memory_config() == ttnn.DRAM_MEMORY_CONFIG:
            return x
        return ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    @staticmethod
    def _to_l1(x):
        if x.memory_config() == ttnn.L1_MEMORY_CONFIG:
            return x
        return ttnn.to_memory_config(x, memory_config=ttnn.L1_MEMORY_CONFIG)

    @staticmethod
    def _capture(x):
        return ttnn.to_torch(x).float()

    def forward_feature_pyramid(self, device, x, *, num_splits=1, collect_intermediates=False):
        outs = {}
        # logger.info(f"Input with output in: {x.memory_config()} layout: {x.layout} shape: {x.shape}")
        x = self._to_dram(x)
        # logger.info(f"Input after moving to DRAM with output in: {x.memory_config()} layout: {x.layout} shape: {x.shape}")
        x, _, _ = self.conv1(device, x)
        # logger.info(f"After conv1 with output in: {x.memory_config()} layout: {x.layout} shape: {x.shape}")
        x = ttnn.move(x)
        x = self._to_dram(x)
        # logger.info(f"After moving to DRAM with output in: {x.memory_config()} layout: {x.layout} shape: {x.shape}")
        if collect_intermediates:
            outs["stem_conv"] = self._capture(x)
        x = self.bn1(device, x, num_splits=num_splits)
        # logger.info(f"After bn1 with output in: {x.memory_config()} layout: {x.layout} shape: {x.shape}")
        # x = self._to_dram(x)
        # logger.info(f"After moving to DRAM with output in: {x.memory_config()} layout: {x.layout} shape: {x.shape}")
        if collect_intermediates:
            outs["stem_bn"] = self._capture(x)

        x = self.relu(x)
        x = self._to_l1(x)
        # logger.info(f"After relu with output in: {x.memory_config()} layout: {x.layout} shape: {x.shape}")
        if collect_intermediates:
            outs["stem_relu"] = self._capture(x)

        # x = ttnn.typecast(x, ttnn.bfloat16)
        x = self._to_dram(x)
        x = self.maxpool(x)
        # logger.info(f"After maxpool with output in: {x.memory_config()} layout: {x.layout} shape: {x.shape}")

        x = self._to_l1(x)
        if collect_intermediates:
            outs["stem_pool"] = self._capture(x)

        x, layer1_outs = self._run_layer(
            self.layer1,
            device,
            x,
            num_splits=num_splits,
            collect_intermediates=collect_intermediates,
        )
        if collect_intermediates:
            outs["layer1"] = self._capture(x)
            outs["layer1_blocks"] = layer1_outs
        feats8, layer2_outs = self._run_layer(
            self.layer2,
            device,
            x,
            num_splits=num_splits,
            collect_intermediates=collect_intermediates,
        )
        if collect_intermediates:
            outs["layer2"] = self._capture(feats8)
            outs["layer2_blocks"] = layer2_outs
        feats16, layer3_outs = self._run_layer(
            self.layer3,
            device,
            feats8,
            shard="BS",
            num_splits=num_splits,
            collect_intermediates=collect_intermediates,
        )
        if collect_intermediates:
            outs["layer3"] = self._capture(feats16)
            outs["layer3_blocks"] = layer3_outs
        feats32, layer4_outs = self._run_layer(
            self.layer4,
            device,
            feats16,
            num_splits=num_splits,
            collect_intermediates=collect_intermediates,
        )
        if collect_intermediates:
            outs["layer4"] = self._capture(feats32)
            outs["layer4_blocks"] = layer4_outs

        return self._to_dram(feats8), self._to_dram(feats16), self._to_dram(feats32), outs

    def forward(self, device, x, *, num_splits=1, collect_intermediates=False):
        return self.forward_feature_pyramid(
            device,
            x,
            num_splits=num_splits,
            collect_intermediates=collect_intermediates,
        )


def tt_resnet(device, torch_model, input_height, input_width, layers, batch_size=1, dtype=ttnn.bfloat8_b):
    from types import SimpleNamespace as NS

    def out_hw(h, w, k, s, p):
        return (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1

    def i(x):
        return x[0] if isinstance(x, tuple) else x

    def conv_args(h, w, conv):
        return NS(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            batch_size=batch_size,
            input_height=h,
            input_width=w,
            stride=conv.stride,
            padding=conv.padding,
        )

    def gn_args(h, w, gn):
        return NS(
            num_groups=gn.num_groups,
            num_channels=gn.num_channels,
            eps=gn.eps,
            input_height=h,
            input_width=w,
        )

    def make_layer_args(layer, expected_blocks, h, w):
        assert len(layer) == expected_blocks, f"Expected {expected_blocks} blocks, got {len(layer)}"

        args = []
        for block in layer:
            c1, b1 = block.conv1, block.bn1
            c2, b2 = block.conv2, block.bn2

            h1, w1 = out_hw(h, w, i(c1.kernel_size), i(c1.stride), i(c1.padding))
            h2, w2 = out_hw(h1, w1, i(c2.kernel_size), i(c2.stride), i(c2.padding))

            a = NS(
                conv1=conv_args(h, w, c1),
                bn1=gn_args(h1, w1, b1),
                conv2=conv_args(h1, w1, c2),
                bn2=gn_args(h2, w2, b2),
            )

            if block.downsample is not None:
                dc, db = block.downsample[0], block.downsample[1]
                a.downsample = [
                    conv_args(h, w, dc),
                    gn_args(h2, w2, db),
                ]

            args.append(a)
            h, w = h2, w2

        return args, h, w

    stem_h, stem_w = out_hw(
        input_height,
        input_width,
        i(torch_model.conv1.kernel_size),
        i(torch_model.conv1.stride),
        i(torch_model.conv1.padding),
    )

    h, w = out_hw(stem_h, stem_w, 3, 2, 1)

    l1, h, w = make_layer_args(torch_model.layer1, layers[0], h, w)
    l2, h, w = make_layer_args(torch_model.layer2, layers[1], h, w)
    l3, h, w = make_layer_args(torch_model.layer3, layers[2], h, w)
    l4, h, w = make_layer_args(torch_model.layer4, layers[3], h, w)

    parameters = NS(
        conv1=torch_model.conv1,
        bn1=torch_model.bn1,
        layer1=list(torch_model.layer1),
        layer2=list(torch_model.layer2),
        layer3=list(torch_model.layer3),
        layer4=list(torch_model.layer4),
    )

    layer_args = NS(
        conv1=conv_args(input_height, input_width, torch_model.conv1),
        bn1=gn_args(stem_h, stem_w, torch_model.bn1),
        maxpool=NS(
            input_height=stem_h,
            input_width=stem_w,
            input_channels=torch_model.conv1.out_channels,
            batch_size=batch_size,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
        ),
        layer1=l1,
        layer2=l2,
        layer3=l3,
        layer4=l4,
    )

    return TTResNet(
        device=device,
        parameters=parameters,
        layer_args=layer_args,
        dtype=dtype,
    )


def tt_resnet18(device, torch_model, input_height, input_width, batch_size=1, dtype=ttnn.bfloat8_b):
    return tt_resnet(
        device=device,
        torch_model=torch_model,
        input_height=input_height,
        input_width=input_width,
        layers=[2, 2, 2, 2],
        batch_size=batch_size,
        dtype=dtype,
    )
