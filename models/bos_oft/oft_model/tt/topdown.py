from model_dev.oft_model.tt.resnet import TTBasicBlock

import ttnn


class TTTopDown:
    def __init__(self, parameters, layer_args, *, dtype=ttnn.bfloat16):
        self.blocks = [
            TTBasicBlock(block_parameters, block_args, dtype=dtype)
            for block_parameters, block_args in zip(parameters, layer_args)
        ]

    @staticmethod
    def _capture(x):
        return ttnn.to_torch(x).float()

    def forward(self, device, x, *, num_splits=1):
        outs = {}
        for block_index, block in enumerate(self.blocks, start=1):
            x, block_outs = block.forward(device, x, num_splits=num_splits)
            outs[f"block{block_index}"] = block_outs
        outs["out"] = self._capture(x)
        return x, outs

    def __call__(self, device, x, *, num_splits=1):
        return self.forward(device, x, num_splits=num_splits)


def tt_topdown(device, torch_model, input_height, input_width, batch_size=1, dtype=ttnn.bfloat16):
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

    def make_layer_args(layer, h, w):
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

    layer_args, _, _ = make_layer_args(torch_model, input_height, input_width)

    return TTTopDown(
        parameters=list(torch_model),
        layer_args=layer_args,
        dtype=dtype,
    )
