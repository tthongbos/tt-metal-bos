import sys
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.nn as nn

import ttnn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_dev.oft_model.tt.resnet import TTResNet

CFG = NS(
    device_id=0,
    l1_small_size=32768,
    batch=1,
    h=64,
    w=128,
    cin=3,
    stem_c=64,
    stem_k=7,
    stem_s=2,
    stem_p=3,
    stem_groups=16,
    pool_k=3,
    pool_s=2,
    pool_p=1,
    pool_d=1,
    block_groups=16,
    eps=1e-5,
    layers=[
        (64, 64, 1),
        (64, 128, 2),
        (128, 256, 2),
        (256, 512, 2),
    ],
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    num_splits=1,
)


def out_hw(h, w, k, s, p):
    return (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1


def conv_args(h, w, cin, cout, s, p):
    return NS(
        in_channels=cin,
        out_channels=cout,
        batch_size=CFG.batch,
        input_height=h,
        input_width=w,
        stride=(s, s),
        padding=(p, p),
    )


def gn_args(h, w, c, groups):
    return NS(
        num_groups=groups,
        num_channels=c,
        eps=CFG.eps,
        input_height=h,
        input_width=w,
    )


def make_block(h, w, cin, cout, stride):
    groups = CFG.block_groups
    h1, w1 = out_hw(h, w, 3, stride, 1)

    params = NS(
        conv1=nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        bn1=nn.GroupNorm(groups, cout, eps=CFG.eps),
        conv2=nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
        bn2=nn.GroupNorm(groups, cout, eps=CFG.eps),
    )

    args = NS(
        conv1=conv_args(h, w, cin, cout, stride, 1),
        bn1=gn_args(h1, w1, cout, groups),
        conv2=conv_args(h1, w1, cout, cout, 1, 1),
        bn2=gn_args(h1, w1, cout, groups),
    )

    if stride != 1 or cin != cout:
        params.downsample = [
            nn.Conv2d(cin, cout, 1, stride, 0, bias=False),
            nn.GroupNorm(groups, cout, eps=CFG.eps),
        ]
        args.downsample = [
            conv_args(h, w, cin, cout, stride, 0),
            gn_args(h1, w1, cout, groups),
        ]

    return params, args, h1, w1


def make_inputs():
    torch.manual_seed(0)

    stem_h, stem_w = out_hw(CFG.h, CFG.w, CFG.stem_k, CFG.stem_s, CFG.stem_p)
    h, w = out_hw(stem_h, stem_w, CFG.pool_k, CFG.pool_s, CFG.pool_p)

    params_layers = []
    args_layers = []

    for cin, cout, stride in CFG.layers:
        p, a, h, w = make_block(h, w, cin, cout, stride)
        params_layers.append([p])
        args_layers.append([a])

    parameters = NS(
        conv1=nn.Conv2d(CFG.cin, CFG.stem_c, CFG.stem_k, CFG.stem_s, CFG.stem_p, bias=False),
        bn1=nn.GroupNorm(CFG.stem_groups, CFG.stem_c, eps=CFG.eps),
        layer1=params_layers[0],
        layer2=params_layers[1],
        layer3=params_layers[2],
        layer4=params_layers[3],
    )

    layer_args = NS(
        conv1=conv_args(CFG.h, CFG.w, CFG.cin, CFG.stem_c, CFG.stem_s, CFG.stem_p),
        bn1=gn_args(stem_h, stem_w, CFG.stem_c, CFG.stem_groups),
        maxpool=NS(
            input_height=stem_h,
            input_width=stem_w,
            input_channels=CFG.stem_c,
            batch_size=CFG.batch,
            kernel_size=CFG.pool_k,
            stride=CFG.pool_s,
            padding=CFG.pool_p,
            dilation=CFG.pool_d,
        ),
        layer1=args_layers[0],
        layer2=args_layers[1],
        layer3=args_layers[2],
        layer4=args_layers[3],
    )

    return parameters, layer_args


def main():
    device = ttnn.open_device(device_id=CFG.device_id, l1_small_size=CFG.l1_small_size)
    print("device arch:", device.arch())

    try:
        parameters, layer_args = make_inputs()

        model = TTResNet(
            device=device,
            parameters=parameters,
            layer_args=layer_args,
            dtype=CFG.dtype,
        )

        torch_input = torch.randn(CFG.batch, CFG.h, CFG.w, CFG.cin)

        tt_input = ttnn.from_torch(
            torch_input,
            dtype=CFG.dtype,
            layout=CFG.layout,
            device=device,
        )

        feats8, feats16, feats32, _ = model.forward(
            device,
            tt_input,
            num_splits=CFG.num_splits,
        )

        print("feats8 :", feats8.shape)
        print("feats16:", feats16.shape)
        print("feats32:", feats32.shape)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
