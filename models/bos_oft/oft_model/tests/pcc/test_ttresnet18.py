import sys
from pathlib import Path

import torch

import ttnn

ttnn.graph.disable_detailed_buffer_tracing()
import tracy

PROJECT_ROOT = Path("/workspace/turtorial-newbie").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_dev.oft_model.reference.architecture.resnet import resnet18
from model_dev.oft_model.tt.resnet_test import tt_resnet18


def pcc(a, b, eps=1e-12):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    return ((a * b).sum() / (a.square().sum().sqrt() * b.square().sum().sqrt() + eps)).item()


def tt_to_nchw(tt_x, ref_x):
    x = tt_x.float() if isinstance(tt_x, torch.Tensor) else ttnn.to_torch(tt_x).float()
    b, c, h, w = ref_x.shape

    if tuple(x.shape) == tuple(ref_x.shape):
        return x

    # TTNN thường là [B, 1, H*W, C]
    if x.ndim == 4 and x.shape[0] == b and x.shape[-2] >= h * w and x.shape[-1] >= c:
        x = x[:, :, : h * w, :c]
        return x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    # Or [B, H, W, C].
    if x.ndim == 4 and x.shape[0] == b and x.shape[1] >= h and x.shape[2] >= w and x.shape[3] >= c:
        x = x[:, :h, :w, :c]
        return x.permute(0, 3, 1, 2).contiguous()

    raise RuntimeError(f"Cannot convert TT shape {tuple(x.shape)} to Torch shape {tuple(ref_x.shape)}")


def print_pcc(name, tt_tensor, ref_tensor):
    tt_x = tt_to_nchw(tt_tensor, ref_tensor)
    print(f"{name:<28} pcc: {pcc(tt_x, ref_tensor):.8f}  tt: {tuple(tt_x.shape)}  torch: {tuple(ref_tensor.shape)}")


def print_block_pcc(outs, outs_ref):
    block_ops = [
        "conv1",
        "grnorm1",
        "relu1",
        "conv2",
        "grnorm2",
        "downsample",
        "add",
        "relu2",
        "out",
    ]

    print("\nper-layer outputs")
    for name in ["stem_conv", "stem_bn", "stem_relu", "stem_pool", "layer1", "layer2", "layer3", "layer4"]:
        print_pcc(name, outs[name], outs_ref[name])

    print("\nper-basic-block outputs")
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        tt_blocks = outs.get(f"{layer_name}_blocks", {})
        ref_blocks = outs_ref.get(f"{layer_name}_blocks", {})

        for block_name, ref_block in ref_blocks.items():
            tt_block = tt_blocks.get(block_name)
            if tt_block is None:
                print(f"{layer_name}.{block_name:<18} missing in TTNN outputs")
                continue

            for op_name in block_ops:
                if op_name not in ref_block or op_name not in tt_block:
                    continue
                print_pcc(f"{layer_name}.{block_name}.{op_name}", tt_block[op_name], ref_block[op_name])


def main():
    batch = 1
    h = 384
    w = 1280
    c = 3
    collect_intermediates = False

    device = ttnn.open_device(device_id=0, l1_small_size=32768)

    try:
        torch_model = resnet18(pretrained=False, dtype=torch.bfloat16).eval()

        tt_model = tt_resnet18(
            device,
            torch_model,
            input_height=h,
            input_width=w,
            batch_size=batch,
            dtype=ttnn.bfloat8_b,
        )

        torch.manual_seed(0)

        torch_input_nhwc = torch.randn(batch, h, w, c, dtype=torch.bfloat16)
        torch_input_nchw = torch_input_nhwc.permute(0, 3, 1, 2).contiguous()

        with torch.no_grad():
            ref8, ref16, ref32, outs_ref = torch_model.forward_feature_pyramid(torch_input_nchw)

        tt_input = ttnn.from_torch(
            torch_input_nhwc,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        tracy.signpost("warmup")
        tt8, tt16, tt32, outs = tt_model.forward(
            device,
            tt_input,
            num_splits=1,
            collect_intermediates=collect_intermediates,
        )
        ttnn.synchronize_device(device)
        del tt8, tt16, tt32, outs
        tracy.signpost("perf")

        tt8, tt16, tt32, outs = tt_model.forward(
            device,
            tt_input,
            num_splits=1,
            collect_intermediates=collect_intermediates,
        )
        ttnn.synchronize_device(device)
        tt8 = tt_to_nchw(tt8, ref8)
        tt16 = tt_to_nchw(tt16, ref16)
        tt32 = tt_to_nchw(tt32, ref32)

        print("feature pyramid")
        print(f"{'feats8':<28} pcc: {pcc(tt8, ref8):.8f}  tt: {tuple(tt8.shape)}  torch: {tuple(ref8.shape)}")
        print(f"{'feats16':<28} pcc: {pcc(tt16, ref16):.8f}  tt: {tuple(tt16.shape)}  torch: {tuple(ref16.shape)}")
        print(f"{'feats32':<28} pcc: {pcc(tt32, ref32):.8f}  tt: {tuple(tt32.shape)}  torch: {tuple(ref32.shape)}")

        if collect_intermediates:
            print_block_pcc(outs, outs_ref)
        else:
            print("\nper-layer outputs skipped (collect_intermediates=False)")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
