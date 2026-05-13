import sys
import time
from pathlib import Path

import torch

import ttnn

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_dev.oft_model.reference.architecture.topdown import TopDown
from model_dev.oft_model.tt.topdown import tt_topdown


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

    if x.ndim == 4 and x.shape[0] == b and x.shape[-2] >= h * w and x.shape[-1] >= c:
        x = x[:, :, : h * w, :c]
        return x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    if x.ndim == 4 and x.shape[0] == b and x.shape[1] >= h and x.shape[2] >= w and x.shape[3] >= c:
        x = x[:, :h, :w, :c]
        return x.permute(0, 3, 1, 2).contiguous()

    raise RuntimeError(f"Cannot convert TT shape {tuple(x.shape)} to Torch shape {tuple(ref_x.shape)}")


def print_pcc(name, tt_tensor, ref_tensor):
    tt_x = tt_to_nchw(tt_tensor, ref_tensor)
    print(f"{name:<24} pcc: {pcc(tt_x, ref_tensor):.8f}  tt: {tuple(tt_x.shape)}  torch: {tuple(ref_tensor.shape)}")


def run_torch_topdown_with_outs(model, x):
    outs = {}
    for block_index, block in enumerate(model, start=1):
        x, block_outs = block(x, return_intermediates=True)
        outs[f"block{block_index}"] = block_outs
    outs["out"] = x
    return x, outs


def print_block_pcc(outs, outs_ref):
    block_ops = [
        "conv1",
        "grnorm1",
        "relu1",
        "conv2",
        "grnorm2",
        "add",
        "relu2",
        "out",
    ]

    print("\nper-basic-block outputs")
    for block_name, ref_block in outs_ref.items():
        if block_name == "out":
            continue

        tt_block = outs.get(block_name)
        if tt_block is None:
            print(f"{block_name:<24} missing in TTNN outputs")
            continue

        for op_name in block_ops:
            if op_name not in ref_block or op_name not in tt_block:
                continue
            print_pcc(f"{block_name}.{op_name}", tt_block[op_name], ref_block[op_name])


def main():
    batch = 1
    channels = 512
    h = 12
    w = 40
    layers = 8
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    print("device arch:", device.arch())

    try:
        torch.manual_seed(0)
        torch_model = TopDown(channels=channels, layers=layers, dtype=torch.bfloat16).eval()

        tt_model = tt_topdown(
            device,
            torch_model,
            input_height=h,
            input_width=w,
            batch_size=batch,
            dtype=ttnn.bfloat16,
        )

        torch.manual_seed(123)
        torch_input_nchw = torch.randn(batch, channels, h, w, dtype=torch.bfloat16)
        torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).contiguous()

        with torch.no_grad():
            ref_out, outs_ref = run_torch_topdown_with_outs(torch_model, torch_input_nchw)

        tt_input = ttnn.from_torch(
            torch_input_nhwc,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        ttnn.synchronize_device(device)
        start = time.perf_counter()
        tt_out, outs = tt_model.forward(device, tt_input, num_splits=1)
        ttnn.synchronize_device(device)
        elapsed_s = time.perf_counter() - start

        print("topdown output")
        print_pcc("out", tt_out, ref_out)
        print_block_pcc(outs, outs_ref)

        print(f"\nDebug latency: {elapsed_s:.4f} seconds")
        print(f"Debug FPS: {batch / elapsed_s:.4f}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
