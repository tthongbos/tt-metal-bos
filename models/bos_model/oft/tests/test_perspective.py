import os
import sys

import torch
import torch.nn.functional as F

import ttnn

# Add the root directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from bos_model.oft.reference.oft.model.utils import make_grid
from bos_model.oft.reference.oft.model.utils import perspective as torch_perspective
from bos_model.oft.tt.ttnn_oft import perspective as ttnn_perspective


# TODO: Cleaning the function up later
def test_perspective():
    device = ttnn.open_device(device_id=0)

    # Set up shapes
    B = 1
    Y = 8
    D = 160
    W = 160

    calib = torch.rand(B, 3, 4)
    # Randomly initialize test inputs
    y_corners = torch.arange(0, 4, 0.5) - 4.0 / 2.0
    y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])
    grid = make_grid((79.0, 80.0), (-40.0, -1.74, 0.0), 0.5).unsqueeze(0)

    corners = grid.unsqueeze(1) + y_corners.view(-1, 1, 1, 3)

    matrix_torch = calib.view(-1, 1, 1, 1, 3, 4)
    vector_torch = corners

    # Convert to TTNN format
    matrix_ttnn = ttnn.from_torch(matrix_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    matrix_ttnn = ttnn.to_device(matrix_ttnn, device=device)

    vector_ttnn = ttnn.from_torch(vector_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    vector_ttnn = ttnn.to_device(vector_ttnn, device=device)

    print("Running Torch perspective...")
    out_torch = torch_perspective(matrix_torch, vector_torch)

    print("Running TTNN perspective...")
    out_ttnn = ttnn_perspective(matrix_ttnn, vector_ttnn)

    # Debug parts of perspective
    print("Testing multiplication...")

    A = matrix_torch[:, :, :, :, :, :-1]
    b = matrix_torch[:, :, :, :, :, -1:]
    v_col = vector_torch.unsqueeze(-1)

    homo = torch.matmul(A, v_col) + b
    homo = homo.squeeze(-1)

    A_ttnn = ttnn.from_torch(A, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    A_ttnn = ttnn.to_device(A_ttnn, device)

    b_ttnn = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    b_ttnn = ttnn.to_device(b_ttnn, device)

    v_ttnn = ttnn.from_torch(v_col.transpose(-2, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    v_ttnn = ttnn.to_device(v_ttnn, device)

    res = ttnn.add(ttnn.matmul(v_ttnn, ttnn.transpose(A_ttnn, -2, -1)), ttnn.transpose(b_ttnn, -2, -1))

    res_torch_eval = ttnn.to_torch(res).squeeze(-2)
    print("Homo torch max min", homo.max(), homo.min())
    print("Homo ttnn max min", res_torch_eval.max(), res_torch_eval.min())

    print("homo z min val:", homo[:, :, :, :, 2:3].min().item(), "max:", homo[:, :, :, :, 2:3].max().item())
    print(
        "res_torch_eval z min val:",
        res_torch_eval[:, :, :, :, 2:3].min().item(),
        "max:",
        res_torch_eval[:, :, :, :, 2:3].max().item(),
    )

    z_ttnn = res_torch_eval[:, :, :, :, 2:3]
    zeros_ttnn = (z_ttnn == 0).sum().item()
    print("Zeros in z_ttnn:", zeros_ttnn)

    print("Testing divide with epsilon...")
    eps = 1e-4

    homogenous = res  # Shape: [1, 8, 158, 160, 1, 3] from previous debug log... ah
    z_ttnn = homogenous[:, :, :, :, :, 2:3]
    xy_ttnn = homogenous[:, :, :, :, :, :2]

    z_eps_ttnn = ttnn.add(z_ttnn, eps)
    div_ttnn = ttnn.mul(xy_ttnn, ttnn.reciprocal(z_eps_ttnn))

    div_torch = ttnn.to_torch(div_ttnn)
    print("Div ttnn min max:", div_torch.min().item(), div_torch.max().item())

    print(
        "homo[:, :, :, :, :2] / homo[:, :, :, :, 2:3] isnan torch:",
        (homo[:, :, :, :, :2] / homo[:, :, :, :, 2:3]).isnan().sum(),
    )
    print(
        "res_torch_eval[:, :, :, :, :2] / res_torch_eval[:, :, :, :, 2:3] isnan ttnn:",
        (res_torch_eval[:, :, :, :, :2] / res_torch_eval[:, :, :, :, 2:3]).isnan().sum(),
    )

    # Convert TTNN output back to Torch to compare
    out_ttnn_to_torch = ttnn.to_torch(out_ttnn)

    # # Compute using operations like utils.perspective directly
    out_torch_homo = torch.matmul(matrix_torch[..., :-1], vector_torch.unsqueeze(-1)) + matrix_torch[..., [-1]]
    out_torch_homo = out_torch_homo.squeeze(-1)
    out_torch_exact = out_torch_homo[..., :-1] / out_torch_homo[..., [-1]]

    # Check max differences in the homogenous coordinate step
    diff_homo = torch.abs(homo - res_torch_eval)
    print("Max homo diff:", diff_homo.max().item())

    # diff_final = torch.abs(out_torch_exact - out_ttnn_to_torch)
    # print("Max final diff:", diff_final.max().item())


    print("out_torch min max:", out_torch.min().item(), out_torch.max().item())
    print("out_torch isnan:", out_torch.isnan().sum().item())
    print("out_ttnn_to_torch min max:", out_ttnn_to_torch.min().item(), out_ttnn_to_torch.max().item())
    print("out_ttnn_to_torch isnan:", out_ttnn_to_torch.isnan().sum().item())

    # Compare
    pcc = torch.corrcoef(torch.stack([out_torch_exact.flatten(), out_ttnn_to_torch.flatten()]))[0, 1].item()
    rmse = torch.sqrt(((out_torch_exact - out_ttnn_to_torch) ** 2).mean()).item()

    print(f"\\n=== Accuracy Test ===")
    print(f"TTNN vs PyTorch output shape for perspective:")
    print(f"Torch shape: {out_torch.shape}")
    print(f"TTNN shape: {out_ttnn_to_torch.shape}")
    print(f"PCC:  {pcc:.6f}")
    print(f"RMSE: {rmse:.6f}")

    ttnn.close_device(device)


if __name__ == "__main__":
    test_perspective()
