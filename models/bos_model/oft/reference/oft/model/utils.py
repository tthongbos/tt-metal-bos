import torch

import ttnn


def rotate(vector, angle):
    """
    Rotate a vector around the y-axis
    """

    sinA, cosA = torch.sin(angle), torch.cos(angle)
    xvals = cosA * vector[..., 0] + sinA * vector[..., 2]
    yvals = vector[..., 1]
    zvals = -sinA * vector[..., 0] + cosA * vector[..., 2]

    return torch.stack([xvals, yvals, zvals], dim=-1)


def perspective(matrix, vector):
    """
    Applies perspective projection to a vector using projection matrix
    """
    vector = vector.unsqueeze(-1)
    homogeneous = torch.matmul(matrix[..., :-1], vector) + matrix[..., [-1]]
    # return homogeneous
    homogeneous = homogeneous.squeeze(-1)
    # print(homogeneous.shape)
    # return homogeneous
    return homogeneous[..., :-1] / homogeneous[..., [-1]]


def make_grid(grid_size, grid_offset, grid_res):
    """
    Constructs an array representing the corners of an orthographic grid
    """
    depth, width = grid_size
    xoff, yoff, zoff = grid_offset

    xcoords = torch.arange(0.0, width, grid_res) + xoff
    zcoords = torch.arange(0.0, depth, grid_res) + zoff

    zz, xx = torch.meshgrid(zcoords, xcoords)
    return torch.stack([xx, torch.full_like(xx, yoff), zz], dim=-1)


def comparing_torch_ttnn(torch_tensor, ttnn_tensor):
    # Compare a Torch tensor and a TTNN tensor by converting the TTNN tensor to Torch
    print("\n=== Accuracy Test ===")
    torch_ttnn_tensor = ttnn.to_torch(ttnn_tensor)

    pcc = torch.corrcoef(torch.stack([torch_ttnn_tensor.flatten(), torch_tensor.flatten()]))[0, 1].item()
    rmse = torch.sqrt(((torch_ttnn_tensor - torch_tensor) ** 2).mean()).item()
    abs = torch.abs(torch_ttnn_tensor - torch_tensor).mean().item()
    torch_min, torch_max = torch.min(torch_tensor).item(), torch.max(torch_tensor).item()
    print(f"Tensor values range from {torch_min:.6f} to {torch_max:.6f}")
    ttnn_min, ttnn_max = torch.min(torch_ttnn_tensor).item(), torch.max(torch_ttnn_tensor).item()
    print(f"TTNN tensor values range from {ttnn_min:.6f} to {ttnn_max:.6f}")
    print(f"TTNN vs PyTorch:  PCC={pcc:.6f}, RMSE={rmse:.6f}, Abs={abs:.6f}")
