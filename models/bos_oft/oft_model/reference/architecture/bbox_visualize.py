from pathlib import Path

import torch


def bbox_corners(obj):
    dtype = obj.position.dtype
    device = obj.position.device
    offsets = torch.tensor(
        [
            [-0.5, 0.0, -0.5],
            [0.5, 0.0, -0.5],
            [-0.5, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [-0.5, -1.0, -0.5],
            [0.5, -1.0, -0.5],
            [-0.5, -1.0, 0.5],
            [0.5, -1.0, 0.5],
        ],
        dtype=dtype,
        device=device,
    )
    corners = offsets * obj.dimensions
    sin_angle = torch.sin(obj.angle)
    cos_angle = torch.cos(obj.angle)
    xvals = cos_angle * corners[..., 0] + sin_angle * corners[..., 2]
    yvals = corners[..., 1]
    zvals = -sin_angle * corners[..., 0] + cos_angle * corners[..., 2]
    return torch.stack([xvals, yvals, zvals], dim=-1) + obj.position


def project_points(calib, points):
    ones = torch.ones_like(points[..., :1])
    points_h = torch.cat([points, ones], dim=-1)
    projected = torch.matmul(calib, points_h.unsqueeze(-1)).squeeze(-1)
    return projected[..., :2] / projected[..., 2:3]


def save_bbox_visualization(image, image_hw, calib, objects, output_path):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Polygon
    except ImportError as exc:
        raise ImportError("Matplotlib is required to save visualizations: python3 -m pip install matplotlib") from exc

    height, width = image_hw
    image_for_plot = image[0, :, :height, :width].detach().cpu().permute(1, 2, 0).numpy()
    calib = calib.squeeze(0).detach().cpu()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(image_for_plot)
    ax.axis(False)

    edges = [
        (0, 1),
        (1, 3),
        (3, 2),
        (2, 0),
        (4, 5),
        (5, 7),
        (7, 6),
        (6, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    faces = ([1, 3, 7, 5], [0, 2, 6, 4])

    for obj in objects:
        corners = bbox_corners(obj).detach().cpu()
        img_corners = project_points(calib, corners).numpy()
        for face in faces:
            ax.add_patch(Polygon(img_corners[list(face)], edgecolor="lime", fill=False, linewidth=1.5))
        for start, end in edges:
            ax.add_line(Line2D(*img_corners[[start, end]].T, color="lime", linewidth=1.2))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved bbox visualization: {output_path}")
