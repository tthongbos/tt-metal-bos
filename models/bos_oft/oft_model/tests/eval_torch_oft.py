from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

_SCRIPT_PATH = Path(__file__).resolve()
_PROJECT_ROOT = _SCRIPT_PATH.parents[3] if len(_SCRIPT_PATH.parents) > 3 else Path.cwd()

try:
    from ..reference.architecture.bbox_visualize import bbox_corners, project_points
    from ..reference.architecture.oftnet import OftNet
    from ..reference.architecture.oftnet_pipeline import (
        DEFAULT_CHECKPOINT_PATH,
        GRID_HEIGHT,
        GRID_RES,
        GRID_SIZE,
        H_PADDED,
        NMS_THRESH,
        W_PADDED,
        Y_OFFSET,
        decode_oftnet_outputs,
    )
    from ..utils.pipeline_utils import load_calib, load_oft_checkpoint, load_padded_image_tensor, make_grid
except ImportError:
    sys.path.append(str(_PROJECT_ROOT))
    from model_dev.oft_model.reference.architecture.bbox_visualize import bbox_corners, project_points
    from model_dev.oft_model.reference.architecture.oftnet import OftNet
    from model_dev.oft_model.reference.architecture.oftnet_pipeline import (
        DEFAULT_CHECKPOINT_PATH,
        GRID_HEIGHT,
        GRID_RES,
        GRID_SIZE,
        H_PADDED,
        NMS_THRESH,
        W_PADDED,
        Y_OFFSET,
        decode_oftnet_outputs,
    )
    from model_dev.oft_model.utils.pipeline_utils import (
        load_calib,
        load_oft_checkpoint,
        load_padded_image_tensor,
        make_grid,
    )


DEFAULT_DATASET_ROOT = _PROJECT_ROOT / "dataset" / "kiti" / "training"

# KITTI official object classes.
CLASS_NAMES = ("Car", "Pedestrian", "Cyclist")
DIFFICULTY_NAMES = ("easy", "moderate", "hard")
METRIC_NAMES = ("image", "ground", "box3d")

# Same static parameters as evaluate_object.cpp.
MIN_HEIGHT = (40.0, 25.0, 25.0)
MAX_OCCLUSION = (0, 1, 2)
MAX_TRUNCATION = (0.15, 0.30, 0.50)
N_SAMPLE_PTS = 41

# In the provided KITTI evaluator, IMAGE / GROUND / BOX3D all use the same class thresholds.
MIN_OVERLAP = {
    "image": {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5},
    "ground": {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5},
    "box3d": {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5},
}

NO_DETECTION = -10_000_000.0
INVALID_POS = -1000.0
EPS = 1e-12


@dataclass
class EvalObject:
    """Common KITTI object representation for both ground truth and detections."""

    class_name: str
    bbox: tuple[float, float, float, float]
    alpha: float = -10.0
    score: float = 1.0
    truncation: float = -1.0
    occlusion: int = -1
    h: float = -1.0
    w: float = -1.0
    l: float = -1.0
    x: float = INVALID_POS
    y: float = INVALID_POS
    z: float = INVALID_POS
    ry: float = -10.0
    # Optional direct geometry from bbox_corners(obj). For detections this avoids
    # guessing whether the decoded object stores h,w,l / x,y,z / ry exactly in
    # KITTI convention. Ground truth objects keep these as None and use KITTI fields.
    bev_polygon: tuple[tuple[float, float], ...] | None = None
    y_top: float | None = None
    y_bottom: float | None = None

    @property
    def height_2d(self) -> float:
        return abs(self.bbox[3] - self.bbox[1])

    @property
    def has_valid_2d(self) -> bool:
        x1, y1, x2, y2 = self.bbox
        return all(math.isfinite(v) for v in self.bbox) and x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1

    @property
    def has_valid_ground(self) -> bool:
        if self.bev_polygon is not None:
            return (
                len(self.bev_polygon) >= 3
                and all(math.isfinite(coord) for point in self.bev_polygon for coord in point)
                and polygon_area(list(self.bev_polygon)) > EPS
            )
        return (
            math.isfinite(self.x)
            and math.isfinite(self.z)
            and self.x != INVALID_POS
            and self.z != INVALID_POS
            and self.w > 0
            and self.l > 0
            and math.isfinite(self.ry)
        )

    @property
    def has_valid_3d(self) -> bool:
        if self.bev_polygon is not None:
            return self.has_valid_ground and self.y_top is not None and self.y_bottom is not None
        return self.has_valid_ground and math.isfinite(self.y) and self.y != INVALID_POS and self.h > 0


@dataclass
class PrData:
    scores: list[float]
    similarity: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0


@dataclass
class ClassDifficultyResult:
    ap: float
    precision_curve: list[float]
    recall_curve: list[float]
    thresholds: list[float]
    aos_ap: float | None = None
    aos_curve: list[float] | None = None
    n_groundtruth: int = 0


def normalize_class_name(name: str) -> str:
    for class_name in CLASS_NAMES:
        if name.lower() == class_name.lower():
            return class_name
    return name


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def compute_alpha(ry: float, x: float, z: float) -> float:
    if not all(math.isfinite(v) for v in (ry, x, z)) or z == 0:
        return -10.0
    return normalize_angle(ry - math.atan2(x, z))


def tensor_to_float_list(value, expected_len: int | None = None) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, (int, float)):
        values = [float(value)]
    else:
        values = [float(v) for v in value]

    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(values)}: {values}")
    return values


def unpack_dimensions(dimensions: Iterable[float], dimension_order: str) -> tuple[float, float, float]:
    values = list(float(v) for v in dimensions)
    if len(values) != 3:
        raise ValueError(f"Expected 3 dimensions, got {len(values)}: {values}")
    mapping = {axis: values[index] for index, axis in enumerate(dimension_order)}
    return mapping["h"], mapping["w"], mapping["l"]


def load_groundtruth(label_path: Path) -> list[EvalObject]:
    objects: list[EvalObject] = []
    with open(label_path) as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 15:
                continue

            class_name = normalize_class_name(fields[0])
            truncation = float(fields[1])
            occlusion = int(float(fields[2]))
            alpha = float(fields[3])
            x1, y1, x2, y2 = map(float, fields[4:8])
            h, w, l = map(float, fields[8:11])
            x, y, z = map(float, fields[11:14])
            ry = float(fields[14])

            objects.append(
                EvalObject(
                    class_name=class_name,
                    bbox=(x1, y1, x2, y2),
                    alpha=alpha,
                    truncation=truncation,
                    occlusion=occlusion,
                    h=h,
                    w=w,
                    l=l,
                    x=x,
                    y=y,
                    z=z,
                    ry=ry,
                )
            )
    return objects


def prediction_to_eval_object(obj, calib, image_hw, dimension_order: str = "hwl") -> EvalObject | None:
    """Convert an OFTNet decoded object to a KITTI-style detection object."""

    corners = bbox_corners(obj).detach().cpu()
    corner_values = [[float(value) for value in point] for point in corners.tolist()]
    bev_polygon = tuple(convex_hull((point[0], point[2]) for point in corner_values))
    y_values = [point[1] for point in corner_values]
    y_top = min(y_values) if y_values else None
    y_bottom = max(y_values) if y_values else None

    projected = project_points(calib.squeeze(0).detach().cpu(), corners)
    xs = projected[:, 0]
    ys = projected[:, 1]

    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max())
    y2 = float(ys.max())
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None

    image_height, image_width = image_hw
    x1 = min(max(x1, 0.0), float(image_width - 1))
    y1 = min(max(y1, 0.0), float(image_height - 1))
    x2 = min(max(x2, 0.0), float(image_width - 1))
    y2 = min(max(y2, 0.0), float(image_height - 1))
    if x2 <= x1 or y2 <= y1:
        return None

    class_name = normalize_class_name(str(getattr(obj, "classname", "Car")))
    score = float(getattr(obj, "score", 1.0))

    position = tensor_to_float_list(getattr(obj, "position"), expected_len=3)
    dimensions = tensor_to_float_list(getattr(obj, "dimensions"), expected_len=3)
    h, w, l = unpack_dimensions(dimensions, dimension_order=dimension_order)
    x, y, z = position
    ry = float(getattr(obj, "angle"))
    alpha = compute_alpha(ry, x, z)

    return EvalObject(
        class_name=class_name,
        bbox=(x1, y1, x2, y2),
        alpha=alpha,
        score=score,
        h=h,
        w=w,
        l=l,
        x=x,
        y=y,
        z=z,
        ry=ry,
        bev_polygon=bev_polygon,
        y_top=y_top,
        y_bottom=y_bottom,
    )


def bbox_overlap(
    a_bbox: tuple[float, float, float, float], b_bbox: tuple[float, float, float, float], criterion: int = -1
) -> float:
    ax1, ay1, ax2, ay2 = a_bbox
    bx1, by1, bx2, by2 = b_bbox
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    w = x2 - x1
    h = y2 - y1
    if w <= 0.0 or h <= 0.0:
        return 0.0

    inter = w * h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    if criterion == -1:
        denom = area_a + area_b - inter
    elif criterion == 0:
        denom = area_a
    elif criterion == 1:
        denom = area_b
    else:
        raise ValueError(f"Unknown overlap criterion: {criterion}")

    return 0.0 if denom <= 0.0 else inter / denom


def image_box_overlap(det: EvalObject, gt: EvalObject, criterion: int = -1) -> float:
    return bbox_overlap(det.bbox, gt.bbox, criterion=criterion)


def polygon_signed_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i, point in enumerate(poly):
        nxt = poly[(i + 1) % len(poly)]
        area += point[0] * nxt[1] - nxt[0] * point[1]
    return 0.5 * area


def polygon_area(poly: list[tuple[float, float]]) -> float:
    return abs(polygon_signed_area(poly))


def convex_hull(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return a counter-clockwise convex hull for a small set of 2D points."""

    unique = sorted(set((float(x), float(y)) for x, y in points))
    if len(unique) <= 1:
        return unique

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= EPS:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= EPS:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def vertical_bounds(obj: EvalObject) -> tuple[float, float]:
    if obj.y_top is not None and obj.y_bottom is not None:
        return min(float(obj.y_top), float(obj.y_bottom)), max(float(obj.y_top), float(obj.y_bottom))
    # KITTI labels store y as the bottom of the box.
    return min(obj.y - obj.h, obj.y), max(obj.y - obj.h, obj.y)


def is_inside_half_plane(point, edge_start, edge_end, clip_is_ccw: bool) -> bool:
    cross = (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - (edge_end[1] - edge_start[1]) * (
        point[0] - edge_start[0]
    )
    return cross >= -EPS if clip_is_ccw else cross <= EPS


def line_intersection(p1, p2, q1, q2):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < EPS:
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)


def polygon_clip(
    subject_polygon: list[tuple[float, float]], clip_polygon: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman polygon clipping for convex polygons."""

    output = subject_polygon[:]
    if len(output) < 3 or len(clip_polygon) < 3:
        return []

    clip_is_ccw = polygon_signed_area(clip_polygon) > 0.0
    for i, edge_start in enumerate(clip_polygon):
        edge_end = clip_polygon[(i + 1) % len(clip_polygon)]
        input_list = output
        output = []
        if not input_list:
            break

        prev = input_list[-1]
        prev_inside = is_inside_half_plane(prev, edge_start, edge_end, clip_is_ccw)
        for curr in input_list:
            curr_inside = is_inside_half_plane(curr, edge_start, edge_end, clip_is_ccw)
            if curr_inside:
                if not prev_inside:
                    output.append(line_intersection(prev, curr, edge_start, edge_end))
                output.append(curr)
            elif prev_inside:
                output.append(line_intersection(prev, curr, edge_start, edge_end))
            prev = curr
            prev_inside = curr_inside
    return output


def bev_corners(obj: EvalObject) -> list[tuple[float, float]]:
    """Return BEV polygon in KITTI x-z plane."""

    if obj.bev_polygon is not None:
        return list(obj.bev_polygon)

    c = math.cos(obj.ry)
    s = math.sin(obj.ry)
    local = [
        (obj.l / 2.0, obj.w / 2.0),
        (obj.l / 2.0, -obj.w / 2.0),
        (-obj.l / 2.0, -obj.w / 2.0),
        (-obj.l / 2.0, obj.w / 2.0),
    ]

    corners = []
    for lx, lz in local:
        # Same matrix as evaluate_object.cpp:
        # [ cos(ry)  sin(ry)] [local_x]
        # [-sin(ry)  cos(ry)] [local_z]
        x = c * lx + s * lz + obj.x
        z = -s * lx + c * lz + obj.z
        corners.append((x, z))
    return corners


def bev_intersection_area(det: EvalObject, gt: EvalObject) -> float:
    if not det.has_valid_ground or not gt.has_valid_ground:
        return 0.0
    det_poly = bev_corners(det)
    gt_poly = bev_corners(gt)
    inter_poly = polygon_clip(det_poly, gt_poly)
    return polygon_area(inter_poly)


def ground_box_overlap(det: EvalObject, gt: EvalObject, criterion: int = -1) -> float:
    if not det.has_valid_ground or not gt.has_valid_ground:
        return 0.0
    inter_area = bev_intersection_area(det, gt)
    det_area = polygon_area(bev_corners(det))
    gt_area = polygon_area(bev_corners(gt))

    if criterion == -1:
        denom = det_area + gt_area - inter_area
    elif criterion == 0:
        denom = det_area
    elif criterion == 1:
        denom = gt_area
    else:
        raise ValueError(f"Unknown overlap criterion: {criterion}")

    return 0.0 if denom <= 0.0 else inter_area / denom


def box3d_overlap(det: EvalObject, gt: EvalObject, criterion: int = -1) -> float:
    if not det.has_valid_3d or not gt.has_valid_3d:
        return 0.0
    inter_area = bev_intersection_area(det, gt)
    det_top, det_bottom = vertical_bounds(det)
    gt_top, gt_bottom = vertical_bounds(gt)
    inter_h = max(0.0, min(det_bottom, gt_bottom) - max(det_top, gt_top))
    inter_vol = inter_area * inter_h

    det_vol = polygon_area(bev_corners(det)) * max(0.0, det_bottom - det_top)
    gt_vol = polygon_area(bev_corners(gt)) * max(0.0, gt_bottom - gt_top)

    if criterion == -1:
        denom = det_vol + gt_vol - inter_vol
    elif criterion == 0:
        denom = det_vol
    elif criterion == 1:
        denom = gt_vol
    else:
        raise ValueError(f"Unknown overlap criterion: {criterion}")

    return 0.0 if denom <= 0.0 else inter_vol / denom


def is_same_class(obj: EvalObject, class_name: str) -> bool:
    return obj.class_name.lower() == class_name.lower()


def is_neighbor_class(obj: EvalObject, class_name: str) -> bool:
    obj_name = obj.class_name.lower()
    class_lower = class_name.lower()
    return (class_lower == "car" and obj_name == "van") or (
        class_lower == "pedestrian" and obj_name == "person_sitting"
    )


def clean_data(
    current_class: str,
    gt: list[EvalObject],
    det: list[EvalObject],
    difficulty_index: int,
) -> tuple[list[int], list[EvalObject], list[int], int]:
    """
    Replicate KITTI cleanData().

    ignored_gt:  0 = valid GT, 1 = ignored GT / neighboring class, -1 = not relevant
    ignored_det: 0 = valid detection, 1 = ignored due to min height, -1 = not relevant
    """

    ignored_gt: list[int] = []
    ignored_det: list[int] = []
    dontcare: list[EvalObject] = []
    n_gt = 0

    for obj in gt:
        if is_same_class(obj, current_class):
            valid_class = 1
        elif is_neighbor_class(obj, current_class):
            valid_class = 0
        else:
            valid_class = -1

        ignore = (
            obj.occlusion > MAX_OCCLUSION[difficulty_index]
            or obj.truncation > MAX_TRUNCATION[difficulty_index]
            or obj.height_2d <= MIN_HEIGHT[difficulty_index]
        )

        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            n_gt += 1
        elif valid_class == 0 or (ignore and valid_class == 1):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)

        if obj.class_name.lower() == "dontcare":
            dontcare.append(obj)

    for obj in det:
        valid_class = 1 if is_same_class(obj, current_class) else -1
        if obj.height_2d < MIN_HEIGHT[difficulty_index]:
            ignored_det.append(1)
        elif valid_class == 1:
            ignored_det.append(0)
        else:
            ignored_det.append(-1)

    return ignored_gt, dontcare, ignored_det, n_gt


def compute_statistics(
    current_class: str,
    gt: list[EvalObject],
    det: list[EvalObject],
    dontcare: list[EvalObject],
    ignored_gt: list[int],
    ignored_det: list[int],
    compute_fp: bool,
    boxoverlap: Callable[[EvalObject, EvalObject, int], float],
    metric: str,
    compute_aos: bool = False,
    thresh: float = 0.0,
) -> PrData:
    stat = PrData(scores=[])
    delta: list[float] = []
    assigned_detection = [False] * len(det)
    ignored_threshold = [False] * len(det)
    min_overlap = MIN_OVERLAP[metric][current_class]

    if compute_fp:
        for index, obj in enumerate(det):
            if obj.score < thresh:
                ignored_threshold[index] = True

    for gt_index, gt_obj in enumerate(gt):
        if ignored_gt[gt_index] == -1:
            continue

        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0.0
        assigned_ignored_det = False

        for det_index, det_obj in enumerate(det):
            if ignored_det[det_index] == -1:
                continue
            if assigned_detection[det_index]:
                continue
            if ignored_threshold[det_index]:
                continue

            overlap = boxoverlap(det_obj, gt_obj, -1)

            if (not compute_fp) and overlap > min_overlap and det_obj.score > valid_detection:
                det_idx = det_index
                valid_detection = det_obj.score
            elif (
                compute_fp
                and overlap > min_overlap
                and (overlap > max_overlap or assigned_ignored_det)
                and ignored_det[det_index] == 0
            ):
                max_overlap = overlap
                det_idx = det_index
                valid_detection = 1.0
                assigned_ignored_det = False
            elif (
                compute_fp and overlap > min_overlap and valid_detection == NO_DETECTION and ignored_det[det_index] == 1
            ):
                det_idx = det_index
                valid_detection = 1.0
                assigned_ignored_det = True

        if valid_detection == NO_DETECTION and ignored_gt[gt_index] == 0:
            stat.fn += 1
        elif valid_detection != NO_DETECTION and (ignored_gt[gt_index] == 1 or ignored_det[det_idx] == 1):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            stat.tp += 1
            stat.scores.append(det[det_idx].score)
            if compute_aos:
                delta.append(gt_obj.alpha - det[det_idx].alpha)
            assigned_detection[det_idx] = True

    if compute_fp:
        for det_index in range(len(det)):
            if not (
                assigned_detection[det_index]
                or ignored_det[det_index] == -1
                or ignored_det[det_index] == 1
                or ignored_threshold[det_index]
            ):
                stat.fp += 1

        nstuff = 0
        for dc_obj in dontcare:
            for det_index, det_obj in enumerate(det):
                if assigned_detection[det_index]:
                    continue
                if ignored_det[det_index] == -1 or ignored_det[det_index] == 1:
                    continue
                if ignored_threshold[det_index]:
                    continue
                overlap = boxoverlap(det_obj, dc_obj, 0)
                if overlap > min_overlap:
                    assigned_detection[det_index] = True
                    nstuff += 1
        stat.fp -= nstuff

        if compute_aos:
            similarities = [0.0] * stat.fp
            similarities.extend((1.0 + math.cos(value)) / 2.0 for value in delta)
            if stat.tp > 0 or stat.fp > 0:
                stat.similarity = sum(similarities)
            else:
                stat.similarity = -1.0

    return stat


def get_thresholds(scores: list[float], n_groundtruth: int) -> list[float]:
    if n_groundtruth <= 0:
        return []

    thresholds: list[float] = []
    scores = sorted(scores, reverse=True)
    current_recall = 0.0

    for index, score in enumerate(scores):
        left_recall = (index + 1.0) / n_groundtruth
        if index < len(scores) - 1:
            right_recall = (index + 2.0) / n_groundtruth
        else:
            right_recall = left_recall

        if (right_recall - current_recall) < (current_recall - left_recall) and index < len(scores) - 1:
            continue

        thresholds.append(score)
        if len(thresholds) >= N_SAMPLE_PTS:
            break
        current_recall += 1.0 / (N_SAMPLE_PTS - 1.0)

    return thresholds


def evaluate_class_difficulty(
    current_class: str,
    groundtruth: list[list[EvalObject]],
    detections: list[list[EvalObject]],
    metric: str,
    difficulty_index: int,
    compute_aos: bool,
) -> ClassDifficultyResult:
    if len(groundtruth) != len(detections):
        raise ValueError("groundtruth and detections must have the same number of frames")

    boxoverlap = {
        "image": image_box_overlap,
        "ground": ground_box_overlap,
        "box3d": box3d_overlap,
    }[metric]

    n_gt_total = 0
    all_scores: list[float] = []
    all_ignored_gt: list[list[int]] = []
    all_ignored_det: list[list[int]] = []
    all_dontcare: list[list[EvalObject]] = []

    for frame_gt, frame_det in zip(groundtruth, detections):
        ignored_gt, dontcare, ignored_det, n_gt = clean_data(current_class, frame_gt, frame_det, difficulty_index)
        n_gt_total += n_gt
        all_ignored_gt.append(ignored_gt)
        all_ignored_det.append(ignored_det)
        all_dontcare.append(dontcare)

        pr_tmp = compute_statistics(
            current_class=current_class,
            gt=frame_gt,
            det=frame_det,
            dontcare=dontcare,
            ignored_gt=ignored_gt,
            ignored_det=ignored_det,
            compute_fp=False,
            boxoverlap=boxoverlap,
            metric=metric,
        )
        all_scores.extend(pr_tmp.scores)

    thresholds = get_thresholds(all_scores, n_gt_total)
    pr = [PrData(scores=[]) for _ in thresholds]

    for frame_index, (frame_gt, frame_det) in enumerate(zip(groundtruth, detections)):
        for threshold_index, threshold in enumerate(thresholds):
            tmp = compute_statistics(
                current_class=current_class,
                gt=frame_gt,
                det=frame_det,
                dontcare=all_dontcare[frame_index],
                ignored_gt=all_ignored_gt[frame_index],
                ignored_det=all_ignored_det[frame_index],
                compute_fp=True,
                boxoverlap=boxoverlap,
                metric=metric,
                compute_aos=compute_aos,
                thresh=threshold,
            )
            pr[threshold_index].tp += tmp.tp
            pr[threshold_index].fp += tmp.fp
            pr[threshold_index].fn += tmp.fn
            if tmp.similarity != -1.0:
                pr[threshold_index].similarity += tmp.similarity

    precision = [0.0] * N_SAMPLE_PTS
    recall = [0.0] * N_SAMPLE_PTS
    aos_curve = [0.0] * N_SAMPLE_PTS if compute_aos else None

    for index, threshold_pr in enumerate(pr):
        tp = threshold_pr.tp
        fp = threshold_pr.fp
        fn = threshold_pr.fn
        precision[index] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[index] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if compute_aos and aos_curve is not None:
            aos_curve[index] = threshold_pr.similarity / (tp + fp) if (tp + fp) > 0 else 0.0

    # Same monotonic envelope as evaluate_object.cpp: precision[i] = max(precision[i:]).
    for index in range(len(thresholds)):
        precision[index] = max(precision[index:])
        if compute_aos and aos_curve is not None:
            aos_curve[index] = max(aos_curve[index:])

    ap = sum(precision) / N_SAMPLE_PTS
    aos_ap = sum(aos_curve) / N_SAMPLE_PTS if compute_aos and aos_curve is not None else None

    return ClassDifficultyResult(
        ap=ap,
        precision_curve=precision,
        recall_curve=recall,
        thresholds=thresholds,
        aos_ap=aos_ap,
        aos_curve=aos_curve,
        n_groundtruth=n_gt_total,
    )


def should_compute_aos(detections: list[list[EvalObject]]) -> bool:
    for frame_det in detections:
        for det in frame_det:
            if det.alpha == -10.0 or not math.isfinite(det.alpha):
                return False
    return True


def evaluate_kitti_style(
    groundtruth: list[list[EvalObject]],
    detections: list[list[EvalObject]],
    classes: Iterable[str],
    metrics: Iterable[str],
    compute_aos: bool = True,
) -> dict:
    results: dict = {}
    aos_enabled = compute_aos and should_compute_aos(detections)

    for class_name in classes:
        class_name = normalize_class_name(class_name)
        results[class_name] = {}
        for metric in metrics:
            metric_compute_aos = aos_enabled and metric == "image"
            results[class_name][metric] = {}
            for difficulty_index, difficulty_name in enumerate(DIFFICULTY_NAMES):
                difficulty_result = evaluate_class_difficulty(
                    current_class=class_name,
                    groundtruth=groundtruth,
                    detections=detections,
                    metric=metric,
                    difficulty_index=difficulty_index,
                    compute_aos=metric_compute_aos,
                )
                results[class_name][metric][difficulty_name] = asdict(difficulty_result)
    return results


def build_model(checkpoint_path, dtype=torch.float32):
    model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=GRID_RES,
        grid_height=GRID_HEIGHT,
        frontend_pretrained=False,
        dtype=dtype,
    )
    model = load_oft_checkpoint(model, checkpoint_path)
    model.eval()
    return model


def infer_image(model, image_path, calib_path, grid, pad_hw, nms_thresh, dtype=torch.float32):
    image, image_hw = load_padded_image_tensor(image_path, pad_hw=pad_hw, dtype=dtype)
    calib = load_calib(calib_path, dtype=dtype)
    with torch.no_grad():
        scores, pos_offsets, dim_offsets, ang_offsets = model(image, calib, grid)
        objects, _ = decode_oftnet_outputs(
            scores,
            pos_offsets,
            dim_offsets,
            ang_offsets,
            grid,
            dtype=dtype,
            nms_thresh=nms_thresh,
        )
    return objects, calib, image_hw


def serialize_eval_object(obj: EvalObject) -> dict:
    return {
        "class_name": obj.class_name,
        "score": obj.score,
        "bbox_2d": {
            "x1": obj.bbox[0],
            "y1": obj.bbox[1],
            "x2": obj.bbox[2],
            "y2": obj.bbox[3],
        },
        "alpha": obj.alpha,
        "dimensions_hwl": [obj.h, obj.w, obj.l],
        "location_xyz": [obj.x, obj.y, obj.z],
        "rotation_y": obj.ry,
        "bev_polygon": obj.bev_polygon,
        "y_top_bottom": [obj.y_top, obj.y_bottom] if obj.y_top is not None and obj.y_bottom is not None else None,
    }


def collect_predictions_and_groundtruth(
    dataset_root: Path,
    checkpoint_path: Path,
    num_images: int | None,
    nms_thresh: float,
    dimension_order: str,
):
    image_dir = dataset_root / "image_2"
    calib_dir = dataset_root / "calib"
    label_dir = dataset_root / "label_2"

    image_ids = sorted(path.stem for path in image_dir.glob("*.png"))
    if num_images is not None and num_images > 0:
        image_ids = image_ids[:num_images]

    dtype = torch.float32
    pad_hw = (H_PADDED, W_PADDED)
    grid = make_grid(
        grid_size=GRID_SIZE,
        grid_offset=(-GRID_SIZE[0] / 2.0, Y_OFFSET, 0.0),
        grid_res=GRID_RES,
        dtype=dtype,
    )
    model = build_model(checkpoint_path, dtype=dtype)

    all_groundtruth: list[list[EvalObject]] = []
    all_detections: list[list[EvalObject]] = []
    prediction_records: list[dict] = []

    for index, image_id in enumerate(image_ids, start=1):
        image_path = image_dir / f"{image_id}.png"
        calib_path = calib_dir / f"{image_id}.txt"
        label_path = label_dir / f"{image_id}.txt"

        predicted_objects, calib, image_hw = infer_image(
            model,
            image_path=image_path,
            calib_path=calib_path,
            grid=grid,
            pad_hw=pad_hw,
            nms_thresh=nms_thresh,
            dtype=dtype,
        )

        frame_detections: list[EvalObject] = []
        for obj in predicted_objects:
            det_obj = prediction_to_eval_object(obj, calib, image_hw, dimension_order=dimension_order)
            if det_obj is not None:
                frame_detections.append(det_obj)

        frame_groundtruth = load_groundtruth(label_path)
        all_detections.append(frame_detections)
        all_groundtruth.append(frame_groundtruth)

        prediction_records.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "calib_path": str(calib_path),
                "label_path": str(label_path),
                "predictions": [serialize_eval_object(obj) for obj in frame_detections],
            }
        )

        if index % 10 == 0 or index == len(image_ids):
            gt_eval_count = sum(1 for obj in frame_groundtruth if normalize_class_name(obj.class_name) in CLASS_NAMES)
            print(f"[{index}/{len(image_ids)}] {image_id}: pred={len(frame_detections)} gt_objects={gt_eval_count}")

    return image_ids, all_groundtruth, all_detections, prediction_records


def print_iou_diagnostics(
    groundtruth: list[list[EvalObject]],
    detections: list[list[EvalObject]],
    classes: Iterable[str],
) -> None:
    """Print max-IoU diagnostics for BEV/3D to explain all-zero AP."""

    print()
    print("BEV/3D IoU diagnostics before AP thresholding")
    for class_name in classes:
        class_name = normalize_class_name(class_name)
        for metric_name, overlap_fn in (("ground", ground_box_overlap), ("box3d", box3d_overlap)):
            best_ious: list[float] = []
            for frame_gt, frame_det in zip(groundtruth, detections):
                class_gt = [obj for obj in frame_gt if is_same_class(obj, class_name)]
                class_det = [obj for obj in frame_det if is_same_class(obj, class_name)]
                for gt_obj in class_gt:
                    best_ious.append(max((overlap_fn(det_obj, gt_obj, -1) for det_obj in class_det), default=0.0))

            if not best_ious:
                print(f"  {class_name:10s} {metric_name:6s}: no GT objects")
                continue

            threshold = MIN_OVERLAP[metric_name][class_name]
            max_iou = max(best_ious)
            mean_iou = sum(best_ious) / len(best_ious)
            hits = sum(1 for value in best_ious if value >= threshold)
            print(
                f"  {class_name:10s} {metric_name:6s}: "
                f"max_iou={max_iou:.4f} mean_best_iou={mean_iou:.4f} "
                f"gt_with_iou>={threshold:.1f}: {hits}/{len(best_ious)}"
            )


def write_stats_files(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix_by_metric = {
        "image": "detection",
        "ground": "detection_ground",
        "box3d": "detection_3d",
    }

    for class_name, class_results in results.items():
        class_lower = class_name.lower()
        for metric, metric_results in class_results.items():
            suffix = suffix_by_metric[metric]
            stats_path = output_dir / f"stats_{class_lower}_{suffix}.txt"
            with open(stats_path, "w") as f:
                for difficulty_name in DIFFICULTY_NAMES:
                    curve = metric_results[difficulty_name]["precision_curve"]
                    f.write(" ".join(f"{value:.6f}" for value in curve) + "\n")

            if metric == "image":
                has_aos = all(
                    metric_results[difficulty_name].get("aos_curve") is not None for difficulty_name in DIFFICULTY_NAMES
                )
                if has_aos:
                    aos_path = output_dir / f"stats_{class_lower}_orientation.txt"
                    with open(aos_path, "w") as f:
                        for difficulty_name in DIFFICULTY_NAMES:
                            curve = metric_results[difficulty_name]["aos_curve"]
                            f.write(" ".join(f"{value:.6f}" for value in curve) + "\n")


def print_summary(results: dict) -> None:
    print()
    print("KITTI-style evaluation finished")
    print("AP is 41-point AP, following evaluate_object.cpp")
    print()

    for class_name, class_results in results.items():
        print(f"[{class_name}]")
        for metric, metric_results in class_results.items():
            easy = 100.0 * metric_results["easy"]["ap"]
            moderate = 100.0 * metric_results["moderate"]["ap"]
            hard = 100.0 * metric_results["hard"]["ap"]
            print(f"  {metric:6s} AP: easy={easy:7.3f}  moderate={moderate:7.3f}  hard={hard:7.3f}")

            if metric == "image" and metric_results["easy"].get("aos_ap") is not None:
                easy_aos = 100.0 * metric_results["easy"]["aos_ap"]
                moderate_aos = 100.0 * metric_results["moderate"]["aos_ap"]
                hard_aos = 100.0 * metric_results["hard"]["aos_ap"]
                print(f"  aos    AP: easy={easy_aos:7.3f}  moderate={moderate_aos:7.3f}  hard={hard_aos:7.3f}")
        print()


def parse_classes(value: list[str]) -> list[str]:
    if len(value) == 1 and value[0].lower() == "all":
        return list(CLASS_NAMES)
    classes = [normalize_class_name(item) for item in value]
    unknown = [item for item in classes if item not in CLASS_NAMES]
    if unknown:
        raise ValueError(f"Unknown class(es): {unknown}. Supported: {CLASS_NAMES}")
    return classes


def parse_metrics(value: list[str]) -> list[str]:
    if len(value) == 1 and value[0].lower() == "all":
        return list(METRIC_NAMES)
    metrics = [item.lower() for item in value]
    unknown = [item for item in metrics if item not in METRIC_NAMES]
    if unknown:
        raise ValueError(f"Unknown metric(s): {unknown}. Supported: {METRIC_NAMES}")
    return metrics


def evaluate_dataset(
    dataset_root=DEFAULT_DATASET_ROOT,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    num_images: int | None = 100,
    classes: Iterable[str] = CLASS_NAMES,
    metrics: Iterable[str] = METRIC_NAMES,
    nms_thresh=NMS_THRESH,
    predictions_json_path=None,
    stats_dir=None,
    dimension_order: str = "hwl",
    debug_iou: bool = False,
):
    dataset_root = Path(dataset_root)
    checkpoint_path = Path(checkpoint_path)

    image_ids, groundtruth, detections, prediction_records = collect_predictions_and_groundtruth(
        dataset_root=dataset_root,
        checkpoint_path=checkpoint_path,
        num_images=num_images,
        nms_thresh=nms_thresh,
        dimension_order=dimension_order,
    )

    if debug_iou:
        print_iou_diagnostics(groundtruth, detections, classes)

    results = evaluate_kitti_style(
        groundtruth=groundtruth,
        detections=detections,
        classes=classes,
        metrics=metrics,
        compute_aos=True,
    )

    print_summary(results)

    if stats_dir is not None:
        stats_dir = Path(stats_dir)
        write_stats_files(results, stats_dir)
        print(f"saved KITTI-style stats files: {stats_dir}")

    if predictions_json_path is not None:
        predictions_json_path = Path(predictions_json_path)
        predictions_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(predictions_json_path, "w") as f:
            json.dump(
                {
                    "num_images": len(image_ids),
                    "classes": list(classes),
                    "metrics": list(metrics),
                    "dimension_order": dimension_order,
                    "kitti_style_results": results,
                    "predictions": prediction_records,
                },
                f,
                indent=2,
            )
        print(f"saved predictions + metrics json: {predictions_json_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run OFTNet inference and KITTI official-style object detection evaluation."
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--num-images", type=int, default=100, help="Use <=0 to evaluate all images found in image_2.")
    parser.add_argument("--classes", nargs="+", default=["all"], help="all, or any of: Car Pedestrian Cyclist")
    parser.add_argument("--metrics", nargs="+", default=["all"], help="all, or any of: image ground box3d")
    parser.add_argument("--nms-thresh", type=float, default=NMS_THRESH)
    parser.add_argument("--dimension-order", default="hwl", choices=["hwl", "lhw", "wlh", "whl", "lwh", "hlw"])
    parser.add_argument("--predictions-json", default="predictions_kitti_style.json")
    parser.add_argument("--stats-dir", default=None, help="Optional output directory for KITTI stats_*.txt files.")
    parser.add_argument(
        "--debug-iou", action="store_true", help="Print BEV/3D max-IoU diagnostics before AP computation."
    )
    args = parser.parse_args()

    classes = parse_classes(args.classes)
    metrics = parse_metrics(args.metrics)
    num_images = None if args.num_images is not None and args.num_images <= 0 else args.num_images

    evaluate_dataset(
        dataset_root=args.dataset_root,
        checkpoint_path=args.checkpoint,
        num_images=num_images,
        classes=classes,
        metrics=metrics,
        nms_thresh=args.nms_thresh,
        predictions_json_path=args.predictions_json,
        stats_dir=args.stats_dir,
        dimension_order=args.dimension_order,
        debug_iou=args.debug_iou,
    )


if __name__ == "__main__":
    main()
