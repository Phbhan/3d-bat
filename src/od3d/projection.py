"""
Server-side re-implementation of tool_image.ts's calculateProjectedBoundingBox
and projectPoints, so the 3D->2D projection math can run in Python instead
of the browser.

Mirrors the TS logic 1:1, including:
- coordinate-system-dependent corner ordering (forward/left/up vs forward/right/up)
- yaw-only rotation about the box center
- the "vehicle_camera_basler_16mm" infrastructure<->vehicle transform special case
- dropping boxes where any corner projects behind the camera (point2D[2] <= 0)

Supports batched projection of multiple boxes in a single call.
"""
import numpy as np
import torch
from src.od3d.od3d import project_bbox3d


def _rotation_matrix_z(yaw: float) -> np.ndarray:
    """4x4 homogeneous rotation matrix about Z, matching THREE.Euler(0,0,yaw,'XYZ')."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ])


def _corner_points(x, y, z, length, width, height, x_dir, y_dir, z_dir):
    # z is forced to height/2 so the box sits on the ground plane (matches TS).
    z = height / 2
    if x_dir == "forward" and y_dir == "left" and z_dir == "up":
        return [
            (x + length / 2, y + width / 2, z - height / 2),
            (x + length / 2, y - width / 2, z - height / 2),
            (x - length / 2, y - width / 2, z - height / 2),
            (x - length / 2, y + width / 2, z - height / 2),
            (x + length / 2, y + width / 2, z + height / 2),
            (x + length / 2, y - width / 2, z + height / 2),
            (x - length / 2, y - width / 2, z + height / 2),
            (x - length / 2, y + width / 2, z + height / 2),
        ]
    elif x_dir == "forward" and y_dir == "right" and z_dir == "up":
        return [
            (x - length / 2, y + width / 2, z - height / 2),
            (x + length / 2, y + width / 2, z - height / 2),
            (x + length / 2, y - width / 2, z - height / 2),
            (x - length / 2, y - width / 2, z - height / 2),
            (x - length / 2, y + width / 2, z + height / 2),
            (x + length / 2, y + width / 2, z + height / 2),
            (x + length / 2, y - width / 2, z + height / 2),
            (x - length / 2, y - width / 2, z + height / 2),
        ]
    else:
        raise ValueError(f"Unsupported coordinate system: x={x_dir}, y={y_dir}, z={z_dir}")


def _box_to_corners_tensor(x_pos, y_pos, z_pos, length, width, height, yaw, coordinate_system):
    """
    Build a single box's 9 points (8 corners + center) as a (3, 9) float64 array.
    """
    x_dir = coordinate_system["x-axis"]
    y_dir = coordinate_system["y-axis"]
    z_dir = coordinate_system["z-axis"]

    corners = _corner_points(
        x_pos, y_pos, z_pos, length, width, height, x_dir, y_dir, z_dir
    )

    rot = _rotation_matrix_z(yaw)

    rotated_points = []
    for cx, cy, cz in corners:
        local = np.array([cx - x_pos, cy - y_pos, cz - z_pos, 1.0], dtype=np.float64)
        rotated = rot @ local
        point = np.array(
            [rotated[0] + x_pos, rotated[1] + y_pos, rotated[2] + z_pos],
            dtype=np.float64,
        )
        rotated_points.append(point)

    # 9th point = box center
    center = np.array([x_pos, y_pos, z_pos], dtype=np.float64)
    rotated_points.append(center)

    # (9, 3) -> (3, 9)
    return np.asarray(rotated_points, dtype=np.float64).T


def calculate_projected_bounding_box(
    x_pos, y_pos, z_pos, length, width, height, yaw,
    coordinate_system,
    zoom_factor,
    channel_name,
):
    """
    Project a single 3D box into one camera channel.
    Thin wrapper around the batched path so existing single-box callers keep working.
    """
    results = calculate_projected_bounding_boxes(
        boxes=[{
            "x": x_pos, "y": y_pos, "z": z_pos,
            "length": length, "width": width, "height": height, "yaw": yaw,
        }],
        coordinate_system=coordinate_system,
        zoom_factor=zoom_factor,
        channel_name=channel_name,
    )
    # Always return the first (and only) element of the array
    return results[0]


def calculate_projected_bounding_boxes(
    boxes,
    coordinate_system,
    zoom_factor,
    channel_name,
):
    """
    Project multiple 3D boxes into one camera channel in a single fisheye call.

    Args:
        boxes: list of dicts, each with keys
               x, y, z, length, width, height, yaw
               (also accepts a single dict — it is wrapped into a list)
        coordinate_system: {"x-axis":..., "y-axis":..., "z-axis":...}
        zoom_factor: display scale for this channel
        channel_name: key into fisheye_cams, e.g. "CAM_FRONT"

    Returns:
        list of length N, where each element is the 8 corner points as
        [[x, y], ...] (Python lists).  Always a list, even when N == 1.
    """
    if isinstance(boxes, dict):
        boxes = [boxes]
    if not boxes:
        return []

    # Stack every box into (N, 3, 9)
    stacked = np.stack(
        [
            _box_to_corners_tensor(
                b["x"], b["y"], b["z"],
                b["length"], b["width"], b["height"],
                b["yaw"],
                coordinate_system,
            )
            for b in boxes
        ],
        axis=0,
    )  # (N, 3, 9)

    # (1, N, 3, 9) — batch dimension is the second axis, matching the original
    # single-box layout of (1, 1, 3, 9)
    bbox = torch.from_numpy(stacked).unsqueeze(0)

    image_points = project_bbox3d(bbox, channel_name)
    # image_points expected shape: (1, N, 2, 9) or compatible

    pts = image_points[0]          # (N, 2, 9)
    pts = pts.permute(0, 2, 1)     # (N, 9, 2)
    pts = pts[:, :8, :]            # drop center point → (N, 8, 2)
    pts = pts * zoom_factor

    return pts.cpu().numpy().tolist()  # list[list[list[float]]]  N x 8 x 2


def project_points(points3d, projection_matrix, scaling_factor):
    """
    points3d: list of [x, y, z]
    Returns: {"points2D": [[x,y], ...], "points3D": [[x,y,z], ...], "distances": [...]}
    Only points in front of the camera (point2D[2] > 0) are kept, matching TS.
    """
    proj = np.array(projection_matrix, dtype=float)
    points2d, kept_points3d, distances = [], [], []

    for p in points3d:
        p_h = np.array([p[0], p[1], p[2], 1.0])
        point2d = proj @ p_h
        if point2d[2] > 0:
            window_x = point2d[0] / point2d[2]
            window_y = point2d[1] / point2d[2]
            kept_points3d.append(p)
            distances.append(float(np.linalg.norm(p_h[:3])))
            points2d.append([window_x / scaling_factor, window_y / scaling_factor])

    return {"points2D": points2d, "points3D": kept_points3d, "distances": distances}