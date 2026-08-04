"""
Server-side re-implementation of tool_image.ts's calculateProjectedBoundingBox
and projectPoints, so the 3D->2D projection math can run in Python instead
of the browser.

Mirrors the TS logic 1:1, including:
- coordinate-system-dependent corner ordering (forward/left/up vs forward/right/up)
- yaw-only rotation about the box center
- the "vehicle_camera_basler_16mm" infrastructure<->vehicle transform special case
- dropping boxes where any corner projects behind the camera (point2D[2] <= 0)
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
    # print("x_dir: ", x_dir, "y_dir: ", y_dir, "z_dir: ", z_dir)
    # print("x: ", x, "y: ", y, "z: ", z, "length: ", length, "width: ", width, "height: ", height)
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


def calculate_projected_bounding_box(
    x_pos, y_pos, z_pos, length, width, height, yaw,
    coordinate_system,          # {"x-axis":..., "y-axis":..., "z-axis":...}
    zoom_factor,                # this.labelTool.imageScale[channelIdx]
    channel_name,
):
    x_dir = coordinate_system["x-axis"]
    y_dir = coordinate_system["y-axis"]
    z_dir = coordinate_system["z-axis"]

    # Generate the 8 box corners
    corners = _corner_points(
        x_pos,
        y_pos,
        z_pos,
        length,
        width,
        height,
        x_dir,
        y_dir,
        z_dir,
    )

    # Rotation matrix around Z
    rot = _rotation_matrix_z(yaw)

    rotated_points = []
    for cx, cy, cz in corners:
        # Corner in local object coordinates
        local = np.array(
            [
                cx - x_pos,
                cy - y_pos,
                cz - z_pos,
                1.0,
            ],
            dtype=np.float64,
        )
        # Rotate around object center
        rotated = rot @ local
        # Back to world coordinates
        point = np.array(
            [
                rotated[0] + x_pos,
                rotated[1] + y_pos,
                rotated[2] + z_pos,
            ],
            dtype=np.float64,
        )
        rotated_points.append(point)

    # Add box center (9th point)
    center = np.array([x_pos, y_pos, z_pos], dtype=np.float64)
    rotated_points.append(center)

    # (9,3) -> (3,9)
    bbox = np.asarray(rotated_points, dtype=np.float64).T
    # (1,1,3,9)
    bbox = torch.from_numpy(bbox).unsqueeze(0).unsqueeze(0)

    # Project using fisheye camera
    image_points = project_bbox3d(
        bbox,
        channel_name
    )
    # (9,2)
    pts = image_points[0, 0].permute(1, 0)

    # Apply display zoom
    pts *= zoom_factor
    # print("zoom_factor: ", zoom_factor)
    # Return only the 8 box corners
    return pts[:8].cpu().numpy().tolist()


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