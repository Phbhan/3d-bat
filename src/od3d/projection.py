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
from src.od3d.od3d import project_bbox3d, fisheye_cams


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

    Returns:
        tuple:
            - points: the 8 corner points as [[x, y], ...] (Python list),
              scaled by zoom_factor.
            - valid_mask: list of 8 bools. True where that corner's raw
              pixel coordinates are trustworthy (its true angle from the
              optical axis is inside the lens's calibrated FOV). A corner
              marked False should not be connected to another corner using
              its raw pixel value — see `calculate_projected_bounding_box_edges`
              for a version that clips the box outline at the FOV boundary
              instead of returning unreliable corners.
    """
    points, valid_masks = calculate_projected_bounding_boxes(
        boxes=[{
            "x": x_pos, "y": y_pos, "z": z_pos,
            "length": length, "width": width, "height": height, "yaw": yaw,
        }],
        coordinate_system=coordinate_system,
        zoom_factor=zoom_factor,
        channel_name=channel_name,
    )
    # Always return the first (and only) element of the array
    return points[0], valid_masks[0]


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
        tuple:
            - points: list of length N, where each element is the 8 corner
              points as [[x, y], ...] (Python lists), scaled by
              zoom_factor. Always a list, even when N == 1.
            - valid_mask: list of length N, where each element is 8 bools
              (one per corner). True means that corner's raw pixel
              coordinates are trustworthy (its true angle from the optical
              axis is inside the lens's calibrated FOV). Corners marked
              False should not be connected to other corners using their
              raw pixel value — either skip that edge client-side or use
              `calculate_projected_bounding_box_edges` for an outline
              that's already clipped to the FOV boundary.
    """
    if isinstance(boxes, dict):
        boxes = [boxes]
    if not boxes:
        return [], []

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

    image_points, valid_mask = project_bbox3d(bbox, channel_name)
    # image_points: (1, N, 2, 9), valid_mask: (1, N, 9)

    pts = image_points[0]          # (N, 2, 9)
    pts = pts.permute(0, 2, 1)     # (N, 9, 2)
    pts = pts[:, :8, :]            # drop center point → (N, 8, 2)
    pts = pts * zoom_factor

    mask = valid_mask[0]           # (N, 9)
    mask = mask[:, :8]             # drop center point → (N, 8)

    return pts.cpu().numpy().tolist(), mask.cpu().numpy().tolist()  # (N x 8 x 2), (N x 8)


# The 12 edges of a box, indexed into the 8-corner ordering produced by
# _corner_points() (0-3 bottom face, 4-7 top face, same winding on both).
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]


def calculate_projected_bounding_box_edges_clipped(
    x_pos, y_pos, z_pos, length, width, height, yaw,
    coordinate_system,
    zoom_factor,
    channel_name,
):
    """
    Project a single 3D box's edges into one camera channel, clipping any
    edge that leaves the lens's calibrated FOV at the exact boundary
    instead of dropping the whole edge or letting the raw fisheye
    polynomial fold a far-outside corner back into the frame.

    NOTE: this returns 12 edge *segments* (each with its own two
    endpoints, since a clipped edge no longer shares a corner with its
    neighbor), not 8 shared corner points. tool_image.ts's
    calculateAndDrawLineSegments currently expects exactly 8 shared corner
    points indexed 0-7, so this function is not a drop-in replacement for
    the current /project_bounding_box endpoint — use
    calculate_projected_bounding_box_edges (below) for that. This one is
    for a future frontend that draws pre-built segments directly.

    Returns:
        List of 12 entries (one per BOX_EDGES edge), each either
        [[x1, y1], [x2, y2]] (already scaled by zoom_factor) or None if
        that edge falls entirely outside the FOV and should be skipped.
    """
    projection_cam = fisheye_cams[channel_name]

    corners_world = _box_to_corners_tensor(
        x_pos, y_pos, z_pos, length, width, height, yaw, coordinate_system,
    )  # (3, 9): 8 corners + center
    corners_world = torch.from_numpy(corners_world[:, :8]).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 8)

    corners_cam = projection_cam.project_world_to_cam(corners_world)  # (1, 1, 3, 8)
    corners_cam = corners_cam[0, 0]  # (3, 8)

    edges = projection_cam.clip_box_edges_to_fov(corners_cam, BOX_EDGES)

    return [
        None if edge is None else [
            (edge[0] * zoom_factor).tolist(),
            (edge[1] * zoom_factor).tolist(),
        ]
        for edge in edges
    ]


def calculate_projected_bounding_box_edges(
    boxes,
    coordinate_system,
    zoom_factor,
    channel_name,
):
    """
    Batched projection for the /project_bounding_box endpoint, compatible
    with tool_image.ts's existing 8-shared-corner-point contract.

    Same corner shape/order as calculate_projected_bounding_boxes, but any
    corner whose true angle from the optical axis falls outside the lens's
    calibrated FOV is returned as `None` instead of its raw pixel value
    (which past that point is not a faithful projection — see
    FisheyeCam.project_cam_to_fe_image). `None` survives JSON as `null`
    (unlike NaN/Infinity, which Python's json module happily emits but
    JS's strict JSON.parse rejects, so the browser's response.json() call
    would throw). tool_image.ts's drawLine() already guards on
    `pointStart !== undefined` and `isFinite(...)`, so once the Vector2
    conversion in projectBoundingBoxes() maps a `null` entry to
    `undefined` (one-line change — see below), every line segment
    touching an out-of-FOV corner is silently skipped, with no other
    frontend changes required.

        // in projectBoundingBoxes(), replace:
        //   boxPoints.map(pt => new THREE.Vector2(pt[0], pt[1]))
        // with:
        //   boxPoints.map(pt => pt === null ? undefined : new THREE.Vector2(pt[0], pt[1]))

    This trades precision for compatibility: an edge with one out-of-FOV
    corner is dropped entirely rather than clipped exactly at the FOV
    boundary (see calculate_projected_bounding_box_edges_clipped above
    for that sharper version, which needs a bigger frontend change since
    it returns edge segments instead of shared corners).

    Returns:
        list of length N (one per input box), each element the 8 corner
        points as [[x, y], ...] scaled by zoom_factor, with `None` in
        place of any corner outside the calibrated FOV. Always a list,
        even when N == 1.
    """
    points, valid_mask = calculate_projected_bounding_boxes(
        boxes=boxes,
        coordinate_system=coordinate_system,
        zoom_factor=zoom_factor,
        channel_name=channel_name,
    )

    return [
        [corner if valid else None for corner, valid in zip(box_points, box_mask)]
        for box_points, box_mask in zip(points, valid_mask)
    ]


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