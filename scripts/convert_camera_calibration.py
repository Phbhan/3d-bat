"""
Convert a "cameraData.json" style calibration file (camPos, matrixR, vectT,
matrixK, matrixD) into 3d-bat's `camera_channels` config.json format
(positionCamera, fieldOfView, rotationYaw, projectionMatrix, rotation).

Assumptions (verified by reverse-engineering the existing config.json's
CAM_FRONT / CAM_FRONT_LEFT / CAM_BACK / CAM_FRONT_RIGHT entries):

- World/vehicle coordinate system: x = forward, y = right, z = up
  (matches config.json's "coordinate_system").
- matrixR / vectT define the WORLD -> CAMERA transform:
      X_cam = R @ X_world + T
- matrixK is given as [fx, cx, fy, cy] (square pixels). Image width/height
  are estimated as 2*cx / 2*cy (principal point assumed centered), used
  only to estimate fieldOfView.
- camPos order: 0 -> left, 1 -> front, 2 -> back, 3 -> right
  (as specified).

Derivations:
- positionCamera (camera center in world coords): C = -R^T @ T
- rotationYaw: yaw of the camera's forward/boresight direction, where the
  forward direction in world coords is the THIRD ROW of R.
  yaw = atan2(fwd_y, fwd_x), normalized to [0, 2*pi).
  (Verified: camPos=1 (front) -> yaw ~= 0 rad; camPos=2 (back) -> yaw ~= pi;
  matches CAM_FRONT / CAM_BACK's rotationYaw pattern in config.json.)
- rotation (integer degrees used for UI/CSS layout): round(degrees(yaw)).
  (Verified: this exactly reproduces config.json's rotation field from its
  rotationYaw field for all 4 existing cameras: 305<->5.323 rad,
  0<->0 rad, 110<->1.920 rad, 180<->3.14159 rad.)
- fieldOfView: 2 * atan(image_width / (2*fx)), rounded to nearest degree.
- projectionMatrix: P = K @ [R | T], a 3x4 matrix, where
      K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

Usage:
    python scripts/convert_camera_calibration.py /home/hanpb2/PycharmProjects/OD3D-dev/model_converter_3d/configs/pnk_vf6/cameraData.json [output.json]
"""
import json
import math
import sys

# camPos -> channel name mapping, per the stated order (0: left, 1: front,
# 2: back, 3: right). Rename these to match whatever folder/channel naming
# convention your dataset actually uses (e.g. CAM_LEFT vs LIDAR_LEFT_CAM).
CAM_POS_TO_CHANNEL = {
    0: "CAM_LEFT",
    1: "CAM_FRONT",
    2: "CAM_BACK",
    3: "CAM_RIGHT",
}


def build_intrinsics(matrix_k):
    fx, cx, fy, cy = matrix_k
    k = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]
    return k, fx, fy, cx, cy


def matmul(a, b):
    """a: m x n, b: n x p -> m x p (plain python lists of lists)."""
    m, n = len(a), len(a[0])
    n2, p = len(b), len(b[0])
    assert n == n2, f"Shape mismatch: {n} != {n2}"
    result = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            result[i][j] = sum(a[i][k] * b[k][j] for k in range(n))
    return result


def transpose(a):
    return [list(row) for row in zip(*a)]


def convert_camera(item):
    cam_pos = item["camPos"]
    r_flat = item["matrixR"]
    t = item["vectT"]
    k_flat = item["matrixK"]

    r = [r_flat[0:3], r_flat[3:6], r_flat[6:9]]
    k, fx, fy, cx, cy = build_intrinsics(k_flat)

    # --- camera position in world/vehicle coords: C = -R^T @ T ---
    r_t = transpose(r)
    c = [-sum(r_t[i][j] * t[j] for j in range(3)) for i in range(3)]

    # --- yaw of the camera's forward (boresight) direction ---
    # forward direction in world coords = third row of R
    fwd = r[2]
    yaw_rad = math.atan2(fwd[1], fwd[0])
    if yaw_rad < 0:
        yaw_rad += 2 * math.pi

    rotation_deg = round(math.degrees(yaw_rad))

    # --- field of view estimate from focal length + assumed image width ---
    image_width = 2 * cx
    fov_rad = 2 * math.atan(image_width / (2 * fx))
    field_of_view = round(math.degrees(fov_rad))

    # --- projection matrix P = K @ [R | T] ---
    rt_matrix = [r[0] + [t[0]], r[1] + [t[1]], r[2] + [t[2]]]
    p = matmul(k, rt_matrix)

    channel_name = CAM_POS_TO_CHANNEL.get(cam_pos, f"CAM_{cam_pos}")

    return {
        "channel": channel_name,
        "positionCamera": c,
        "fieldOfView": field_of_view,
        "rotationYaw": yaw_rad,
        "projectionMatrix": p,
        "rotation": rotation_deg,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_camera_calibration.py cameraData.json [output.json]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "camera_channels_converted.json"

    with open(input_path) as f:
        data = json.load(f)

    channels = [convert_camera(item) for item in data["Items"]]

    with open(output_path, "w") as f:
        json.dump(channels, f, indent=2)

    print(f"Wrote {len(channels)} camera channels to {output_path}")
    for ch in channels:
        print(f"  {ch['channel']}: rotationYaw={ch['rotationYaw']:.4f} rad "
              f"({ch['rotation']} deg), fieldOfView={ch['fieldOfView']}, "
              f"positionCamera={[round(v, 3) for v in ch['positionCamera']]}")


if __name__ == "__main__":
    main()