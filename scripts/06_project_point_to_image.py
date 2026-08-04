"""
Tutorial 06: Project MULTI-LiDAR points onto a camera image

"""

import cv2
import numpy as np
import laspy
import json
from pathlib import Path
from bisect import bisect_left

# ================================
# Configuration
# ================================

SEQUENCE_FOLDER = Path("/home/hanpb2/workspace/Data/DataOD3D/code/3d-bat/input/hanpb2/20260711_1512_VF6_03_1783757531_1783759331")

LIDAR_SUB_FOLDER = [
    "LIDAR_TOP"
]

CAMERA = "CAM_P_F"
LIDAR_TIMESTAMP = "1783757599-099982977"

INPUT_CAM_FOLDER = SEQUENCE_FOLDER / "images"

CALIB_ROOT = Path("/home/hanpb2/workspace/Data/Data_PNK/calib/VF6_03")
INTRINSIC_JSON = CALIB_ROOT / "VF6_03_Intrinsics.json"
EXTRINSIC_JSON = CALIB_ROOT / "VF6_03_Extrinsics.json"

TIME_TOLERANCE_MS = 10
UNDISTORT_BALANCE = 1.0  # 0=crop, 1=full FOV

# side lidar filter
PAIR_THRESHOLD_NS = 40_000_000
CYLINDER_RADIUS = 6.0
MIN_Z_SIDE = -0.5
MAX_Z_SIDE = 2.0

DEPTH_BINS = np.arange(0, 300, 10)
DEPTH_COLORS = [(255,255,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255)]*6
DEPTH_COLORS = DEPTH_COLORS[:len(DEPTH_BINS)-1]

POINT_RADIUS_DEFAULT = 1


def point_radius(camera: str, D: np.ndarray, is_fisheye: bool) -> int:
    # 5-coeff pinhole (e.g. CAM_P_60_FCAM): keep; 8-coeff or fisheye: half radius
    if is_fisheye or len(np.asarray(D).ravel()) >= 8:
        return max(0, POINT_RADIUS_DEFAULT // 2)
    return POINT_RADIUS_DEFAULT


CAMERA_OFFSETS_MS = {
    "CAM_P_LB": 0,
    "CAM_P_L": 0,
    "CAM_P_FL": 0,
    "CAM_P_FL_New": 0,
    "CAM_P_BL_New": 0,
    "CAM_P_F": -33.33,
    "CAM_P_60_FCAM": -33.33,
    "CAM_P_120_FCAM": -33.33,
    "CAM_P_FR": -33.33,
    "CAM_P_FR_New": -66.67,
    "CAM_P_RB": -66.67,
    "CAM_P_BR_New": -66.67,
    "CAM_P_B_New": -66.67,

    "CAM_F_F": -33.33,
    "CAM_F_L": 0,
    "CAM_F_R": -33.33,
    "CAM_F_B": -66.67,
}
# ================================
# Timestamp helpers
# ================================

def parse_timestamp(ts_str: str) -> int:
    sec, nsec = map(int, ts_str.split('-'))
    return sec * 1_000_000_000 + nsec

def ns_to_str(ns: int) -> str:
    sec = ns // 1_000_000_000
    usec = (ns % 1_000_000_000) // 1000
    return f"{sec}-{usec:06d}000"

def closest_lidar_file(ts_target: int, ts_sorted: list[int], ts_to_file: dict[int, str]) -> str | None:
    if not ts_sorted:
        return None
    idx = bisect_left(ts_sorted, ts_target)
    cands = []
    if idx < len(ts_sorted):
        cands.append((abs(ts_sorted[idx] - ts_target), ts_sorted[idx]))
    if idx > 0:
        cands.append((abs(ts_sorted[idx - 1] - ts_target), ts_sorted[idx - 1]))
    diff, best = min(cands)
    return ts_to_file[best] if diff <= PAIR_THRESHOLD_NS else None

def index_side_sensors(sequence_folder: Path) -> dict[str, tuple[list[int], dict[int, str]]]:
    out = {}
    for sensor in LIDAR_SUB_FOLDER:
        if sensor == "LIDAR_TOP":
            continue
        folder = sequence_folder / "point_clouds" / sensor
        if not folder.is_dir():
            out[sensor] = ([], {})
            continue
        ts_list, ts_to_file = [], {}
        for p in folder.glob("*.laz"):
            try:
                ns = parse_timestamp(p.stem)
                ts_list.append(ns)
                ts_to_file[ns] = p.name
            except (ValueError, IndexError):
                pass
        ts_list.sort()
        out[sensor] = (ts_list, ts_to_file)
    return out


# ================================
# Load calibrations
# ================================

with open(INTRINSIC_JSON) as f:
    INTRINSICS = json.load(f)

with open(EXTRINSIC_JSON) as f:
    EXTRINSICS = json.load(f)

# lidar extrinsics
lidar_T = {}
for l in LIDAR_SUB_FOLDER:
    lidar_T[l] = np.array(EXTRINSICS[l])

# camera extrinsic
Rt = np.array(EXTRINSICS[CAMERA])[:3,:]

# camera intrinsics
cam_data = INTRINSICS[CAMERA]
K = np.array(cam_data["camera_matrix"]).reshape(3,3)
D = np.array(cam_data["distortion_coefficients"])

is_fisheye = CAMERA.startswith("CAM_F")
POINT_RADIUS = point_radius(CAMERA, D, is_fisheye)

# ================================
# Find closest camera
# ================================

offset_ms = CAMERA_OFFSETS_MS[CAMERA]

lidar_ns = parse_timestamp(LIDAR_TIMESTAMP)
ideal_camera_ns = lidar_ns + int(offset_ms * 1e6)

cam_dir = INPUT_CAM_FOLDER / CAMERA
cam_files = list(cam_dir.glob("*.jpg"))

ts_list=[]
ts_to_file={}

for f in cam_files:
    try:
        ns=parse_timestamp(f.stem)
        ts_list.append(ns)
        ts_to_file[ns]=f
    except:
        pass

ts_list.sort()

idx = bisect_left(ts_list, ideal_camera_ns)

cands=[]
if idx < len(ts_list):
    cands.append((abs(ts_list[idx]-ideal_camera_ns),ts_list[idx]))
if idx>0:
    cands.append((abs(ts_list[idx-1]-ideal_camera_ns),ts_list[idx-1]))

diff,best_ns=min(cands)

best_path=ts_to_file[best_ns]

print("Camera frame:", best_path.name)

# ================================
# Load image
# ================================

img = cv2.imread(str(best_path))
h, w = img.shape[:2]

if is_fisheye:
    K_proj = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=UNDISTORT_BALANCE
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_proj, (w, h), cv2.CV_16SC2
    )
    img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    D_proj = np.zeros(5, dtype=np.float64)
else:
    K_proj, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=UNDISTORT_BALANCE)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), K_proj, (w, h), cv2.CV_16SC2)
    img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    D_proj = np.zeros(5, dtype=np.float64)

# ================================
# Load and merge lidar (in-memory, no LAZ concat file)
# ================================

target_ns = parse_timestamp(LIDAR_TIMESTAMP)
side_index = index_side_sensors(SEQUENCE_FOLDER)
points_all = []

for sensor in LIDAR_SUB_FOLDER:
    if sensor == "LIDAR_TOP":
        path = SEQUENCE_FOLDER / "point_clouds" / sensor / f"{LIDAR_TIMESTAMP}.laz"
    else:
        ts_list, ts_to_file = side_index[sensor]
        fname = closest_lidar_file(target_ns, ts_list, ts_to_file)
        if fname is None:
            print(f"Skip {sensor}: no scan within {PAIR_THRESHOLD_NS / 1e6:.0f} ms")
            continue
        path = SEQUENCE_FOLDER / "point_clouds" / sensor / fname
    print("path:", path)
    if not path.exists():
        continue
    
    las = laspy.read(path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    pts = np.vstack((x, y, z, np.ones_like(x)))
    tv = lidar_T[sensor] @ pts
    xt, yt, zt = tv[0], tv[1], tv[2]

    print(f"Loaded {sensor}: {len(xt):,} points ({path.name})")
    if sensor.startswith("LIDAR_E_"):
        x_shift = xt - 0.910
        dist_sq = x_shift ** 2 + yt ** 2
        keep = (dist_sq <= CYLINDER_RADIUS ** 2) & (zt >= MIN_Z_SIDE) & (zt <= MAX_Z_SIDE)
        xt, yt, zt = xt[keep], yt[keep], zt[keep]

    if len(xt) == 0:
        continue

    print(f"{sensor}: {len(xt):,} points ({path.name})")
    points_all.append(np.vstack((xt, yt, zt)).T)

if not points_all:
    raise RuntimeError("No LiDAR points merged")

points = np.concatenate(points_all)
print("Merged points:", len(points))

# ================================
# Project
# ================================

points_h=np.hstack((points,np.ones((len(points),1))))
points_cam=(Rt@points_h.T).T

valid=points_cam[:,2]>0.1
points_cam=points_cam[valid]

dist=np.linalg.norm(points_cam[:,:3],axis=1)

bin_idx=np.digitize(dist,DEPTH_BINS)-1
bin_idx=np.clip(bin_idx,0,len(DEPTH_COLORS)-1)

obj_pts=points_cam[:,np.newaxis,:].astype(np.float32)

rvec=tvec=np.zeros(3)

# After undistort (pinhole image), always use standard projectPoints.
uv, _ = cv2.projectPoints(
    obj_pts, rvec, tvec, K_proj.astype(np.float32), D_proj.astype(np.float32)
)

uv=uv.reshape(-1,2).astype(int)

out=img.copy()

valid_count=0

for i,(u,v) in enumerate(uv):

    if 0<=u<w and 0<=v<h:

        cv2.circle(out,(u,v),POINT_RADIUS,DEPTH_COLORS[bin_idx[i]],-1)
        valid_count+=1

print("Projected:",valid_count)

# ================================
# Display (before / after)
# ================================

fixed = (960, 960)
left = cv2.resize(img, fixed)
right = cv2.resize(out, fixed)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(left, "Before", (30, 60), font, 1.5, (0, 0, 0), 4, cv2.LINE_AA)
cv2.putText(left, "Before", (30, 60), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(right, "After", (30, 60), font, 1.5, (0, 0, 0), 4, cv2.LINE_AA)
cv2.putText(right, "After", (30, 60), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

preview = np.hstack((left, right))
window_name = f"{CAMERA} | Before vs After | balance={UNDISTORT_BALANCE:g}"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name, preview)
cv2.waitKey(0)
cv2.destroyAllWindows()