#!/usr/bin/env python3
"""
Build a per-frame NAVSIM-like JSON from a single raw log folder, with REAL
lidar_path / camera data_path and REAL ego2global pose (from IMU/NAV).

Expected folder layout (matches what you described):

    <raw_root>/
        LIDAR/LIDAR_TOP/1783757599-099982977.laz          <- lidar frames (timestamp = filename)
        CAMERA/CAM_P_L/1783757599-031618213.jpg            <- one folder per camera
        CAMERA/CAM_P_FL/...
        CAMERA/CAM_P_FR/...
        CAMERA/CAM_P_LB/...
        CAMERA/CAM_P_RB/...
        OTHERS/IMU/*.csv   (or OTHER/IMU/*.csv)             <- pose source
        OTHERS/NAV/*.csv   (optional, for lat/lon translation)

For every LIDAR_TOP frame:
    1. Find the nearest image (by timestamp) in every camera folder found
       under CAMERA/.
    2. Find the nearest IMU/NAV row (by timestamp) and derive
       ego2global_translation / ego2global_rotation from it -- once for the
       LiDAR keyframe timestamp, and AGAIN for each camera's own matched
       image timestamp (they are not the same instant).
    3. Compute sensor2ego, sensor2lidar, camera_intrinsics, lidar2ego from
       the intrinsics/extrinsics calibration files.

       sensor2lidar is computed by bridging through the GLOBAL frame using
       each sensor's own ego pose at its own capture time, exactly like
       mmdet3d's `obtain_sensor2top`:

           sensor -> ego(sensor_time) -> global -> ego(lidar_time) -> lidar

       This matters because the LiDAR and each camera do NOT fire at the
       same instant (see CAMERA_OFFSETS_MS below), and the vehicle keeps
       moving in that gap. A naive `R_lidar2ego.T @ R_cam2ego` (bridging
       through a single shared ego frame) silently assumes ego pose is
       identical at both timestamps, which is wrong whenever the vehicle is
       moving.

Camera folder names (e.g. CAM_P_L, CAM_P_FL, CAM_P_FR, CAM_P_LB, CAM_P_RB)
are used as-is to look up intrinsics/extrinsics, and renamed to a friendlier
output key via RAW2OUT_CAM if a mapping exists (otherwise the raw folder
name is used as the output key directly, so any camera folder is handled
automatically -- you aren't limited to the ones listed above).

Usage:
    python build_synced_frames_json.py \
        --raw_root "500h/20260711_1512_VF6_03_1783757531_1783759331" \
        --intr_path VF6_03_Intrinsics.json \
        --extr_path VF6_03_Extrinsics.json \
        --primary_lidar LIDAR_TOP \
        --out_path 20260711_1512_VF6_03.json
"""
import argparse
import csv
import json
import math
import re
from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as SciRot

try:
    from pyproj import CRS, Transformer
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False

EARTH_RADIUS_M = 6378137.0
WGS84_EPSG = 4326

# Known offsets (ms) between LiDAR sweep time and each camera's ideal shutter
# time, copied from the full pipeline script. Cameras not listed use 0 ms.
CAMERA_OFFSETS_MS = {
    "CAM_P_LB": -100.00,
    "CAM_P_L": -66.67,
    "CAM_P_FL": -66.67,
    "CAM_P_F": -33.33,
    "CAM_P_FR": 0.0,
    "CAM_P_RB": 0.0,
    "CAM_P_R": 0.0,
    "CAM_P_B": 0.0,
}

# Raw camera folder name -> friendly output key. Anything not listed here
# just uses the raw folder name as the output key.
RAW2OUT_CAM = {
    "CAM_P_F": "CAM_FRONT",
    "CAM_P_FL": "CAM_FRONT_LEFT",
    "CAM_P_FR": "CAM_FRONT_RIGHT",
    "CAM_P_L": "CAM_LEFT",
    "CAM_P_LB": "CAM_BACK_LEFT",
    "CAM_P_R": "CAM_RIGHT",
    "CAM_P_RB": "CAM_BACK_RIGHT",
    "CAM_P_B": "CAM_BACK",
}


# --------------------------------------------------------------------------
# Timestamp / file indexing helpers
# --------------------------------------------------------------------------
def parse_sec_nsec(stem: str) -> Tuple[int, int]:
    stem = stem.strip()
    if "-" in stem:
        parts = stem.split("-")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return int(parts[0]), int(parts[1].ljust(9, "0")[:9])

    m = re.match(r"^(\d+)_(\d+)(?:_(?:pcd|laz|las|csv|jpg|jpeg|png))?$", stem)
    if m:
        return int(m.group(1)), int(m.group(2).ljust(9, "0")[:9])

    if stem.isdigit() and len(stem) > 10:
        ns = int(stem)
        return ns // 1_000_000_000, ns % 1_000_000_000

    if stem.isdigit():
        return int(stem), 0

    raise ValueError(f"Unsupported timestamp stem: {stem}")


def ts_ns_from_stem(stem: str) -> int:
    sec, nsec = parse_sec_nsec(stem)
    return sec * 1_000_000_000 + nsec


def ts_us_from_stem(stem: str) -> int:
    return ts_ns_from_stem(stem) // 1000


def normalize_key(key: str) -> str:
    return key.replace("\ufeff", "").strip() if key is not None else ""


def build_time_index(folder: Path, exts: Tuple[str, ...]) -> List[Tuple[int, Path]]:
    items: List[Tuple[int, Path]] = []
    for ext in exts:
        for p in folder.glob(f"*{ext}"):
            try:
                t = ts_ns_from_stem(p.stem)
            except Exception:
                continue
            items.append((t, p))
    items.sort(key=lambda x: x[0])
    return items


def nearest_by_time(index: List[Tuple[int, Path]], target_ns: int) -> Tuple[int, Path, float]:
    if not index:
        raise RuntimeError("No valid timestamps found")
    ts_list = [t for t, _ in index]
    idx = bisect_left(ts_list, target_ns)

    candidates: List[Tuple[int, int, Path]] = []
    if idx < len(index):
        candidates.append((abs(index[idx][0] - target_ns), index[idx][0], index[idx][1]))
    if idx > 0:
        candidates.append((abs(index[idx - 1][0] - target_ns), index[idx - 1][0], index[idx - 1][1]))

    diff_ns, best_ns, best_path = min(candidates, key=lambda x: x[0])
    return best_ns, best_path, diff_ns / 1_000_000.0


def ideal_camera_timestamp_ns(lidar_ns: int, raw_cam: str, use_offset: bool = True) -> int:
    if not use_offset:
        return lidar_ns
    offset_ms = float(CAMERA_OFFSETS_MS.get(raw_cam, 0.0))
    return lidar_ns + int(offset_ms * 1_000_000)


def discover_camera_folders(camera_root: Path) -> List[str]:
    """Any subfolder of CAMERA/ that contains at least one jpg/jpeg/png is a camera."""
    cams: List[str] = []
    if not camera_root.exists():
        return cams
    for sub in sorted(camera_root.iterdir()):
        if not sub.is_dir() or sub.name not in RAW2OUT_CAM.keys():
            continue
        if any(sub.glob("*.jpg")) or any(sub.glob("*.jpeg")) or any(sub.glob("*.png")):
            cams.append(sub.name)
    return cams


# --------------------------------------------------------------------------
# CSV / pose helpers (IMU / NAV)
# --------------------------------------------------------------------------
def load_csv_timeseries(folder: Path) -> List[Tuple[int, dict]]:
    records: List[Tuple[int, dict]] = []
    if not folder.exists():
        return records

    for csv_path in sorted(folder.glob("*.csv")):
        file_ts_ns: Optional[int] = None
        try:
            file_ts_ns = ts_ns_from_stem(csv_path.stem)
        except Exception:
            file_ts_ns = None

        try:
            with csv_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    reader.fieldnames = [normalize_key(k) for k in reader.fieldnames]
                for row in reader:
                    row = {normalize_key(k): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                    ts_raw = str(row.get("Timestamp", "")).strip()
                    ts_ns: Optional[int] = None
                    if ts_raw:
                        try:
                            ts_ns = ts_ns_from_stem(ts_raw)
                        except Exception:
                            ts_ns = None
                    if ts_ns is None:
                        ts_ns = file_ts_ns
                    if ts_ns is None:
                        continue
                    row["_timestamp_ns"] = int(ts_ns)
                    records.append((int(ts_ns), row))
        except Exception as e:
            print(f"[WARN] Failed reading {csv_path}: {e}")

    records.sort(key=lambda x: x[0])
    return records


def nearest_record_with_meta(
    records: List[Tuple[int, dict]], t_ns: int
) -> Tuple[Optional[dict], Optional[int], Optional[float]]:
    if not records:
        return None, None, None
    ts_list = [ts for ts, _ in records]
    idx = bisect_left(ts_list, t_ns)
    if idx == 0:
        best_ts, row = records[0]
    elif idx >= len(records):
        best_ts, row = records[-1]
    else:
        left = records[idx - 1]
        right = records[idx]
        best_ts, row = left if abs(left[0] - t_ns) <= abs(right[0] - t_ns) else right
    return row, int(best_ts), abs(best_ts - t_ns) / 1_000_000.0


def fget(row: Optional[dict], key: str, default: float = 0.0) -> float:
    if not row:
        return float(default)
    val = row.get(key, default)
    if val in (None, ""):
        return float(default)
    try:
        return float(val)
    except Exception:
        return float(default)


def latlon_to_local_xy(lat_deg: float, lon_deg: float, lat0_deg: float, lon0_deg: float) -> Tuple[float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    x = (lon - lon0) * math.cos(lat0) * EARTH_RADIUS_M
    y = (lat - lat0) * EARTH_RADIUS_M
    return x, y


def auto_utm_epsg(lat_deg: float, lon_deg: float) -> int:
    zone = int((float(lon_deg) + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if float(lat_deg) >= 0.0 else 32700) + zone


def build_global_transformer(mode: str, lat0: float, lon0: float, epsg: Optional[int]):
    mode = (mode or "").strip().lower()
    if mode in {"utm", "projected_global", "map_global"}:
        if not _HAS_PYPROJ:
            raise RuntimeError(
                "pyproj is required for --global_coord_mode utm/projected_global/map_global. "
                "Install it, or use --global_coord_mode local_enu."
            )
        target_epsg = int(epsg) if epsg else auto_utm_epsg(lat0, lon0)
        crs = CRS.from_epsg(target_epsg)
        transformer = Transformer.from_crs(CRS.from_epsg(WGS84_EPSG), crs, always_xy=True)
        return transformer, target_epsg, crs.to_string()
    return None, None, "LOCAL_ENU"


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    q_xyzw = SciRot.from_matrix(R.astype(np.float64)).as_quat()
    x, y, z, w = q_xyzw.tolist()
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-12)


def quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """Inverse of rotmat_to_quat_wxyz: [w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = [float(v) for v in np.asarray(q_wxyz).reshape(-1)]
    return SciRot.from_quat([x, y, z, w]).as_matrix().astype(np.float32)


def heading_pitch_roll_to_quat_wxyz(heading_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw_enu_deg = 90.0 - float(heading_deg)
    R = SciRot.from_euler("ZYX", [yaw_enu_deg, float(pitch_deg), float(roll_deg)], degrees=True).as_matrix()
    return rotmat_to_quat_wxyz(R.astype(np.float32))


def derive_pose_from_rows(
    imu_row: Optional[dict],
    nav_row: Optional[dict],
    lat0: float,
    lon0: float,
    alt0: float,
    global_coord_mode: str,
    global_transformer,
) -> Tuple[np.ndarray, np.ndarray, str]:
    coord_mode = (global_coord_mode or "").strip().lower()

    # Translation: prefer NAV lat/lon; fall back to IMU lat/lon if the IMU
    # csv itself carries position fields; else zeros.
    src_row = nav_row if (nav_row and "Latitude" in nav_row) else (
        imu_row if (imu_row and "Latitude" in imu_row) else None
    )
    if src_row is not None:
        lat = fget(src_row, "Latitude", lat0)
        lon = fget(src_row, "Longitude", lon0)
        alt = fget(src_row, "Altitude", alt0)
        if coord_mode in {"utm", "projected_global", "map_global"} and global_transformer is not None:
            x_map, y_map = global_transformer.transform(float(lon), float(lat))
            t_ego2global = np.array([x_map, y_map, alt], dtype=np.float32)
            translation_source = "projected_global"
        else:
            x_east, y_north = latlon_to_local_xy(lat, lon, lat0, lon0)
            t_ego2global = np.array([x_east, y_north, alt - alt0], dtype=np.float32)
            translation_source = "local_enu"
    else:
        t_ego2global = np.zeros(3, dtype=np.float32)
        translation_source = "zeros"

    rotation_source = "identity"
    if imu_row and all(k in imu_row for k in ["qw", "qx", "qy", "qz"]):
        q_ego2global = np.array([
            fget(imu_row, "qw", 1.0),
            fget(imu_row, "qx", 0.0),
            fget(imu_row, "qy", 0.0),
            fget(imu_row, "qz", 0.0),
        ], dtype=np.float32)
        if np.linalg.norm(q_ego2global) < 1e-8:
            q_ego2global = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            q_ego2global = q_ego2global / np.linalg.norm(q_ego2global)
        rotation_source = "imu_quaternion"
    elif nav_row and any(k in nav_row for k in ["Heading2", "Heading"]):
        q_ego2global = heading_pitch_roll_to_quat_wxyz(
            fget(nav_row, "Heading2", fget(nav_row, "Heading", 0.0)),
            fget(nav_row, "Pitch", 0.0),
            fget(nav_row, "Roll", 0.0),
        )
        rotation_source = "nav_heading_pitch_roll"
    else:
        q_ego2global = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    pose_source = f"{translation_source}+{rotation_source}"
    return t_ego2global, q_ego2global, pose_source


# --------------------------------------------------------------------------
# Extrinsics / intrinsics
# --------------------------------------------------------------------------
def euler_xyz_deg_to_R(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    return SciRot.from_euler(
        "XYZ",
        [float(roll_deg), float(pitch_deg), float(yaw_deg)],
        degrees=True,
    ).as_matrix().astype(np.float32)


def pose_sensor_to_ego_from_extr(
    extr: dict, sensor_name: str, trans_scale: float, sensor_kind: str
) -> Tuple[np.ndarray, np.ndarray]:
    if sensor_name not in extr:
        if sensor_kind == "lidar":
            return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        raise KeyError(f"{sensor_name} missing in extrinsics json")

    arr = np.asarray(extr[sensor_name], dtype=np.float32)

    if arr.shape == (4, 4):
        R = arr[:3, :3].astype(np.float32)
        t = arr[:3, 3].astype(np.float32)
        if sensor_kind == "camera":
            R_se = R.T.astype(np.float32)
            t_se = (-R.T @ t).astype(np.float32)
            return R_se, t_se
        return R, t

    flat = arr.reshape(-1)
    if flat.size != 6:
        raise ValueError(f"Unsupported extrinsic format for {sensor_name}: shape={arr.shape}")

    tx, ty, tz, roll, pitch, yaw = flat.tolist()
    t = np.array([tx, ty, tz], dtype=np.float32) * float(trans_scale)
    R = euler_xyz_deg_to_R(roll, pitch, yaw)
    return R, t


def parse_camera_intrinsics(intr: dict, raw_cam: str) -> Tuple[np.ndarray, np.ndarray]:
    if raw_cam not in intr:
        raise KeyError(f"{raw_cam} missing in intrinsics json")
    cam = intr[raw_cam]
    K = np.array(cam["camera_matrix"], dtype=np.float32).reshape(3, 3)
    dist = np.array(cam.get("distortion_coefficients", [0, 0, 0, 0, 0]), dtype=np.float32).reshape(-1)
    return K, dist


def compute_sensor2lidar(
    R_s2e_s: np.ndarray, t_s2e_s: np.ndarray,
    e2g_r_s_mat: np.ndarray, e2g_t_s: np.ndarray,
    R_l2e: np.ndarray, t_l2e: np.ndarray,
    e2g_r_mat: np.ndarray, e2g_t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mirrors mmdet3d's `obtain_sensor2top`: transform a sensor reading into the
    LIDAR_TOP frame by bridging through GLOBAL, using each sensor's own ego
    pose at its own capture timestamp -- NOT a single shared ego frame.

        sensor -> ego(sensor_time) -> global -> ego(lidar_time) -> lidar

    Args:
        R_s2e_s, t_s2e_s: sensor->ego (static calibration) for the source sensor
                           (e.g. a camera).
        e2g_r_s_mat, e2g_t_s: ego->global at the SOURCE sensor's own capture time.
        R_l2e, t_l2e: lidar->ego (static calibration) for LIDAR_TOP (reference sensor).
        e2g_r_mat, e2g_t: ego->global at the LIDAR keyframe time (the sample's
                           reference timestamp).

    Returns:
        (sensor2lidar_rotation, sensor2lidar_translation) such that
        p_lidar = p_sensor @ sensor2lidar_rotation.T + sensor2lidar_translation
    """
    inv_e2g_r_mat = np.linalg.inv(e2g_r_mat)
    inv_l2e_r_mat = np.linalg.inv(R_l2e)

    R = (R_s2e_s.T @ e2g_r_s_mat.T) @ (inv_e2g_r_mat.T @ inv_l2e_r_mat.T)
    T = (t_s2e_s @ e2g_r_s_mat.T + e2g_t_s) @ (inv_e2g_r_mat.T @ inv_l2e_r_mat.T)
    T = T - (e2g_t @ (inv_e2g_r_mat.T @ inv_l2e_r_mat.T) + t_l2e @ inv_l2e_r_mat.T)

    return R.T.astype(np.float32), T.astype(np.float32)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_root", required=True, help="Path to the log folder, e.g. .../20260711_1512_VF6_03_...")
    ap.add_argument("--intr_path", required=True)
    ap.add_argument("--extr_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--input_pred_dir", required=True)

    ap.add_argument("--lidar_subdir", default="LIDAR/LIDAR_TOP", help="Path (relative to raw_root) to the lidar frames.")
    ap.add_argument("--primary_lidar", default="LIDAR_TOP", help="Key used to look up this lidar in the extrinsics JSON.")
    ap.add_argument("--camera_subdir", default="CAMERA", help="Path (relative to raw_root) containing one folder per camera.")
    ap.add_argument("--trans_scale", type=float, default=0.001)
    ap.add_argument("--location", default="phenikaa")
    ap.add_argument("--time_tolerance_ms", type=float, default=50.0, help="Warn if nearest camera frame is farther than this.")
    ap.add_argument(
        "--no_camera_offset",
        action="store_true",
        help=(
            "Disable the per-camera shutter/trigger offset table (CAMERA_OFFSETS_MS) and match "
            "each camera's nearest image using the raw LiDAR timestamp directly. Use this to "
            "reproduce a plain nearest-timestamp selection (matches a reference script that does "
            "not apply any camera offset)."
        ),
    )
    ap.add_argument("--global_coord_mode", default="utm", choices=["utm", "projected_global", "map_global", "local_enu"])
    ap.add_argument("--global_epsg", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0, help="Limit to first N lidar frames (0 = all).")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    intr = json.loads(Path(args.intr_path).read_text())
    extr = json.loads(Path(args.extr_path).read_text())

    # --- LiDAR frames ---
    lidar_folder = raw_root / args.lidar_subdir
    if not lidar_folder.exists():
        raise FileNotFoundError(f"LiDAR folder not found: {lidar_folder}")
    lidar_index = build_time_index(lidar_folder, (".laz", ".las"))
    if not lidar_index:
        raise RuntimeError(f"No .laz/.las files found in {lidar_folder}")
    print(f"[INFO] LiDAR frames found in {lidar_folder}: {len(lidar_index)}")
    if args.no_camera_offset:
        print("[INFO] Camera sync mode: raw nearest-timestamp (no per-camera offset applied)")
    else:
        print(f"[INFO] Camera sync mode: offset-corrected using CAMERA_OFFSETS_MS={CAMERA_OFFSETS_MS}")
    if args.sample > 0:
        lidar_index = lidar_index[: args.sample]

    # --- Cameras: auto-discover folders under CAMERA/ ---
    camera_root = raw_root / args.camera_subdir
    raw_cam_names = discover_camera_folders(camera_root)
    if not raw_cam_names:
        raise RuntimeError(f"No camera folders with images found under {camera_root}")
    print(f"[INFO] Camera folders discovered: {raw_cam_names}")

    cam_indexes: Dict[str, List[Tuple[int, Path]]] = {}
    for raw_cam in raw_cam_names:
        idx = build_time_index(camera_root / raw_cam, (".jpg", ".jpeg", ".png"))
        if not idx:
            print(f"[WARN] No images found for {raw_cam}, skipping this camera.")
            continue
        cam_indexes[raw_cam] = idx
        print(f"[INFO]   {raw_cam}: {len(idx)} images")

    # --- IMU / NAV pose sources ---
    other_root = raw_root / "OTHERS"
    if not other_root.exists():
        other_root = raw_root / "OTHER"
    imu_records = load_csv_timeseries(other_root / "IMU")
    nav_records = load_csv_timeseries(other_root / "NAV")
    print(f"[INFO] IMU rows: {len(imu_records)} (from {other_root / 'IMU'})")
    print(f"[INFO] NAV rows: {len(nav_records)} (from {other_root / 'NAV'})")
    if not imu_records and not nav_records:
        print("[WARN] No IMU/NAV rows found; ego2global will be zeros/identity for all frames.")

    ref_row = nav_records[0][1] if nav_records else (imu_records[0][1] if imu_records else None)
    lat0 = fget(ref_row, "Latitude", 0.0)
    lon0 = fget(ref_row, "Longitude", 0.0)
    alt0 = fget(ref_row, "Altitude", 0.0)
    global_transformer, global_epsg, global_crs_name = build_global_transformer(
        args.global_coord_mode, lat0, lon0, int(args.global_epsg) if int(args.global_epsg) > 0 else None
    )
    print(f"[INFO] Global coord mode: {args.global_coord_mode}  CRS: {global_crs_name}")

    # --- Static lidar2ego (same across all frames) ---
    R_le, t_le = pose_sensor_to_ego_from_extr(extr, args.primary_lidar, args.trans_scale, sensor_kind="lidar")
    q_le = rotmat_to_quat_wxyz(R_le)

    # --- Pre-compute per-camera static sensor2ego + intrinsics only.
    # sensor2lidar is NOT static (it depends on ego motion between the
    # camera's own capture time and the LiDAR keyframe time), so it is
    # computed per-frame below instead of here.
    cam_static: Dict[str, dict] = {}
    cam_calib: Dict[str, dict] = {}
    for raw_cam in cam_indexes:
        if raw_cam not in extr or raw_cam not in intr:
            print(f"[WARN] {raw_cam} missing in intrinsics/extrinsics json, skipping this camera entirely.")
            continue
        R_ce, t_ce = pose_sensor_to_ego_from_extr(extr, raw_cam, args.trans_scale, sensor_kind="camera")
        q_ce = rotmat_to_quat_wxyz(R_ce)
        K, dist = parse_camera_intrinsics(intr, raw_cam)
        cam_static[raw_cam] = {
            "out_name": RAW2OUT_CAM.get(raw_cam, raw_cam),
            "sensor2ego_translation": t_ce.tolist(),
            "sensor2ego_rotation": q_ce.tolist(),
            "camera_intrinsics": f"{K.tolist()}",
            "distortion": dist.tolist(),
        }
        cam_calib[raw_cam] = {"R_ce": R_ce, "t_ce": t_ce}

    if not cam_static:
        raise RuntimeError("No cameras had both a folder of images AND calibration entries. Nothing to do.")

    # --- Build per-frame records ---
    records: List[dict] = []
    warn_sync = 0
    for lidar_ns, lidar_path in lidar_index:
        lidar_name = lidar_path.stem
        lidar_rel = f"{args.input_pred_dir}/point_clouds/{args.primary_lidar}/" + lidar_name + ".pcd"

        # Ego pose at the LiDAR keyframe timestamp -- this is the frame's
        # reference pose, and also the "lidar time" endpoint used when
        # bridging each camera's sensor2lidar transform through global.
        imu_row, _, _ = nearest_record_with_meta(imu_records, lidar_ns)
        nav_row, _, _ = nearest_record_with_meta(nav_records, lidar_ns)
        t_ego2global, q_ego2global, pose_source = derive_pose_from_rows(
            imu_row, nav_row, lat0, lon0, alt0, args.global_coord_mode, global_transformer
        )
        e2g_r_mat = quat_wxyz_to_rotmat(q_ego2global)
        e2g_t = t_ego2global.tolist()
        e2g_q = q_ego2global.tolist()

        cams: Dict[str, dict] = {}
        for raw_cam, static_info in cam_static.items():
            ideal_ns = ideal_camera_timestamp_ns(lidar_ns, raw_cam, use_offset=not args.no_camera_offset)
            cam_ns, img_path, diff_ms = nearest_by_time(cam_indexes[raw_cam], ideal_ns)
            img_path = str(img_path.relative_to(raw_root))
            img_path = f"{args.input_pred_dir}/images/" + img_path.split("CAMERA/")[-1]
            if diff_ms > args.time_tolerance_ms:
                warn_sync += 1

            # Ego pose at THIS camera's own matched image timestamp (cam_ns),
            # not the LiDAR's -- the two capture instants differ by the
            # offset table above, and the vehicle moves in that gap.
            imu_row_c, _, _ = nearest_record_with_meta(imu_records, cam_ns)
            nav_row_c, _, _ = nearest_record_with_meta(nav_records, cam_ns)
            t_ego2global_c, q_ego2global_c, _ = derive_pose_from_rows(
                imu_row_c, nav_row_c, lat0, lon0, alt0, args.global_coord_mode, global_transformer
            )
            e2g_r_s_mat = quat_wxyz_to_rotmat(q_ego2global_c)

            R_ce = cam_calib[raw_cam]["R_ce"]
            t_ce = cam_calib[raw_cam]["t_ce"]
            R_c2l, t_c2l = compute_sensor2lidar(
                R_ce, t_ce,
                e2g_r_s_mat, t_ego2global_c,
                R_le, t_le,
                e2g_r_mat, t_ego2global,
            )

            out_name = static_info["out_name"]
            cams[out_name] = {
                "data_path": img_path,
                "type": out_name,
                "sensor2ego_translation": static_info["sensor2ego_translation"],
                "sensor2ego_rotation": static_info["sensor2ego_rotation"],
                # each camera's OWN ego2global at its own capture time
                # (matches nuScenes' per-sample_data convention).
                "ego2global_translation": t_ego2global_c.tolist(),
                "ego2global_rotation": q_ego2global_c.tolist(),
                "sensor2lidar_rotation": f"{R_c2l.tolist()}",
                "sensor2lidar_translation": f"{t_c2l.tolist()}",
                "camera_intrinsics": static_info["camera_intrinsics"],
                "distortion": static_info["distortion"],
                "sync_diff_ms": round(diff_ms, 3),
            }

        records.append({
            "lidar_path": lidar_rel,
            "timestamp_ns": int(lidar_ns),
            "cams": cams,
            "lidar2ego_translation": t_le.tolist(),
            "lidar2ego_rotation": q_le.tolist(),
            # frame-level reference pose: LIDAR_TOP's own ego2global.
            "ego2global_translation": e2g_t,
            "ego2global_rotation": e2g_q,
            "location": args.location,
            "pose_source": pose_source,
        })

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=4)

    print("==== DONE ====")
    print(f"Wrote {len(records)} frames to {out_path}")
    print(f"Camera sync warnings (> {args.time_tolerance_ms} ms): {warn_sync}")


if __name__ == "__main__":
    main()