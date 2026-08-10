#!/usr/bin/env python3
"""
Merge multi-LiDAR point clouds into the ego/vehicle frame.

Concatenation logic matches `05_pointcloud_concat_and_visualize.py`:
  - Each sensor's points are transformed into the ego frame via
    T @ [x,y,z,1]^T, where T is that sensor's 4x4 sensor->ego extrinsic
    matrix (loaded from --extr; supports 4x4, flattened 16-value, and
    legacy 6-value [tx,ty,tz,roll,pitch,yaw] formats). This is applied to
    EVERY sensor, including LIDAR_TOP -- there is no implicit "LIDAR_TOP
    frame == ego frame" assumption anymore.
  - LIDAR_E_* (edge) sensors additionally get a cylindrical ROI filter
    (radius CYLINDER_RADIUS around x_center=0.910, z in [MIN_Z_SIDE, MAX_Z_SIDE])
    to drop vehicle self-returns / near-field clutter, evaluated in the
    already-transformed ego-frame coordinates. LIDAR_TOP is not filtered.
  - --origin optionally applies an ADDITIONAL fixed translation on top of
    the true extrinsic-derived ego frame (e.g. to re-center at the vehicle
    center or rear axle). Default "lidar_top" applies no extra shift.

Output per merged frame:
  - <out_dir_pcd>/<token>.pcd   xyz-only binary PCD (matches the strict
                                 LoadPointsFromFile._load_pcd_xyz_as_dim5 format)
  - <out_dir_bin>/<token>.bin   KITTI-style float32 x,y,z,intensity
  - <out_dir_laz>/<token>.laz   full LAS/LAZ with intensity + sensor_id extra dim
                                 (only written if --out_dir_laz is given)

Optional: --visualize opens an interactive Open3D viewer (same controls as
05_pointcloud_concat_and_visualize.py: per-layer toggle, color picker, point
size slider) for ONE frame, built from the exact same transformed/filtered
per-sensor arrays used for merging.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import laspy
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import os 

# -------------------------
# Config
# -------------------------
MAX_TIME_DIFF = 0.07
# Used only for legacy 6-value [tx, ty, tz, roll, pitch, yaw] extrinsics
# (translation is stored in mm in that format, hence the 0.001 scale).
TRANS_SCALE = 0.001
TARGET_SCALES = np.array([0.01, 0.01, 0.01], dtype=np.float64)
TARGET_OFFSETS = np.array([0.0, 0.0, 0.0], dtype=np.float64)
ANCHOR_LIDAR = "LIDAR_TOP"

# Cylindrical ROI filter for LIDAR_E_* sensors -- copied verbatim from
# 05_pointcloud_concat_and_visualize.py so merged output matches the
# reference visualization exactly.
CYLINDER_RADIUS = 6.0
CYLINDER_X_CENTER = 0.910
MIN_Z_SIDE = -0.5
MAX_Z_SIDE = 2.0


LIDAR_IDX = {
    "LIDAR_E_F":  0,
    "LIDAR_E_L":  1,
    "LIDAR_E_R":  2,
    "LIDAR_E_B":  3,
    "LIDAR_AT_F": 4,
    "LIDAR_TOP":  5,
}
ALLOWED_LIDARS = set(LIDAR_IDX.keys())

# Sensor colors reused for --visualize (matches the reference viewer).
try:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
    _HAS_OPEN3D = True
    SENSOR_COLORS = {
        "LIDAR_E_F": gui.Color(0.0, 0.8, 0.0),
        "LIDAR_E_L": gui.Color(0.0, 0.8, 0.0),
        "LIDAR_E_R": gui.Color(0.0, 0.8, 0.0),
        "LIDAR_E_B": gui.Color(0.0, 0.8, 0.0),
        "LIDAR_TOP": gui.Color(0.0, 0.4, 1.0),
        "LIDAR_AT_F": gui.Color(1.0, 0.0, 0.0),
    }
except ImportError:
    _HAS_OPEN3D = False
    SENSOR_COLORS = {}

DOWNSAMPLE_VOXEL_SIZE = 0.05
POINT_SIZE_DEFAULT = 1.0
BACKGROUND_COLOR = [0.9, 0.9, 0.9]
SHOW_COORDINATE_FRAME = True

# -------------------------
# Vehicle-center origin offset
# -------------------------
CENTER_ORIGIN_OFFSET = np.array(
    [-1.403, 0, 0],
    dtype=np.float64,
)


# -------------------------
# Helper functions
# -------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_extrinsics(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def get_transform_matrix(extrinsics: Dict, sensor_name: str) -> np.ndarray:
    """
    Return a sensor-to-ego 4x4 homogeneous transform matrix for sensor_name.

    Accepts three JSON formats:
      - a 4x4 nested list/array
      - 16 flattened values
      - legacy 6 values [tx, ty, tz, roll, pitch, yaw] (translation in mm,
        rotation in degrees, applied as intrinsic XYZ Euler angles)
    """
    if sensor_name not in extrinsics:
        raise KeyError(f"Missing extrinsic for sensor: {sensor_name}")

    arr = np.asarray(extrinsics[sensor_name], dtype=np.float64)

    if arr.shape == (4, 4):
        return arr

    flat = arr.reshape(-1)
    if flat.size == 16:
        return flat.reshape(4, 4)

    if flat.size == 6:
        tx, ty, tz, roll, pitch, yaw = flat.tolist()
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = R.from_euler(
            "xyz", [roll, pitch, yaw], degrees=True
        ).as_matrix()
        transform[:3, 3] = np.array([tx, ty, tz], dtype=np.float64) * TRANS_SCALE
        return transform

    raise ValueError(
        f"Unsupported extrinsic format for {sensor_name}: shape={arr.shape}. "
        "Expected a 4x4 matrix, 16 flattened values, or 6 legacy values."
    )


def ts_ns_from_stem(stem: str) -> int:
    sec, nsec = stem.split("-")
    # Zero-pad the fractional part like the reference parser -- a filename
    # with a truncated-precision fraction (e.g. "099" instead of "099000000")
    # would otherwise be silently off by orders of magnitude.
    nsec = nsec.ljust(9, "0")[:9]
    return int(sec) * 1_000_000_000 + int(nsec)


def build_lidar_index(folder: Path) -> List[Tuple[int, Path]]:
    items: List[Tuple[int, Path]] = []
    for p in folder.glob("*.laz"):
        try:
            t = ts_ns_from_stem(p.stem)
        except Exception:
            continue
        items.append((t, p))
    items.sort(key=lambda x: x[0])
    return items


def nearest_by_time(index: List[Tuple[int, Path]], t_ns: int) -> Tuple[int, Path]:
    lo, hi = 0, len(index) - 1
    if hi < 0:
        raise RuntimeError("Empty index")
    if t_ns <= index[0][0]:
        return index[0]
    if t_ns >= index[-1][0]:
        return index[-1]
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if index[mid][0] < t_ns:
            lo = mid
        else:
            hi = mid
    return index[lo] if abs(index[lo][0] - t_ns) <= abs(index[hi][0] - t_ns) else index[hi]

def transform_points(
    transform: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    is_side: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform raw sensor-frame points into the ego frame via T @ [x,y,z,1]^T,
    then (for LIDAR_E_* sensors) apply the cylindrical ROI filter in the
    resulting ego-frame coordinates.

    Args:
        transform: (4, 4) sensor-to-ego homogeneous matrix.
        x, y, z: raw points in the sensor's own local frame.
        is_side: True for LIDAR_E_* sensors -> apply the cylindrical filter.
                 False (e.g. LIDAR_TOP) -> keep all points.

    Returns:
        xyz: Transformed (and possibly filtered) points, shape (N, 3), in
             the ego frame.
        keep: Boolean mask applied to the original points.
    """
    homogeneous = np.vstack(
        (
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(z, dtype=np.float64),
            np.ones_like(x, dtype=np.float64),
        )
    )
    transformed = transform @ homogeneous
    xt, yt, zt = transformed[0], transformed[1], transformed[2]

    keep = np.ones(xt.shape[0], dtype=bool)

    if is_side:
        x_shift = xt - CYLINDER_X_CENTER
        keep = (
            (x_shift**2 + yt**2 <= CYLINDER_RADIUS**2)
            & (zt >= MIN_Z_SIDE)
            & (zt <= MAX_Z_SIDE)
        )

    xyz = np.column_stack((xt[keep], yt[keep], zt[keep]))

    return xyz, keep


def transform_origin(
    xyz: np.ndarray,
    origin: str,
) -> np.ndarray:
    """
    Apply an OPTIONAL additional translation on top of the true ego frame
    (points already transformed via the sensor's extrinsic).

    Args:
        xyz: (N, 3) point cloud, already in the ego frame.
        origin:
            "lidar_top" -> no extra shift; use the ego frame as defined
                           directly by the extrinsics, i.e. the
                           back/rear-axle origin (default).
            "back"      -> alias for "lidar_top"/default. The extrinsics'
                           ego frame IS the back/rear-axle origin already,
                           so this is a no-op kept only for CLI backward
                           compatibility.
            "center"    -> additionally shift by CENTER_ORIGIN_OFFSET
                           meters (back(ego) -> vehicle center).

    Returns:
        (N, 3) point cloud.
    """
    if origin in ("lidar_top", "back"):
        return xyz

    if origin == "center":
        return xyz + CENTER_ORIGIN_OFFSET

    raise ValueError(
        f"Unknown origin '{origin}'. "
        "Expected 'lidar_top' or 'center'."
    )


def ensure_sensor_id_dim(las: laspy.LasData) -> None:
    if "sensor_id" in set(las.point_format.dimension_names):
        return
    las.add_extra_dim(
        laspy.ExtraBytesParams(name="sensor_id", type=np.uint8, description="LiDAR sensor index")
    )


# -------------------------
# Writers
# -------------------------
def write_pcd(filename: Path, points: np.ndarray):
    """
    Write an xyz-only binary PCD.

    IMPORTANT: this must match the strict format that the downstream
    `LoadPointsFromFile._load_pcd_xyz_as_dim5` loader requires:

        FIELDS x y z
        SIZE   4 4 4
        TYPE   F F F
        COUNT  1 1 1
        DATA   binary

    That loader hard-fails (ValueError) on anything else -- it does not read
    intensity or sensor_id from the .pcd at all, it only ever reads xyz and
    pads to 5 dims with two zero columns. So intensity / sensor_id are
    intentionally NOT written here; they still round-trip through the
    companion .bin (and now .laz) files.

    Input:
        points: (N,5) columns = x y z intensity sensor_id
                (only the first 3 columns, x y z, are used)
    """
    ensure_dir(filename.parent)

    xyz = np.ascontiguousarray(points[:, :3], dtype=np.float32)
    n = xyz.shape[0]

    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {n}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {n}
DATA binary
"""

    with open(filename, "wb") as f:
        f.write(header.encode("ascii"))
        xyz.tofile(f)


def write_bin(filename: Path, points: np.ndarray):
    """
    Save KITTI-style .bin file.

    Input:
        points: (N,5)  x y z intensity sensor_id

    Output:
        float32 binary: x y z intensity
    """
    ensure_dir(filename.parent)
    cloud = points[:, :4].astype(np.float32)
    cloud.tofile(filename)


def write_laz(filename: Path, points: np.ndarray, template_las: laspy.LasData):
    """
    Save a full LAS/LAZ file with intensity + sensor_id as an extra dim.

    Input:
        points: (N,5)  x y z intensity sensor_id
        template_las: any one of the source LasData objects, used to inherit
                       point_format / file_version.
    """
    ensure_dir(filename.parent)

    out = laspy.create(point_format=template_las.header.point_format, file_version=template_las.header.version)
    out.header.scales = TARGET_SCALES
    out.header.offsets = TARGET_OFFSETS
    ensure_sensor_id_dim(out)

    n = points.shape[0]
    out.points = laspy.ScaleAwarePointRecord.zeros(
        n, point_format=out.header.point_format, scales=TARGET_SCALES, offsets=TARGET_OFFSETS
    )
    out.x = points[:, 0]
    out.y = points[:, 1]
    out.z = points[:, 2]
    out.intensity = np.clip(points[:, 3], 0, 65535).astype(np.uint16)
    out.sensor_id = points[:, 4].astype(np.uint8)

    out.write(str(filename))


# -------------------------
# Merge one timestamp
# -------------------------
def merge_one_timestamp(
    t_ns: int,
    lidar_indexes: Dict[str, List[Tuple[int, Path]]],
    transform_matrices: Dict[str, np.ndarray],
    out_pcd: Path,
    out_bin: Path,
    out_laz: Optional[Path] = None,
    max_time_diff_sec: float = MAX_TIME_DIFF,
    collect_sensor_points: bool = False,
    origin: str = "lidar_top",
) -> Tuple[bool, Dict[str, np.ndarray]]:
    """
    Returns (wrote_ok, sensor_points) where sensor_points maps
    lidar_name -> transformed/filtered xyz (only populated if
    collect_sensor_points=True, e.g. for --visualize).

    Each sensor's raw points are converted into the ego frame via its own
    sensor-to-ego extrinsic (transform_matrices[lidar_name]) before the
    cylindrical ROI filter (LIDAR_E_* only) and the optional --origin shift
    are applied. LIDAR_TOP is transformed by its own extrinsic just like
    every other sensor -- it is no longer treated as the ego frame itself.
    """
    merged: List[np.ndarray] = []
    sensor_points: Dict[str, np.ndarray] = {}
    template_las = None

    for lidar_name, index in lidar_indexes.items():
        if lidar_name not in LIDAR_IDX or len(index) == 0:
            continue

        transform = transform_matrices.get(lidar_name)
        if transform is None:
            print(f"[WARN] Skip {lidar_name}: missing extrinsic")
            continue

        t_best, best_file = nearest_by_time(index, t_ns)
        dt = abs(t_best - t_ns) / 1e9
        if dt > max_time_diff_sec:
            continue

        las = laspy.read(best_file)
        if template_las is None:
            template_las = las

        x = np.asarray(las.x, dtype=np.float64)
        y = np.asarray(las.y, dtype=np.float64)
        z = np.asarray(las.z, dtype=np.float64)

        # Sensor-frame -> ego-frame via T @ [x, y, z, 1]^T, then (for
        # LIDAR_E_* only) the cylindrical ROI filter in ego coordinates.
        xyz, keep = transform_points(
            transform, x, y, z, is_side=lidar_name.startswith("LIDAR_E_")
        )

        if xyz.size == 0:
            continue

        # --------------------------------------------------
        # Optional additional shift on top of the true ego frame,
        # e.g. to re-center at the vehicle center or rear axle.
        # "lidar_top" (default) applies no extra shift.
        # --------------------------------------------------
        xyz = transform_origin(xyz, origin)

        intensity = np.asarray(las.intensity, dtype=np.float32)[keep]
        sensor_id = np.full(xyz.shape[0], LIDAR_IDX[lidar_name], dtype=np.uint8)

        merged.append(np.column_stack((xyz, intensity, sensor_id)))
        if collect_sensor_points:
            sensor_points[lidar_name] = xyz

    if len(merged) == 0:
        return False, sensor_points

    merged_arr = np.concatenate(merged, axis=0)

    write_pcd(out_pcd, merged_arr)
    write_bin(out_bin, merged_arr)
    if out_laz is not None:
        write_laz(out_laz, merged_arr, template_las)

    return True, sensor_points


# -------------------------
# Visualization (reused from 05_pointcloud_concat_and_visualize.py)
# -------------------------
def clone_color(c) -> "gui.Color":
    return gui.Color(c.red, c.green, c.blue, c.alpha)


def make_pcd(xyz: np.ndarray, downsample: bool = False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if downsample and DOWNSAMPLE_VOXEL_SIZE > 0:
        pcd = pcd.voxel_down_sample(voxel_size=DOWNSAMPLE_VOXEL_SIZE)
    return pcd


class LidarViewer:
    PANEL_WIDTH_EM = 17

    def __init__(self, sensor_points: Dict[str, np.ndarray], title: str):
        self.sensor_points = sensor_points
        self.sensor_colors = {
            name: clone_color(SENSOR_COLORS.get(name, gui.Color(0.7, 0.7, 0.7)))
            for name in sensor_points
        }
        self.color_buttons: Dict[str, "gui.Button"] = {}
        self.point_size = POINT_SIZE_DEFAULT
        self.sensor_names = list(sensor_points)

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(title, 1600, 1000)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background(BACKGROUND_COLOR + [1.0])

        for name in self.sensor_names:
            self._add_geometry(name)

        if SHOW_COORDINATE_FRAME:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"
            self.scene.scene.add_geometry("__frame__", frame, mat)

        self._build_panel()
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.panel)
        self.window.add_child(self.scene)
        self.scene.setup_camera(60.0, self.scene.scene.bounding_box, self.scene.scene.bounding_box.get_center())

    def _material(self, color) -> "rendering.MaterialRecord":
        mat = rendering.MaterialRecord()
        mat.shader = "unlitSolidColor"
        mat.base_color = [color.red, color.green, color.blue, 1.0]
        mat.point_size = self.point_size
        return mat

    def _add_geometry(self, name: str):
        pcd = make_pcd(self.sensor_points[name], downsample=True)
        self.scene.scene.add_geometry(name, pcd, self._material(self.sensor_colors[name]))

    def _update_all_materials(self):
        for name in self.sensor_names:
            self.scene.scene.modify_geometry_material(name, self._material(self.sensor_colors[name]))
        self.scene.force_redraw()

    def _build_panel(self):
        em = self.window.theme.font_size
        self.panel = gui.Vert(0, gui.Margins(em * 0.6, em * 0.6, em * 0.6, em * 0.6))
        self.panel.add_child(gui.Label("LiDAR"))

        slider = gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(0.5, 8.0)
        slider.double_value = self.point_size
        slider.set_on_value_changed(self._on_point_size)
        self.panel.add_child(slider)

        for name in self.sensor_names:
            row = gui.Horiz(int(em * 0.4), gui.Margins(0, 0, 0, int(em * 0.4)))

            cb = gui.Checkbox(name)
            cb.checked = True
            cb.set_on_checked(lambda checked, n=name: self.scene.scene.show_geometry(n, checked))

            btn = gui.Button("")
            btn.background_color = self.sensor_colors[name]
            btn.tooltip = "Pick color"
            btn.set_on_clicked(lambda n=name: self._open_color_dialog(n))
            self.color_buttons[name] = btn

            row.add_child(cb)
            row.add_stretch()
            row.add_child(btn)
            self.panel.add_child(row)

    def _open_color_dialog(self, name: str):
        em = self.window.theme.font_size
        dlg = gui.Dialog(name)
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))

        picker = gui.ColorEdit()
        picker.color_value = clone_color(self.sensor_colors[name])
        dlg_layout.add_child(picker)

        btns = gui.Horiz()
        ok = gui.Button("OK")
        cancel = gui.Button("Cancel")
        btns.add_stretch()
        btns.add_child(cancel)
        btns.add_child(ok)
        dlg_layout.add_child(btns)

        dlg.add_child(dlg_layout)

        ok.set_on_clicked(lambda n=name, p=picker: self._apply_color(n, p.color_value))
        cancel.set_on_clicked(self.window.close_dialog)
        self.window.show_dialog(dlg)

    def _apply_color(self, name: str, color):
        color = clone_color(color)
        self.sensor_colors[name] = color
        self.color_buttons[name].background_color = color
        self.scene.scene.modify_geometry_material(name, self._material(color))
        self.scene.force_redraw()
        self.window.close_dialog()

    def _on_point_size(self, value: float):
        self.point_size = value
        self._update_all_materials()

    def _on_layout(self, ctx):
        r = self.window.content_rect
        w = int(self.PANEL_WIDTH_EM * ctx.theme.font_size)
        self.panel.frame = gui.Rect(r.x, r.y, w, r.height)
        self.scene.frame = gui.Rect(r.x + w, r.y, r.width - w, r.height)

    def run(self):
        gui.Application.instance.run()


# -------------------------
# Main
# -------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--extr", default="EXTRINSICS_FILE_PATH")
    ap.add_argument("--lidar_root", default="LIDAR_DATA_PATH")
    ap.add_argument("--out_dir_pcd", default="OUT_DIR")
    ap.add_argument("--out_dir_bin", default="OUT_DIR_BIN")
    ap.add_argument("--out_dir_laz", default="OUT_DIR_LAZ")
    ap.add_argument("--anchor", default=ANCHOR_LIDAR)
    ap.add_argument(
        "--origin",
        choices=["lidar_top", "center", "back"],
        default="lidar_top",
        help=(
            "Additional shift applied ON TOP of the true extrinsic-derived "
            "ego frame. 'lidar_top' (default): no extra shift -- the ego "
            "frame IS the back/rear-axle origin, as defined by the "
            "extrinsics. 'back': no-op alias for the default, kept for "
            "backward compatibility. 'center': shifts by "
            "CENTER_ORIGIN_OFFSET meters (back-ego -> vehicle center)"
        ),
    )
    ap.add_argument("--max_dt", type=float, default=MAX_TIME_DIFF)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--visualize",
        action="store_true",
        help="Open the interactive Open3D viewer for ONE frame instead of/after batch export.",
    )
    ap.add_argument(
        "--visualize_ts",
        default="",
        help="Timestamp stem (e.g. 1783757599-099982977) to visualize. Defaults to the first anchor frame.",
    )
    ap.add_argument(
        "--visualize_only",
        action="store_true",
        help="Skip the batch export loop entirely and only show the viewer.",
    )
    args = ap.parse_args()

    extr = load_extrinsics(args.extr)
    lidar_root = Path(args.lidar_root)
    out_dir_pcd = Path(args.out_dir_pcd)
    out_dir_bin = Path(args.out_dir_bin)
    out_dir_laz = Path(args.out_dir_laz) if args.out_dir_laz else None
    ensure_dir(out_dir_pcd)
    ensure_dir(out_dir_bin)
    if out_dir_laz is not None:
        ensure_dir(out_dir_laz)

    lidar_indexes: Dict[str, List[Tuple[int, Path]]] = {}
    for p in sorted(lidar_root.glob("LIDAR*")):
        if not p.is_dir():
            continue
        lidar_name = p.name
        if lidar_name not in ALLOWED_LIDARS:
            continue
        idx = build_lidar_index(p)
        if len(idx) > 0:
            lidar_indexes[lidar_name] = idx

    transform_matrices: Dict[str, np.ndarray] = {}
    for lidar_name in list(lidar_indexes.keys()):
        try:
            transform_matrices[lidar_name] = get_transform_matrix(extr, lidar_name)
        except (KeyError, ValueError) as exc:
            print(f"[WARN] {exc}")

    print("Lidars in use:", list(lidar_indexes.keys()))
    print("Lidars with valid extrinsic:", list(transform_matrices.keys()))
    print("Output origin:", args.origin)

    if args.origin == "center":
        print(
            "Center-origin offset (back-ego -> center, PLACEHOLDER, "
            "recalculate before trusting this output): "
            f"x={CENTER_ORIGIN_OFFSET[0]:+.3f} m, "
            f"y={CENTER_ORIGIN_OFFSET[1]:+.3f} m, "
            f"z={CENTER_ORIGIN_OFFSET[2]:+.3f} m"
        )
    elif args.origin == "back":
        print(
            "'back' is a no-op: the extrinsics' ego frame is already the "
            "back/rear-axle origin by default."
        )
    
    if args.anchor not in lidar_indexes:
        raise RuntimeError(f"Anchor lidar {args.anchor} has no data. Available: {list(lidar_indexes.keys())}")

    if args.anchor not in transform_matrices:
        print(f"[WARN] Anchor {args.anchor} has no valid extrinsic; its points will be dropped from every merged frame.")

    anchor_index = lidar_indexes[args.anchor]
    print("len(anchor_index):", len(anchor_index))

    if args.limit > 0:
        anchor_index = anchor_index[: args.limit]

    if not args.visualize_only:
        ok = 0
        miss = 0
        for t_ns, p in tqdm(anchor_index, desc=f"Merging (anchor={args.anchor})"):
            token = p.stem
            out_pcd = out_dir_pcd / f"{token}.pcd"
            out_bin = out_dir_bin / f"{token}.bin"
            out_laz = (out_dir_laz / f"{token}.laz") if out_dir_laz is not None else None

            wrote, _ = merge_one_timestamp(
                t_ns,
                lidar_indexes,
                transform_matrices,
                out_pcd,
                out_bin,
                out_laz,
                max_time_diff_sec=args.max_dt,
                origin=args.origin,
            )
            if wrote:
                ok += 1
            else:
                miss += 1

        print("==== DONE ====")
        print("Output .pcd:", out_dir_pcd)
        print("Output .bin:", out_dir_bin)
        if out_dir_laz is not None:
            print("Output .laz:", out_dir_laz)
        print("Wrote:", ok, "frames")
        print("No data:", miss, "frames (no lidar matched)")

    if args.visualize:
        if not _HAS_OPEN3D:
            print("[WARN] --visualize requested but open3d is not installed. Skipping.")
            return

        if args.visualize_ts:
            target_ns = ts_ns_from_stem(args.visualize_ts)
        else:
            target_ns, _ = anchor_index[0]

        print(f"[VIS] Visualizing timestamp: {target_ns}")
        _, sensor_points = merge_one_timestamp(
            target_ns,
            lidar_indexes,
            transform_matrices,
            out_pcd=(
                Path("/dev/null.pcd")
                if not args.visualize_only
                else out_dir_pcd / "_viz_tmp.pcd"
            ),
            out_bin=(
                Path("/dev/null.bin")
                if not args.visualize_only
                else out_dir_bin / "_viz_tmp.bin"
            ),
            out_laz=None,
            max_time_diff_sec=args.max_dt,
            collect_sensor_points=True,
            origin=args.origin,
        )
        os.remove(out_dir_pcd / "_viz_tmp.pcd") if (out_dir_pcd / "_viz_tmp.pcd").exists() else None
        os.remove(out_dir_pcd / "_viz_tmp.bin") if (out_dir_pcd / "_viz_tmp.bin").exists() else None

        if not sensor_points:
            print("No points to show.")
            return

        total = sum(len(v) for v in sensor_points.values())
        print(f"{len(sensor_points)} layers, {total:,} points")
        print("Left panel: toggle layers, color button, point size slider.")
        LidarViewer(sensor_points, f"LiDAR — {target_ns}").run()


if __name__ == "__main__":
    main()