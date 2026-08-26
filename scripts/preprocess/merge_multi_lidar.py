#!/usr/bin/env python3
"""Merge multi-LiDAR point clouds into a common ego/vehicle frame."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import laspy
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

logger = logging.getLogger("merge_pointclouds")

# -------------------------
# Config
# -------------------------
MAX_TIME_DIFF = 0.07
TRANS_SCALE = 0.001  # legacy 6-value extrinsics store translation in mm
TARGET_SCALES = np.array([0.01, 0.01, 0.01], dtype=np.float64)
TARGET_OFFSETS = np.array([0.0, 0.0, 0.0], dtype=np.float64)
ANCHOR_LIDAR = "LIDAR_TOP"

# Cylindrical ROI filter for LIDAR_E_* (edge) sensors only.
CYLINDER_RADIUS = 6.0
CYLINDER_X_CENTER = 0.910
MIN_Z_SIDE = -0.5
MAX_Z_SIDE = 2.0

LIDAR_IDX = {
    "LIDAR_E_F": 0, "LIDAR_E_L": 1, "LIDAR_E_R": 2, "LIDAR_E_B": 3,
    "LIDAR_AT_F": 4, "LIDAR_TOP": 5,
}
ALLOWED_LIDARS = set(LIDAR_IDX.keys())

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

CENTER_ORIGIN_OFFSET = np.array([-1.403, 0, 0], dtype=np.float64)  # native-ego -> vehicle center


# -------------------------
# Extrinsics
# -------------------------
def load_extrinsics(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def get_transform_matrix(extrinsics: Dict, sensor_name: str) -> np.ndarray:
    """Return sensor->ego 4x4 for sensor_name. Accepts 4x4, flat-16, or legacy [tx,ty,tz,roll,pitch,yaw]."""
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
        transform[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
        transform[:3, 3] = np.array([tx, ty, tz], dtype=np.float64) * TRANS_SCALE
        return transform

    raise ValueError(f"Unsupported extrinsic format for {sensor_name}: shape={arr.shape}")


def get_origin_shift_matrix(origin: str, anchor_to_oldego: np.ndarray, is_extrinsic=False) -> np.ndarray:
    """
    Return the transform that changes coordinates from the native ego frame
    into the requested output-origin frame.

    ``transform_points()`` first converts every raw LiDAR point with
    ``sensor -> ego``. Therefore:

      - lidar_top: ego -> LIDAR_TOP = inv(LIDAR_TOP -> ego)
      - back:      ego -> vehicle-ego = identity
      - center:    ego -> vehicle-center = CENTER_ORIGIN_OFFSET

    The same output-frame change must be applied to the saved sensor
    extrinsics. In other words, when ``--origin back`` is selected, the
    saved LIDAR_* extrinsics become sensor->ego instead of sensor->LIDAR_TOP.
    """
    T = np.eye(4, dtype=np.float64)

    if origin == "lidar_top":
        if is_extrinsic:
            # Existing matrices are sensor -> ego. Keep the same output
            # frame conversion used by transform_origin(): ego -> LIDAR_TOP.
            return np.eye(4, dtype=np.float64)
        return np.linalg.inv(anchor_to_oldego)

    if origin == "back":
        # ``back`` is the vehicle ego frame. Point clouds are already in ego
        # after transform_points(), so no additional point transform is needed.
        # Likewise, sensor->ego extrinsics need no additional shift.
        return T

    if origin == "center":
        T[:3, 3] = CENTER_ORIGIN_OFFSET
        return T

    raise ValueError(f"Unknown origin '{origin}'")


def transform_origin(xyz: np.ndarray, origin: str, anchor_to_oldego: np.ndarray) -> np.ndarray:
    T = get_origin_shift_matrix(origin, anchor_to_oldego)
    if np.allclose(T, np.eye(4)):
        return xyz
    homogeneous = np.vstack((xyz[:, 0], xyz[:, 1], xyz[:, 2], np.ones(xyz.shape[0])))
    return (T @ homogeneous)[:3].T


def shift_extrinsics(transform_matrices: Dict[str, np.ndarray], origin: str, anchor_lidar: str) -> Dict[str, np.ndarray]:
    """Re-derive sensor->origin extrinsics consistent with transform_origin()."""
    if anchor_lidar not in transform_matrices:
        raise KeyError(f"Anchor lidar '{anchor_lidar}' has no extrinsic")
    T_shift = get_origin_shift_matrix(origin, transform_matrices[anchor_lidar], is_extrinsic=True)
    return {name: T_shift @ mat for name, mat in transform_matrices.items()}


def save_extrinsics(path: Path, extrinsics: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({name: mat.tolist() for name, mat in extrinsics.items()}, f, indent=2)


# -------------------------
# Point cloud transforms
# -------------------------
def transform_points(
    transform: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, is_side: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform raw sensor-frame points to ego frame; apply cylindrical ROI filter if is_side."""
    homogeneous = np.vstack((x, y, z, np.ones_like(x)))
    xt, yt, zt = (transform @ homogeneous)[:3]

    keep = np.ones(xt.shape[0], dtype=bool)
    if is_side:
        x_shift = xt - CYLINDER_X_CENTER
        keep = (x_shift**2 + yt**2 <= CYLINDER_RADIUS**2) & (zt >= MIN_Z_SIDE) & (zt <= MAX_Z_SIDE)

    return np.column_stack((xt[keep], yt[keep], zt[keep])), keep


# -------------------------
# Timestamp indexing
# -------------------------
def ts_ns_from_stem(stem: str) -> int:
    sec, nsec = stem.split("-")
    nsec = nsec.ljust(9, "0")[:9]
    return int(sec) * 1_000_000_000 + int(nsec)


def build_lidar_index(folder: Path) -> List[Tuple[int, Path]]:
    items = []
    for p in folder.glob("*.laz"):
        try:
            items.append((ts_ns_from_stem(p.stem), p))
        except Exception:
            continue
    items.sort(key=lambda x: x[0])
    return items


def nearest_by_time(index: List[Tuple[int, Path]], t_ns: int) -> Tuple[int, Path]:
    lo, hi = 0, len(index) - 1
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


# -------------------------
# Writers
# -------------------------
def write_pcd(filename: Path, points: np.ndarray) -> None:
    """xyz-only binary PCD (must match the strict LoadPointsFromFile._load_pcd_xyz_as_dim5 format)."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.ascontiguousarray(points[:, :3], dtype=np.float32)
    n = xyz.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\n"
        f"SIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {n}\nHEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n"
    )
    with open(filename, "wb") as f:
        f.write(header.encode("ascii"))
        xyz.tofile(f)


def write_bin(filename: Path, points: np.ndarray) -> None:
    """KITTI-style float32 x,y,z,intensity."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    points[:, :4].astype(np.float32).tofile(filename)


def write_laz(filename: Path, points: np.ndarray, template_las: laspy.LasData) -> None:
    """Full LAS/LAZ with intensity + sensor_id extra dim."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    out = laspy.create(point_format=template_las.header.point_format, file_version=template_las.header.version)
    out.header.scales = TARGET_SCALES
    out.header.offsets = TARGET_OFFSETS
    if "sensor_id" not in set(out.point_format.dimension_names):
        out.add_extra_dim(laspy.ExtraBytesParams(name="sensor_id", type=np.uint8))

    n = points.shape[0]
    out.points = laspy.ScaleAwarePointRecord.zeros(n, point_format=out.header.point_format, scales=TARGET_SCALES, offsets=TARGET_OFFSETS)
    out.x, out.y, out.z = points[:, 0], points[:, 1], points[:, 2]
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
    out_pcd: Optional[Path],
    out_bin: Optional[Path],
    out_laz: Optional[Path] = None,
    max_time_diff_sec: float = MAX_TIME_DIFF,
    collect_sensor_points: bool = False,
    origin: str = "lidar_top",
    anchor_lidar: str = ANCHOR_LIDAR,
) -> Tuple[bool, Dict[str, np.ndarray]]:
    """Merge each sensor's nearest-in-time frame into one point cloud at t_ns. out_pcd/out_bin=None to skip writing."""
    if anchor_lidar not in transform_matrices:
        raise KeyError(f"Anchor lidar '{anchor_lidar}' has no extrinsic")
    anchor_to_oldego = transform_matrices[anchor_lidar]

    merged: List[np.ndarray] = []
    sensor_points: Dict[str, np.ndarray] = {}
    template_las = None

    for lidar_name, index in lidar_indexes.items():
        if lidar_name not in LIDAR_IDX or not index:
            continue
        transform = transform_matrices.get(lidar_name)
        if transform is None:
            logger.warning("Skip %s: missing extrinsic", lidar_name)
            continue

        t_best, best_file = nearest_by_time(index, t_ns)
        if abs(t_best - t_ns) / 1e9 > max_time_diff_sec:
            continue

        las = laspy.read(best_file)
        template_las = template_las or las

        x, y, z = (np.asarray(getattr(las, a), dtype=np.float64) for a in "xyz")
        xyz, keep = transform_points(transform, x, y, z, is_side=lidar_name.startswith("LIDAR_E_"))
        if xyz.shape[0] == 0:
            continue
        xyz = transform_origin(xyz, origin, anchor_to_oldego)

        intensity = np.asarray(las.intensity, dtype=np.float32)[keep]
        sensor_id = np.full(xyz.shape[0], LIDAR_IDX[lidar_name], dtype=np.uint8)
        merged.append(np.column_stack((xyz, intensity, sensor_id)))
        if collect_sensor_points:
            sensor_points[lidar_name] = xyz

    if not merged:
        return False, sensor_points

    merged_arr = np.concatenate(merged, axis=0)
    if out_pcd is not None:
        write_pcd(out_pcd, merged_arr)
    if out_bin is not None:
        write_bin(out_bin, merged_arr)
    if out_laz is not None:
        write_laz(out_laz, merged_arr, template_las)
    return True, sensor_points


# -------------------------
# Discovery helpers
# -------------------------
def load_all_transform_matrices(extr: Dict) -> Dict[str, np.ndarray]:
    result = {}
    for sensor_name in extr:
        try:
            result[sensor_name] = get_transform_matrix(extr, sensor_name)
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping '%s' extrinsic: %s", sensor_name, exc)
    return result


def discover_lidar_indexes(lidar_root: Path) -> Dict[str, List[Tuple[int, Path]]]:
    result = {}
    for p in sorted(lidar_root.glob("LIDAR*")):
        if p.is_dir() and p.name in ALLOWED_LIDARS:
            idx = build_lidar_index(p)
            if idx:
                result[p.name] = idx
    return result


def require_anchor(name: str, mapping: Dict, what: str) -> None:
    if name not in mapping:
        raise RuntimeError(f"Anchor lidar '{name}' {what}")


# -------------------------
# Visualization
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
        self.sensor_colors = {n: clone_color(SENSOR_COLORS.get(n, gui.Color(0.7, 0.7, 0.7))) for n in sensor_points}
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
        ok, cancel = gui.Button("OK"), gui.Button("Cancel")
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

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--extr", default="EXTRINSICS_FILE_PATH")
    ap.add_argument("--lidar_root", default="LIDAR_DATA_PATH")
    ap.add_argument("--out_dir_pcd", default="OUT_DIR")
    ap.add_argument("--out_dir_bin", default="OUT_DIR_BIN")
    ap.add_argument("--out_dir_laz", default="OUT_DIR_LAZ")
    ap.add_argument("--out_extrinsics", default="", help="Defaults to <out_dir_pcd>/../extrinsics_<origin>.json")
    ap.add_argument("--anchor", default=ANCHOR_LIDAR)
    ap.add_argument("--lidar_origin", choices=["lidar_top", "center", "back"], default="lidar_top")
    ap.add_argument("--max_dt", type=float, default=MAX_TIME_DIFF)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--visualize_ts", default="")
    ap.add_argument("--visualize_only", action="store_true")
    args = ap.parse_args()

    extr = load_extrinsics(args.extr)
    lidar_root = Path(args.lidar_root)
    out_dir_pcd, out_dir_bin = Path(args.out_dir_pcd), Path(args.out_dir_bin)
    out_dir_laz = Path(args.out_dir_laz) if args.out_dir_laz else None
    out_dir_pcd.mkdir(parents=True, exist_ok=True)
    out_dir_bin.mkdir(parents=True, exist_ok=True)
    if out_dir_laz is not None:
        out_dir_laz.mkdir(parents=True, exist_ok=True)

    lidar_indexes = discover_lidar_indexes(lidar_root)
    all_transform_matrices = load_all_transform_matrices(extr)
    transform_matrices = {n: all_transform_matrices[n] for n in lidar_indexes if n in all_transform_matrices}
    lidar_transform_matrices = {n: m for n, m in all_transform_matrices.items() if n.upper().startswith("LIDAR")}

    logger.info("Lidars in use: %s", list(lidar_indexes.keys()))
    logger.info("Lidars with valid extrinsic: %s", list(transform_matrices.keys()))
    logger.info("Output origin: %s", args.lidar_origin)

    require_anchor(args.anchor, transform_matrices, "has no valid extrinsic or no data")
    require_anchor(args.anchor, lidar_transform_matrices, f"has no valid extrinsic in {args.extr}")
    require_anchor(args.anchor, lidar_indexes, "has no data")

    shifted_lidar_extrinsics = shift_extrinsics(lidar_transform_matrices, args.lidar_origin, args.anchor)
    output_extrinsics = {**all_transform_matrices, **shifted_lidar_extrinsics}
    out_extrinsics_path = Path(args.out_extrinsics) if args.out_extrinsics else out_dir_pcd.parent / f"extrinsics_{args.lidar_origin}.json"
    save_extrinsics(out_extrinsics_path, output_extrinsics)
    logger.info(
        "Saved extrinsics for %d sensors to: %s (%d LIDAR_* shifted, %d others unchanged)",
        len(output_extrinsics), out_extrinsics_path, len(shifted_lidar_extrinsics),
        len(output_extrinsics) - len(shifted_lidar_extrinsics),
    )

    anchor_index = lidar_indexes[args.anchor]
    logger.info("len(anchor_index): %d", len(anchor_index))
    if args.limit > 0:
        anchor_index = anchor_index[: args.limit]

    if not args.visualize_only:
        ok = miss = 0
        for t_ns, p in tqdm(anchor_index, desc=f"Merging (anchor={args.anchor})"):
            token = p.stem
            out_laz = (out_dir_laz / f"{token}.laz") if out_dir_laz is not None else None
            wrote, _ = merge_one_timestamp(
                t_ns, lidar_indexes, transform_matrices,
                out_dir_pcd / f"{token}.pcd", out_dir_bin / f"{token}.bin", out_laz,
                max_time_diff_sec=args.max_dt, origin=args.lidar_origin, anchor_lidar=args.anchor,
            )
            ok, miss = (ok + 1, miss) if wrote else (ok, miss + 1)

        logger.info("==== DONE ==== wrote=%d no_data=%d", ok, miss)
        logger.info("Output .pcd: %s | .bin: %s | .laz: %s", out_dir_pcd, out_dir_bin, out_dir_laz)

    if args.visualize:
        if not _HAS_OPEN3D:
            logger.warning("--visualize requested but open3d is not installed. Skipping.")
            return

        target_ns = ts_ns_from_stem(args.visualize_ts) if args.visualize_ts else anchor_index[0][0]
        logger.info("[VIS] Visualizing timestamp: %d", target_ns)
        _, sensor_points = merge_one_timestamp(
            target_ns, lidar_indexes, transform_matrices,
            out_pcd=None, out_bin=None, out_laz=None,
            max_time_diff_sec=args.max_dt, collect_sensor_points=True,
            origin=args.lidar_origin, anchor_lidar=args.anchor,
        )
        if not sensor_points:
            logger.info("No points to show.")
            return

        total = sum(len(v) for v in sensor_points.values())
        logger.info("%d layers, %s points", len(sensor_points), f"{total:,}")
        LidarViewer(sensor_points, f"LiDAR — {target_ns}").run()


if __name__ == "__main__":
    main()