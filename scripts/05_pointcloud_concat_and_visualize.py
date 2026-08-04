"""
Tutorial 05: Multi-LiDAR concatenate and visualize.

"""

import json
import numpy as np
from pathlib import Path
import laspy
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from bisect import bisect_left

# ── Configuration ─────────────────────────────────────────────────────────────

SEQUENCE_FOLDER = Path("/home/hanpb2/workspace/Data/Data_PNK/500h/20260711_1512_VF6_03_1783757531_1783759331/")
EXTRINSIC_JSON  = Path("/home/hanpb2/workspace/Data/Data_PNK/calib/VF6_03/VF6_03_Extrinsics.json")

LIDAR_SUB_FOLDERS = ["LIDAR_E_F", "LIDAR_E_L", "LIDAR_E_R", "LIDAR_E_B", "LIDAR_TOP", "LIDAR_AT_F"]
TARGET_TIMESTAMP_STR = "1783757599-099982977"

DOWNSAMPLE_VOXEL_SIZE = 0.05
POINT_SIZE_DEFAULT    = 1.0
BACKGROUND_COLOR      = [0.9, 0.9, 0.9]
SHOW_COORDINATE_FRAME = True

CYLINDER_RADIUS = 6
MIN_Z_SIDE      = -0.5
MAX_Z_SIDE      = 2.0
THRESHOLD_NS    = 100_000_000  # 100 ms

# Default color per sensor (change here or in the GUI)
SENSOR_COLORS = {
    "LIDAR_E_F":  gui.Color(0.0, 0.8, 0.0),
    "LIDAR_E_L":  gui.Color(0.0, 0.8, 0.0),
    "LIDAR_E_R":  gui.Color(0.0, 0.8, 0.0),
    "LIDAR_E_B":  gui.Color(0.0, 0.8, 0.0),
    "LIDAR_TOP":  gui.Color(0.0, 0.4, 1.0),
    "LIDAR_AT_F": gui.Color(1.0, 0.0, 0.0),
}

# ── Load extrinsics ───────────────────────────────────────────────────────────

with open(EXTRINSIC_JSON) as f:
    extrinsics = json.load(f)

transform_matrices = {
    name: np.array(extrinsics[name], dtype=np.float64)
    for name in LIDAR_SUB_FOLDERS
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_timestamp(ts_str: str) -> int:
    sec, nsec = map(int, ts_str.split("-"))
    return sec * 1_000_000_000 + nsec

def clone_color(c: gui.Color) -> gui.Color:
    return gui.Color(c.red, c.green, c.blue, c.alpha)

def find_closest_file(target_ns: int, ts_sorted: list[int], file_dict: dict) -> str | None:
    if not ts_sorted:
        return None
    idx = bisect_left(ts_sorted, target_ns)
    candidates = []
    if idx < len(ts_sorted):
        candidates.append((abs(ts_sorted[idx] - target_ns), ts_sorted[idx]))
    if idx > 0:
        candidates.append((abs(ts_sorted[idx - 1] - target_ns), ts_sorted[idx - 1]))
    diff, best_ts = min(candidates)
    return file_dict[best_ts] if diff <= THRESHOLD_NS else None

def make_pcd(xyz: np.ndarray, downsample: bool = False) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if downsample and DOWNSAMPLE_VOXEL_SIZE > 0:
        pcd = pcd.voxel_down_sample(voxel_size=DOWNSAMPLE_VOXEL_SIZE)
    return pcd

def transform_points(T: np.ndarray, x, y, z, is_side: bool) -> np.ndarray:
    pts = T @ np.vstack((x, y, z, np.ones_like(x)))
    xt, yt, zt = pts[0], pts[1], pts[2]
    if is_side:
        x_shift = xt - 0.910
        keep = (x_shift**2 + yt**2 <= CYLINDER_RADIUS**2) & (zt >= MIN_Z_SIDE) & (zt <= MAX_Z_SIDE)
        xt, yt, zt = xt[keep], yt[keep], zt[keep]
    return np.column_stack((xt, yt, zt))

def read_lidar(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with laspy.open(path) as f:
        las = f.read()
    return (
        np.asarray(las.x, dtype=np.float64),
        np.asarray(las.y, dtype=np.float64),
        np.asarray(las.z, dtype=np.float64),
    )

def build_sensor_index(folder: Path) -> tuple[list[int], dict[int, str]]:
    ts_list, ts_to_file = [], {}
    for f in folder.glob("*.laz"):
        ts = parse_timestamp(f.stem)
        ts_list.append(ts)
        ts_to_file[ts] = f.name
    ts_list.sort()
    return ts_list, ts_to_file

def load_all_sensors(target_ns: int) -> dict[str, np.ndarray]:
    sensors: dict[str, np.ndarray] = {}

    top_path = SEQUENCE_FOLDER / "LIDAR" / "LIDAR_TOP" / f"{TARGET_TIMESTAMP_STR}.laz"
    if not top_path.is_file():
        raise FileNotFoundError(f"Top lidar not found: {top_path}")

    for sensor in LIDAR_SUB_FOLDERS:
        folder = SEQUENCE_FOLDER / "LIDAR" / sensor
        if sensor == "LIDAR_TOP":
            path = top_path
        else:
            if not folder.is_dir():
                print(f"Skip {sensor}: folder not found")
                continue
            ts_list, ts_to_file = build_sensor_index(folder)
            fname = find_closest_file(target_ns, ts_list, ts_to_file)
            if not fname:
                print(f"Skip {sensor}: no scan within {THRESHOLD_NS / 1e6:.0f} ms")
                continue
            path = folder / fname

        x, y, z = read_lidar(path)
        xyz = transform_points(transform_matrices[sensor], x, y, z, sensor.startswith("LIDAR_E_"))
        if len(xyz) == 0:
            print(f"Skip {sensor}: no points after filter")
            continue

        pcd = make_pcd(xyz, downsample=True)
        sensors[sensor] = np.asarray(pcd.points)
        print(f"{sensor}: {len(sensors[sensor]):,} points ({path.name})")

    return sensors

# ── Viewer ────────────────────────────────────────────────────────────────────

class LidarViewer:
    PANEL_WIDTH_EM = 17

    def __init__(self, sensor_points: dict[str, np.ndarray], title: str):
        self.sensor_points = sensor_points
        self.sensor_colors = {
            name: clone_color(SENSOR_COLORS.get(name, gui.Color(0.7, 0.7, 0.7)))
            for name in sensor_points
        }
        self.color_buttons: dict[str, gui.Button] = {}
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

    def _material(self, color: gui.Color) -> rendering.MaterialRecord:
        mat = rendering.MaterialRecord()
        mat.shader = "unlitSolidColor"
        mat.base_color = [color.red, color.green, color.blue, 1.0]
        mat.point_size = self.point_size
        return mat

    def _add_geometry(self, name: str):
        pcd = make_pcd(self.sensor_points[name])
        self.scene.scene.add_geometry(name, pcd, self._material(self.sensor_colors[name]))

    def _update_all_materials(self):
        for name in self.sensor_names:
            self.scene.scene.modify_geometry_material(
                name, self._material(self.sensor_colors[name])
            )
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

    def _apply_color(self, name: str, color: gui.Color):
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

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Timestamp: {TARGET_TIMESTAMP_STR}")
    sensor_points = load_all_sensors(parse_timestamp(TARGET_TIMESTAMP_STR))
    if not sensor_points:
        print("No points to show.")
        return

    total = sum(len(v) for v in sensor_points.values())
    print(f"\n{len(sensor_points)} layers, {total:,} points")
    print("Left panel: toggle layers, color button, point size slider.")

    LidarViewer(sensor_points, f"LiDAR — {TARGET_TIMESTAMP_STR}").run()

if __name__ == "__main__":
    main()
