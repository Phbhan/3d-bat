#!/usr/bin/env python3
"""
Simplified camera undistortion.

Given:
  - intrinsics JSON:  {cam_name: {"camera_matrix": 3x3, "distortion_coefficients": [...]}}
  - extrinsics JSON:  {cam_name: [...]}  (unchanged by undistortion, just copied through)
  - images_root/<cam_name>/*.jpg|png     (5 cameras: CAM_P_F, CAM_P_FR, CAM_P_RB, CAM_P_FL, CAM_P_LB)

Produces:
  - out_root/<cam_name>/*.jpg            undistorted images
  - out_root/new_intrinsics.json         {cam_name: {camera_matrix: K_new, distortion_coefficients: 0, image_size}}
  - out_root/extrinsics.json             copy of input extrinsics (unaffected by undistortion)

Usage:
  python undistort_cameras.py \
      --images_root /path/to/CAMERA \
      --intr_path   /path/to/Intrinsics.json \
      --extr_path   /path/to/Extrinsics.json \
      --out_root    /path/to/output \
      --alpha 0 \
      --fisheye_cams CAM_P_FL,CAM_P_LB   # only if those cams use the fisheye model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CAMERAS = ["CAM_P_F", "CAM_P_FR", "CAM_P_RB", "CAM_P_FL", "CAM_P_LB"]
IMG_EXTS = (".jpg", ".jpeg", ".png")


def make_undistort_maps(
    K: np.ndarray,
    dist: np.ndarray,
    image_size: tuple[int, int],
    is_fisheye: bool,
    alpha: float,
):
    """Return (K_new, map1, map2) for cv2.remap."""
    W, H = image_size
    if is_fisheye:
        D = np.asarray(dist, dtype=np.float64).reshape(-1)[:4].reshape(4, 1)
        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (W, H), np.eye(3), balance=float(alpha)
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K_new, (W, H), cv2.CV_16SC2
        )
    else:
        D = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
        K_new, _ = cv2.getOptimalNewCameraMatrix(K, D, (W, H), float(alpha), (W, H))
        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K_new, (W, H), cv2.CV_16SC2)
    return K_new, map1, map2


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", required=True, type=Path,
                     help="Folder containing CAM_P_F/, CAM_P_FR/, CAM_P_RB/, CAM_P_FL/, CAM_P_LB/")
    ap.add_argument("--intr_path", required=True, type=Path)
    ap.add_argument("--extr_path", type=Path, default=None)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=0.0,
                     help="0 = crop all invalid pixels, 1 = keep all pixels (black corners)")
    ap.add_argument("--fisheye_cams", default="",
                     help="Comma-separated camera names that use the fisheye distortion model")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    fisheye_set = {c.strip() for c in args.fisheye_cams.split(",") if c.strip()}

    intr = json.loads(args.intr_path.read_text())
    args.out_root.mkdir(parents=True, exist_ok=True)

    new_intr: dict = {}

    for cam in CAMERAS:
        cam_folder = args.images_root / cam
        if not cam_folder.is_dir():
            print(f"[WARN] Skip missing folder: {cam_folder}")
            continue
        if cam not in intr:
            print(f"[WARN] {cam} missing in intrinsics JSON, skipping")
            continue

        images = sorted(p for p in cam_folder.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not images:
            print(f"[WARN] No images in {cam_folder}")
            continue

        K = np.asarray(intr[cam]["camera_matrix"], dtype=np.float64).reshape(3, 3)
        dist = np.asarray(
            intr[cam].get("distortion_coefficients", [0, 0, 0, 0, 0]), dtype=np.float64
        )

        img0 = cv2.imread(str(images[0]))
        if img0 is None:
            print(f"[WARN] Cannot read first image for {cam}: {images[0]}")
            continue
        H, W = img0.shape[:2]

        is_fisheye = cam in fisheye_set
        K_new, map1, map2 = make_undistort_maps(K, dist, (W, H), is_fisheye, args.alpha)

        out_folder = args.out_root / cam
        out_folder.mkdir(parents=True, exist_ok=True)

        n_ok, n_fail = 0, 0
        for img_path in images:
            out_path = out_folder / img_path.name
            if out_path.exists() and not args.overwrite:
                n_ok += 1
                continue
            img = cv2.imread(str(img_path))
            if img is None or img.shape[:2] != (H, W):
                print(f"[WARN] Bad/mismatched image, skipping: {img_path}")
                n_fail += 1
                continue
            undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(out_path), undist)
            n_ok += 1

        new_intr[cam] = {
            "camera_matrix": K_new.tolist(),
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
            "image_size": [W, H],
            "model": "fisheye" if is_fisheye else "pinhole",
            "alpha": float(args.alpha),
        }
        print(f"[OK] {cam}: {n_ok} undistorted, {n_fail} failed -> {out_folder}")

    intr_out_path = args.out_root / "new_intrinsics.json"
    intr_out_path.write_text(json.dumps(new_intr, indent=2))
    print(f"[DONE] New intrinsics written to {intr_out_path}")

    if args.extr_path is not None:
        extr = json.loads(args.extr_path.read_text())
        extr_out_path = args.out_root / "extrinsics.json"
        extr_out_path.write_text(json.dumps(extr, indent=2))
        print(f"[DONE] Extrinsics copied unchanged to {extr_out_path}")
        print("Note: undistortion does not change sensor->ego extrinsics, so these are passed through as-is.")


if __name__ == "__main__":
    main()