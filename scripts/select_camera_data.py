#!/usr/bin/env python3

import argparse
import shutil
from bisect import bisect_left
from pathlib import Path


def timestamp_from_name(path: Path) -> int:
    sec, nsec = path.stem.split("-")
    return int(sec) * 1_000_000_000 + int(nsec)


def main():
    parser = argparse.ArgumentParser(
        description="Keep the image closest to each LiDAR frame by copying it to a new folder."
    )
    parser.add_argument(
        "--lidar_dir",
        required=True,
        help="Directory containing LiDAR .pcd files",
    )
    parser.add_argument(
        "--camera_root",
        required=True,
        help="Root directory containing camera folders (CAM_F_F, CAM_F_B, ...)",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory to save selected images",
    )
    args = parser.parse_args()

    lidar_dir = Path(args.lidar_dir)
    camera_root = Path(args.camera_root)
    out_dir = Path(args.out_dir)

    # -----------------------------
    # Read LiDAR timestamps
    # -----------------------------
    lidar_ts = sorted(
        timestamp_from_name(p)
        for p in lidar_dir.glob("*.pcd")
    )

    print(f"Found {len(lidar_ts)} LiDAR frames.")

    # -----------------------------
    # Process each camera
    # -----------------------------
    for cam_dir in sorted(camera_root.iterdir()):
        if not cam_dir.is_dir():
            continue

        images = sorted(cam_dir.glob("*.jpg"))
        if not images:
            continue

        image_info = [(timestamp_from_name(p), p) for p in images]
        image_ts = [x[0] for x in image_info]

        keep = set()

        for ts in lidar_ts:
            idx = bisect_left(image_ts, ts)

            if idx == 0:
                best = image_info[0][1]
            elif idx == len(image_ts):
                best = image_info[-1][1]
            else:
                before = image_info[idx - 1]
                after = image_info[idx]

                if abs(before[0] - ts) <= abs(after[0] - ts):
                    best = before[1]
                else:
                    best = after[1]

            keep.add(best)

        cam_out = out_dir / cam_dir.name
        cam_out.mkdir(parents=True, exist_ok=True)

        for img in sorted(keep):
            shutil.copy2(img, cam_out / img.name)

        print(
            f"{cam_dir.name:12s} "
            f"selected {len(keep):5d} / {len(images):5d} images"
        )

    print(f"\nFinished.\nOutput saved to: {out_dir}")


if __name__ == "__main__":
    main()