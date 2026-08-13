"""
Generates a minimal, valid empty (or near-empty) .pcd point cloud file per
frame, for datasets that have no real lidar data but need something on disk
so the annotation tool's point-cloud loader doesn't 404 / crash.

Also writes point_cloud_filenames.txt to match, in the naming convention
tool_main.ts expects (Utils.pad(i, 6, 0) -> "000000", "000001", ...).

Usage:
    python scripts/create_fake_point_clouds.py --dataset hanpb2 \
        --sequence 20260710_1005_VF6_03_1783652730_1783654253 \
        --num-frames 100 \
        --lidar-channel LIDAR_TOP \
        --input-root ./input \
        --num-points 1        # 0 = truly empty cloud; >0 = a few placeholder points
"""
import argparse
import os
import struct


def write_pcd_ascii(path: str, num_points: int):
    """Writes a standard ASCII .pcd file with fields x y z intensity."""
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {num_points}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {num_points}\n"
        "DATA ascii\n"
    )
    with open(path, "w") as f:
        f.write(header)
        # A handful of placeholder points near the origin so viewers that
        # choke on a truly empty cloud (0 points) still have something to render.
        for i in range(num_points):
            x = 0.0 + i * 0.01
            y = 0.0
            z = 0.0
            intensity = 0.0
            f.write(f"{x} {y} {z} {intensity}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--lidar-channel", default="LIDAR_TOP")
    parser.add_argument("--input-root", default="./input")
    parser.add_argument("--num-points", type=int, default=0,
                         help="0 for a truly empty cloud, or a small number "
                              "of placeholder points if your loader errors on 0.")
    args = parser.parse_args()

    seq_dir = os.path.join(args.input_root, args.dataset, args.sequence)
    pc_dir = os.path.join(seq_dir, "point_clouds", args.lidar_channel)
    os.makedirs(pc_dir, exist_ok=True)

    filenames = []
    for i in range(args.num_frames):
        frame_id = f"{i:06d}"
        out_path = os.path.join(pc_dir, frame_id + ".pcd")
        write_pcd_ascii(out_path, args.num_points)
        filenames.append(frame_id + ".pcd")

    with open(os.path.join(seq_dir, "point_cloud_filenames.txt"), "w") as f:
        f.write("\n".join(filenames) + "\n")

    print(f"Wrote {args.num_frames} fake .pcd files to {pc_dir}")
    print(f"Wrote point_cloud_filenames.txt to {seq_dir}")


if __name__ == "__main__":
    main()