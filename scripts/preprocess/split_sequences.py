#!/usr/bin/env python3

from pathlib import Path
import shutil
import argparse
import math


# ============================================================
# Configuration
# ============================================================

DATA_DIRS = [
    "annotations_CAM_FRONT",
    "annotations_CAM_FRONT_LEFT",
    "annotations_CAM_FRONT_RIGHT",
    "annotations_CAM_BACK",

    "images/CAM_FRONT",
    "images/CAM_FRONT_LEFT",
    "images/CAM_FRONT_RIGHT",
    "images/CAM_BACK",

    "images_BEV",

    "point_clouds/LIDAR_TOP",
]

DEFAULT_DATA_PER_SPLIT = 20


# ============================================================
# Helper functions
# ============================================================

def get_timestamp(path: Path) -> str:
    """
    Get timestamp from filename.

    Example:
        1783757599-099982977.jpg
        -> 1783757599-099982977
    """
    return path.stem


def timestamp_sort_key(path: Path):
    """
    Sort timestamp numerically.

    Example:
        1783757599-099982977

    is converted to:

        (1783757599, 99982977)

    This avoids incorrect lexicographical sorting.
    """

    timestamp = get_timestamp(path)

    try:
        sec, frac = timestamp.split("-")
        return int(sec), int(frac)
    except ValueError:
        # Fallback if filename doesn't follow timestamp format
        return timestamp


def get_sorted_files(directory: Path):
    """
    Return all files in a directory sorted by timestamp.
    """

    if not directory.exists():
        print(f"[WARNING] Directory does not exist: {directory}")
        return []

    files = [
        f for f in directory.iterdir()
        if f.is_file()
    ]

    files.sort(key=timestamp_sort_key)

    return files


# ============================================================
# Split dataset
# ============================================================

def split_dataset(
    input_root: Path,
    output_root: Path,
    data_per_split: int = DEFAULT_DATA_PER_SPLIT,
):

    print("=" * 70)
    print("Dataset splitting by INDEX")
    print("=" * 70)

    print(f"Input          : {input_root}")
    print(f"Output         : {output_root}")
    print(f"Data per split : {data_per_split}")
    print()

    # --------------------------------------------------------
    # 1. Read every directory independently
    # --------------------------------------------------------

    directory_files = {}

    for relative_dir in DATA_DIRS:

        directory = input_root / relative_dir

        files = get_sorted_files(directory)

        directory_files[relative_dir] = files

        print(
            f"{relative_dir:<45} "
            f"{len(files):>6} files"
        )

    print()

    # --------------------------------------------------------
    # 2. Determine number of samples
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # Since folders may contain different numbers of files,
    # use the minimum number of files.
    #
    # This prevents an incomplete sample from being created.
    # --------------------------------------------------------

    non_empty_lengths = [
        len(files)
        for files in directory_files.values()
        if len(files) > 0
    ]

    if not non_empty_lengths:
        print("[ERROR] No files found.")
        return

    total_samples = min(non_empty_lengths)

    print(f"Total usable samples: {total_samples}")

    if total_samples < max(non_empty_lengths):
        print(
            "[WARNING] Some directories contain more files "
            "than others."
        )
        print(
            f"Only the first {total_samples} files "
            "from each directory will be used."
        )

    # --------------------------------------------------------
    # 3. Number of output datasets
    # --------------------------------------------------------

    num_splits = math.ceil(
        total_samples / data_per_split
    )

    print(f"Number of subdatasets: {num_splits}")
    print()

    # --------------------------------------------------------
    # 4. Create each subdataset
    # --------------------------------------------------------

    for split_idx in range(num_splits):

        start_idx = split_idx * data_per_split
        end_idx = min(
            start_idx + data_per_split,
            total_samples,
        )

        split_name = f"{split_idx:03d}"

        split_root = output_root / split_name

        print(
            f"[{split_name}] "
            f"index {start_idx} -> {end_idx - 1} "
            f"({end_idx - start_idx} samples)"
        )

        # ----------------------------------------------------
        # Process every directory
        # ----------------------------------------------------

        for relative_dir, files in directory_files.items():

            output_dir = split_root / relative_dir

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            selected_files = files[
                start_idx:end_idx
            ]

            for source_file in selected_files:

                destination_file = (
                    output_dir / source_file.name
                )

                shutil.copy2(
                    source_file,
                    destination_file,
                )

    # --------------------------------------------------------
    # 5. Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Done")
    print("=" * 70)

    print(f"Total samples : {total_samples}")
    print(f"Data/split    : {data_per_split}")
    print(f"Subdatasets   : {num_splits}")
    print(f"Output        : {output_root}")


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Split dataset into subdatasets by sorted "
            "file index, without matching timestamps."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input dataset A",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory",
    )

    parser.add_argument(
        "--num-data",
        type=int,
        default=20,
        help="Number of data per subdataset",
    )

    args = parser.parse_args()

    if args.num_data <= 0:
        raise ValueError(
            "--num-data must be greater than 0"
        )

    split_dataset(
        input_root=args.input,
        output_root=args.output,
        data_per_split=args.num_data,
    )


if __name__ == "__main__":
    main()