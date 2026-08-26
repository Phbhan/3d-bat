#!/usr/bin/env python3

from pathlib import Path
import shutil
import argparse


ANNOTATION_DIRS = [
    "annotations_CAM_FRONT",
    "annotations_CAM_FRONT_LEFT",
    "annotations_CAM_FRONT_RIGHT",
    "annotations_CAM_BACK",
]


def merge_annotations(split_root: Path, output_root: Path):
    print("=" * 70)
    print("Merging annotations from split datasets")
    print("=" * 70)

    print(f"Split root : {split_root}")
    print(f"Output     : {output_root}")
    print()

    if not split_root.exists():
        raise FileNotFoundError(
            f"Split dataset does not exist: {split_root}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Find subdatasets: 000, 001, 002, ...
    split_dirs = [
        d for d in split_root.iterdir()
        if d.is_dir()
    ]

    # Sort numerically
    split_dirs.sort(
        key=lambda x: int(x.name)
        if x.name.isdigit()
        else x.name
    )

    total_files = 0

    for split_dir in split_dirs:

        print(f"\nProcessing: {split_dir.name}")

        for annotation_dir in ANNOTATION_DIRS:

            source_dir = split_dir / annotation_dir
            destination_dir = output_root / annotation_dir

            if not source_dir.exists():
                print(
                    f"  [WARNING] Missing: {source_dir}"
                )
                continue

            destination_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            files = [
                f for f in source_dir.iterdir()
                if f.is_file()
            ]

            print(
                f"  {annotation_dir}: "
                f"{len(files)} files"
            )

            for source_file in files:

                destination_file = (
                    destination_dir / source_file.name
                )

                if destination_file.exists():
                    print(
                        f"  [WARNING] File already exists, "
                        f"overwriting: {destination_file}"
                    )

                shutil.copy2(
                    source_file,
                    destination_file,
                )

                total_files += 1

    print()
    print("=" * 70)
    print("Merge complete")
    print("=" * 70)
    print(f"Total annotation files copied: {total_files}")


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Merge annotation folders from split datasets "
            "back into the original dataset."
        )
    )

    parser.add_argument(
        "--split-root",
        required=True,
        type=Path,
        help="Root directory containing 000, 001, 002, ...",
    )

    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Original dataset directory",
    )

    args = parser.parse_args()

    merge_annotations(
        split_root=args.split_root,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()