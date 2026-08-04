#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import re


# Box transformation:
BOX_X_OFFSET = 0.0
BOX_Z_VALUE = 0.0


def rename_json_files(annotation_dir: Path):
    """
    Remove numeric prefixes from annotation filenames.

    Examples:
        1-1783757599-099982977.json
            -> 1783757599-099982977.json

        15-1783757599-099982977.json
            -> 1783757599-099982977.json
    """
    for file_path in annotation_dir.glob("*.json"):
        if len(str(file_path.name).split("-")) < 3:
            print(f"SKIP: {file_path.name} does not have a numeric prefix")
            continue
        # Remove numeric prefix: 1-, 2-, 15-, 123-, ...
        new_name = re.sub(r"^\d+-", "", file_path.name)
        new_path = file_path.with_name(new_name)

        # Skip files that don't have a numeric prefix
        if new_name == file_path.name:
            continue

        if new_path.exists():
            print(f"SKIP: {new_name} already exists")
            continue

        file_path.rename(new_path)
        print(f"Renamed: {file_path.name} -> {new_name}")


def modify_box_annotations(
    annotation_dir: Path,
    x_offset: float = BOX_X_OFFSET,
    z_value: float = BOX_Z_VALUE,
):
    """
    Modify all 3D bounding boxes in OpenLABEL JSON files.

    For every cuboid:
        x = x + x_offset
        y = unchanged
        z = z_value

    The cuboid format is assumed to be:

        val = [
            x,
            y,
            z,
            qx,
            qy,
            qz,
            qw,
            length,
            width,
            height
        ]
    """

    json_files = sorted(annotation_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in: {annotation_dir}")
        return

    total_files = 0
    modified_files = 0
    total_boxes = 0

    for file_path in json_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to read {file_path}: {e}")
            continue

        file_box_count = 0

        try:
            frames = data["openlabel"]["frames"]
        except KeyError:
            print(f"[WARN] No openlabel.frames found: {file_path}")
            continue

        for frame_id, frame_data in frames.items():
            objects = frame_data.get("objects", {})

            for object_id, object_data in objects.items():
                cuboid = (
                    object_data
                    .get("object_data", {})
                    .get("cuboid")
                )

                if cuboid is None:
                    continue

                val = cuboid.get("val")

                if not isinstance(val, list) or len(val) < 3:
                    print(
                        f"[WARN] Invalid cuboid val in "
                        f"{file_path}, frame={frame_id}, object={object_id}"
                    )
                    continue

                old_x = val[0]
                old_y = val[1]
                old_z = val[2]

                # Apply requested transformation.
                val[0] = old_x + x_offset
                val[1] = old_y
                val[2] = z_value

                file_box_count += 1

                print(
                    f"{file_path.name} | "
                    f"frame={frame_id} | "
                    f"object={object_id} | "
                    f"xyz: "
                    f"({old_x:.3f}, {old_y:.3f}, {old_z:.3f}) "
                    f"-> "
                    f"({val[0]:.3f}, {val[1]:.3f}, {val[2]:.3f})"
                )

        if file_box_count > 0:
            # Keep the JSON readable.
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

            modified_files += 1
            total_boxes += file_box_count

        total_files += 1

    print("\n==== BOX MODIFICATION DONE ====")
    print(f"JSON files processed: {total_files}")
    print(f"JSON files modified:  {modified_files}")
    print(f"Boxes modified:       {total_boxes}")
    print(f"X offset:              {x_offset} m")
    print(f"Z value:               {z_value} m")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rename OpenLABEL annotation files and optionally "
            "modify 3D bounding box positions."
        )
    )

    parser.add_argument(
        "--annotations_dir",
        type=Path,
        default="",
        help="Directory containing annotation JSON files.",
    )

    parser.add_argument(
        "--modify-box",
        action="store_true",
        help=(
            "Modify every 3D bounding box."
        ),
    )

    parser.add_argument(
        "--rename",
        action="store_true",
        help=(
            "Remove numeric prefixes from JSON filenames, "
            "e.g. 1-1783757599-xxx.json -> 1783757599-xxx.json."
        ),
    )

    parser.add_argument(
        "--x-offset",
        type=float,
        default=BOX_X_OFFSET,
        help="X offset applied to every bounding box. Default: 0.0 m.",
    )

    parser.add_argument(
        "--z-value",
        type=float,
        default=BOX_Z_VALUE,
        help="Z value assigned to every bounding box. Default: 0.0 m.",
    )

    args = parser.parse_args()

    annotation_dir = args.annotations_dir

    if not annotation_dir.exists():
        raise FileNotFoundError(
            f"Annotation directory does not exist: {annotation_dir}"
        )

    # Modify boxes if requested.
    if args.modify_box:
        modify_box_annotations(
            annotation_dir=annotation_dir,
            x_offset=args.x_offset,
            z_value=args.z_value,
        )

    # Rename files if requested.
    if args.rename:
        print("\n==== RENAMING JSON FILES ====")
        rename_json_files(annotation_dir)


if __name__ == "__main__":
    main()
