#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import re
import shutil
import math

CAMERAS = {
    "CAM_FRONT": lambda x, y: x >= 0 and abs(x) < 20, 
    "CAM_FRONT_RIGHT": lambda x, y: x >= 0 and y <= 0 and abs(y) < 20,
    "CAM_FRONT_LEFT": lambda x, y: x >= 0 and y >= 0 and abs(y) < 20,
    "CAM_BACK": lambda x, y: x < 0 and abs(x) < 20,
}

# Box transformation:
BOX_X_OFFSET = 0.0
BOX_Z_VALUE = 0.0

def split_annotations_by_camera(
    annotation_dir: Path,
    x_offset: float,
    z_value: float,
):
    for cam_name, keep_fn in CAMERAS.items():

        out_dir = annotation_dir.parent / f"annotations_{cam_name}"
        
        # Delete existing folder
        if out_dir.exists():
            shutil.rmtree(out_dir)
            print(f"Deleted: {out_dir}")

        # Create a new empty folder
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {out_dir}")

        for src_file in sorted(annotation_dir.glob("*.json")):

            with open(src_file) as f:
                data = json.load(f)

            data["cam_pos"] = cam_name

            new_labels = []

            for label in data.get("labels", []):

                location = label["box3d"]["location"]

                x = float(location["x"])
                y = float(location["y"])

                location["x"] = x
                location["y"] = y
                location["z"] = z_value

                if keep_fn(x + x_offset, y):
                    new_labels.append(label)

            data["labels"] = new_labels

            dst = out_dir / src_file.name
            with open(dst, "w") as f:
                json.dump(data, f, indent=2)

        print(f"{cam_name}: done")


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
            file_path.replace(new_path)     
        else:
            file_path.rename(new_path)
            print(f"Renamed: {file_path.name} -> {new_name}")


def modify_box_annotations(
    annotation_dir: Path,
    x_offset: float = BOX_X_OFFSET,
    z_value: float = BOX_Z_VALUE,
):
    """
    Modify all 3D bounding boxes in the custom annotation format.

    Input format:

    {
        "name": "...",
        "cam_pos": "...",
        "timestamp": ...,
        "index": ...,
        "weather": "...",
        "labels": [
            {
                "id": ...,
                "category": "...",
                "box3d": {
                    "dimension": {...},
                    "location": {
                        "x": ...,
                        "y": ...,
                        "z": ...
                    },
                    "orientation": {...}
                }
            }
        ]
    }

    For every box:
        x = x + x_offset
        y = unchanged
        z = z_value
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

        labels = data.get("labels")

        if labels is None:
            print(f"[WARN] No labels found: {file_path}")
            continue

        file_box_count = 0

        for label in labels:
            location = (
                label.get("box3d", {})
                     .get("location")
            )

            if location is None:
                continue

            try:
                old_x = float(location["x"])
                old_y = float(location["y"])
                old_z = float(location["z"])
            except KeyError:
                print(f"[WARN] Invalid location in {file_path}")
                continue

            location["x"] = old_x + x_offset
            location["y"] = old_y
            location["z"] = z_value

            file_box_count += 1

            print(
                f"{file_path.name} | "
                f"id={label.get('id')} | "
                f"xyz: "
                f"({old_x:.3f}, {old_y:.3f}, {old_z:.3f}) "
                f"-> "
                f"({location['x']:.3f}, {location['y']:.3f}, {location['z']:.3f})"
            )

        if file_box_count > 0:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

            modified_files += 1
            total_boxes += file_box_count

        total_files += 1

    print("\n==== BOX MODIFICATION DONE ====")
    print(f"JSON files processed: {total_files}")
    print(f"JSON files modified:  {modified_files}")
    print(f"Boxes modified:       {total_boxes}")
    print(f"X offset:             {x_offset} m")
    print(f"Z value:              {z_value} m")


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

    split_annotations_by_camera(
        annotation_dir,
        args.x_offset,
        args.z_value,
    )
    
    for cam_name, keep_fn in CAMERAS.items():
        annotation_dir_cam = annotation_dir.parent / f"annotations_{cam_name}"

        # Modify boxes if requested.
        if args.modify_box:
            modify_box_annotations(
                annotation_dir=annotation_dir_cam,
                x_offset=args.x_offset,
                z_value=args.z_value,
            )
        
        # Rename files if requested.
        if args.rename:
            print("\n==== RENAMING JSON FILES ====")
            rename_json_files(annotation_dir_cam)
        



if __name__ == "__main__":
    main()
