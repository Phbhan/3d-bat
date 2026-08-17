#!/usr/bin/env python3

import argparse
import filecmp
import json
import math
import re
from pathlib import Path
from shutil import copy2

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MAX_FRAMES = 20

CAM_POS_FROM_FOLDER = {
    "CAM_BACK": "rear",
    "CAM_FRONT": "front",
    "CAM_FRONT_LEFT": "left",
    "CAM_FRONT_RIGHT": "right",
}

DEST_CAM_IMAGE_DIR = "CAM_FRONT_LEFT"
DEST_BEV_DIR = "CAM_FRONT"

# Max allowed distance (meters) from ego, per object category.
# Objects farther than this are dropped from the annotation.
CATEGORY_MAX_DIST = {
    "CAR": 15.0,
    "MOTORCYCLE": 12.0,
    "PEDESTRIAN": 12.0,
    "BUS": 20.0,
    "TRUCK": 20.0,
    "CONTAINER": 20.0,
}


def parse_day(name):
    return name.split("_")[0]


def get_files(folder, exts=None):
    if not folder.is_dir():
        return []
    files = [f for f in folder.iterdir() if f.is_file()]
    if exts:
        files = [f for f in files if f.suffix.lower() in exts]
    return sorted(files, key=lambda x: x.name)


def files_by_stem(folder, exts=None):
    return {f.stem: f for f in get_files(folder, exts)}


def extract_timestamp(path):
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        return None
    return int("".join(numbers))


def same_file(file1, file2):
    try:
        return filecmp.cmp(file1, file2, shallow=False)
    except OSError:
        return False


def load_dummy_files(dummy_dir):
    """
    Load exactly one dummy file for each data type.

    dummy/
        CAM_FRONT_LEFT/
            000020.png
        CAM_FRONT/
            000020.png
        annotations/
            000020.json
    """
    if not dummy_dir:
        return None, None, None

    dummy_dir = Path(dummy_dir)

    cam_files = get_files(dummy_dir / "CAM_FRONT_LEFT")
    bev_files = get_files(dummy_dir / "CAM_FRONT")
    ann_files = get_files(dummy_dir / "annotations")

    if len(cam_files) != 1:
        raise ValueError(
            f"Expected exactly 1 dummy camera image in "
            f"{dummy_dir / 'CAM_FRONT_LEFT'}, "
            f"found {len(cam_files)}"
        )

    if len(bev_files) != 1:
        raise ValueError(
            f"Expected exactly 1 dummy BEV image in "
            f"{dummy_dir / 'CAM_FRONT'}, "
            f"found {len(bev_files)}"
        )

    if len(ann_files) != 1:
        raise ValueError(
            f"Expected exactly 1 dummy annotation in "
            f"{dummy_dir / 'annotations'}, "
            f"found {len(ann_files)}"
        )

    return cam_files[0], bev_files[0], ann_files[0]


def object_distance(label):
    location = label.get("box3d", {}).get("location", {})
    x = location.get("x", 0.0)
    y = location.get("y", 0.0)
    return math.hypot(x, y)


def filter_annotation_labels(ann_path, require_clearly_visible=False):
    """
    Load an annotation file and drop any object that:
      - is farther from ego than the max allowed for its category
        (CATEGORY_MAX_DIST), and/or
      - (optionally) has Visibility != "CLEARLY".

    Categories with no configured distance limit are kept as-is
    with respect to distance.

    Returns (data, kept_labels):
        data        - parsed JSON with "labels" replaced by the
                       filtered list, or None if the file could not
                       be read/parsed.
        kept_labels - the filtered list of labels (empty if none
                       survived, or data is None).
    """
    try:
        with open(ann_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  WARNING: could not read annotation {ann_path}: {e}")
        return None, []

    kept = []

    for label in data.get("labels", []):
        category = label.get("category")
        max_dist = CATEGORY_MAX_DIST.get(category)

        # No rule configured for this category: keep it (distance-wise).
        if max_dist is not None and object_distance(label) > max_dist:
            continue

        if require_clearly_visible:
            visibility = label.get("box3d", {}).get("visibility")

            if visibility != "CLEARLY":
                continue

        kept.append(label)

    data["labels"] = kept
    return data, kept


def find_closest_annotation(image_file, annotation_files, used):
    image_ts = extract_timestamp(image_file)

    if image_ts is None:
        return None

    best = None
    best_diff = None

    for ann in annotation_files:
        if ann in used:
            continue

        ann_ts = extract_timestamp(ann)

        if ann_ts is None:
            continue

        diff = abs(image_ts - ann_ts)

        if best_diff is None or diff < best_diff:
            best = ann
            best_diff = diff

    if best is not None:
        used.add(best)

    return best


def build_annotation_matches(image_files, annotation_files):
    """
    Match each camera image to the closest annotation timestamp.

    Annotation matching is completely independent of dummy detection.
    Dummy annotations are filtered separately.
    """
    annotation_files = sorted(
        annotation_files,
        key=lambda x: (
            extract_timestamp(x) is None,
            extract_timestamp(x) or 0,
            x.name,
        ),
    )

    used = set()
    matches = {}

    for image in sorted(image_files, key=lambda x: x.name):
        matches[image] = find_closest_annotation(
            image,
            annotation_files,
            used,
        )

    return matches

def process_chunk(
    stems,
    cam_files,
    bev_files,
    filtered_annotations,
    output_root,
    index,
    cam_pos,
    car_name,
    lsize,
    day,
    dummy_cam,
    dummy_bev,
    dummy_ann,
):
    parts = [f"input_{index}", cam_pos, car_name]

    if lsize:
        parts.append(lsize)
    parts.append(day)

    dest_root = output_root / "_".join(parts)
    out_cam = dest_root / "images" / DEST_CAM_IMAGE_DIR
    out_bev = dest_root / "images" / DEST_BEV_DIR
    out_ann = dest_root / "annotations"

    out_cam.mkdir(parents=True, exist_ok=True)
    out_bev.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)

    # Copy real frames.
    for i, stem in enumerate(stems):
        cam_src = cam_files[stem]
        bev_src = bev_files[stem]

        copy2(
            cam_src,
            out_cam / f"{i:06d}{cam_src.suffix.lower()}",
        )

        copy2(
            bev_src,
            out_bev / f"{i:06d}{bev_src.suffix.lower()}",
        )

        ann_data = filtered_annotations.get(stem)

        if ann_data is not None:
            with open(out_ann / f"{i:06d}.json", "w") as f:
                json.dump(ann_data, f, indent=2)

    # --------------------------------------------------------
    # ALWAYS COPY DUMMY AS THE LAST FRAME
    # --------------------------------------------------------

    if dummy_cam is not None:
        copy2(dummy_cam, out_cam)

    if dummy_bev is not None:
        copy2(dummy_bev, out_bev)

    if dummy_ann is not None:
        copy2(dummy_ann, out_ann)

    print(f"input_{index}_{cam_pos}: {len(stems)} + 1")


def process_camera(
    cam_dir,
    bev_files,
    ann_dir,
    output_root,
    car_name,
    lsize,
    day,
    dummy_cam,
    dummy_bev,
    dummy_ann,
    start_index=0,
    require_clearly_visible=False,
):
    cam_files = files_by_stem(cam_dir, IMAGE_EXTS)

    if not cam_files:
        print(f"WARNING: no images in {cam_dir}")
        return

    # Camera and BEV must have the same filename/stem.
    stems = sorted(set(cam_files) & set(bev_files))

    print(
        f"{cam_dir.name}: "
        f"{len(cam_files)} camera images, "
        f"{len(stems)} frames"
    )

    if not stems:
        return

    # --------------------------------------------------------
    # Annotation matching
    # --------------------------------------------------------
    ann_files = get_files(ann_dir)
    selected_images = [cam_files[stem] for stem in stems]
    annotation_matches = build_annotation_matches(
        selected_images,
        ann_files,
    )

    # --------------------------------------------------------
    # Filter each annotation (distance per category, optionally
    # visibility), then drop any stem left with no annotation at
    # all, or no objects surviving the filter.
    # --------------------------------------------------------
    filtered_annotations = {}
    valid_stems = []

    for stem in stems:
        cam_src = cam_files[stem]
        ann_src = annotation_matches.get(cam_src)

        if ann_src is None:
            continue

        data, kept = filter_annotation_labels(
            ann_src,
            require_clearly_visible=require_clearly_visible,
        )

        if data is None or not kept:
            continue

        filtered_annotations[stem] = data
        valid_stems.append(stem)

    dropped = len(stems) - len(valid_stems)
    if dropped:
        print(
            f"  -> dropped {dropped} frame(s) with no annotation "
            f"or no objects surviving the filter"
        )

    stems = valid_stems

    if not stems:
        return

    # --------------------------------------------------------
    # Split into chunks of MAX_FRAMES.
    # --------------------------------------------------------
    num_chunks = math.ceil(len(stems) / MAX_FRAMES)
    print(f"  -> {num_chunks} output folders")

    for chunk_number in range(num_chunks):
        start = chunk_number * MAX_FRAMES
        end = min(start + MAX_FRAMES,len(stems))

        chunk = stems[start:end]

        process_chunk(
            stems=chunk,
            cam_files=cam_files,
            bev_files=bev_files,
            filtered_annotations=filtered_annotations,
            output_root=output_root,
            index=start_index + chunk_number,
            cam_pos=CAM_POS_FROM_FOLDER[cam_dir.name],
            car_name=car_name,
            lsize=lsize,
            day=day,
            dummy_cam=dummy_cam,
            dummy_bev=dummy_bev,
            dummy_ann=dummy_ann,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-root",
        required=True,
        help="Dataset/session folder",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output dataset folder",
    )
    parser.add_argument(
        "--car-name",
        required=True,
    )
    parser.add_argument(
        "--lsize",
        default="",
    )
    parser.add_argument(
        "--dummy-dir",
        default=None,
        help="Directory containing one dummy camera, BEV and annotation",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--require-clearly-visible",
        action="store_true",
        help="Drop objects whose Visibility is not CLEARLY",
    )

    args = parser.parse_args()

    session_dir = Path(args.input_root)
    output_root = Path(args.output_root)
    dummy_dir = Path(args.dummy_dir) if args.dummy_dir else None

    if not session_dir.is_dir():
        raise SystemExit(f"Input not found: {session_dir}")

    images_dir = session_dir / "images"
    bev_dir = session_dir / "images_BEV"

    if not images_dir.is_dir():
        raise SystemExit(f"Missing images directory: {images_dir}")

    # --------------------------------------------------------
    # Load exactly:
    #   1 dummy camera image
    #   1 dummy BEV image
    #   1 dummy annotation
    # --------------------------------------------------------
    dummy_cam, dummy_bev, dummy_ann = load_dummy_files(dummy_dir)

    if dummy_dir:
        print(f"Dummy camera: {dummy_cam}")
        print(f"Dummy BEV:    {dummy_bev}")
        print(f"Dummy ann:    {dummy_ann}")

    bev_files = files_by_stem(
        bev_dir,
        IMAGE_EXTS,
    )

    cam_dirs = sorted(
        [
            d for d in images_dir.iterdir()
            if d.is_dir() and d.name in CAM_POS_FROM_FOLDER
        ],
        key=lambda x: x.name,
    )

    if not cam_dirs:
        raise SystemExit(
            f"No recognized camera folders in {images_dir}"
        )

    day = parse_day(session_dir.name)

    print(f"Dataset: {session_dir.name}")
    print(f"Day: {day}")
    print(f"BEV images: {len(bev_files)}")
    print(
        "Cameras:",
        ", ".join(d.name for d in cam_dirs),
    )

    # Each camera position starts from its own index.
    for cam_dir in cam_dirs:
        ann_dir = session_dir / f"annotations_{cam_dir.name}"

        process_camera(
            cam_dir=cam_dir,
            bev_files=bev_files,
            ann_dir=ann_dir,
            output_root=output_root,
            car_name=args.car_name,
            lsize=args.lsize,
            day=day,
            dummy_cam=dummy_cam,
            dummy_bev=dummy_bev,
            dummy_ann=dummy_ann,
            start_index=args.start_index,
            require_clearly_visible=args.require_clearly_visible,
        )

    print("Done.")


if __name__ == "__main__":
    main()