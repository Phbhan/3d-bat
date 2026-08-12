import torch
from src.od3d.fisheye_devkit import FisheyeCam

fisheye_cams = {
    "CAM_FRONT_LEFT": FisheyeCam(cam="left", calib_path="src/application/config/vf6_pnk/cameraData.json"),
    "CAM_FRONT": FisheyeCam(cam="front", calib_path="src/application/config/vf6_pnk/cameraData.json"),
    "CAM_BACK": FisheyeCam(cam="rear", calib_path="src/application/config/vf6_pnk/cameraData.json"),
    "CAM_FRONT_RIGHT": FisheyeCam(cam="right", calib_path="src/application/config/vf6_pnk/cameraData.json"),
}


def project_bbox3d(
    bbox_3d_world: torch.Tensor,
    cam_pos: str,
):
    """
    Project 3D bounding-box corners (world frame) into fisheye image coordinates.

    Args:
        bbox_3d_world: tensor of shape (1, N, 3, 9) where
            N = number of boxes (N >= 1),
            3 = xyz,
            9 = 8 corners + center.
            Single-box callers pass (1, 1, 3, 9).
        cam_pos: key into fisheye_cams, e.g. "CAM_FRONT"

    Returns:
        fisheye image points with the same leading batch dims, typically
        (1, N, 2, 9).
    """
    projection_cam = fisheye_cams[cam_pos]
    camera_points = projection_cam.project_world_to_cam(bbox_3d_world)
    fisheye_points, _, valid_mask = projection_cam.project_cam_to_fe_image(camera_points)
    return fisheye_points, valid_mask