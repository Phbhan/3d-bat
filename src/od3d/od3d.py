import torch
from src.od3d.fisheye_devkit import FisheyeCam

fisheye_cams = {
    "CAM_FRONT_LEFT": FisheyeCam(cam="left",calib_path="src/application/config/vf6_pnk/cameraData.json"),
    "CAM_FRONT": FisheyeCam(cam="front",calib_path="src/application/config/vf6_pnk/cameraData.json"),
    "CAM_BACK": FisheyeCam(cam="rear",calib_path="src/application/config/vf6_pnk/cameraData.json"),
    "CAM_FRONT_RIGHT": FisheyeCam(cam="right",calib_path="src/application/config/vf6_pnk/cameraData.json")
}

def project_bbox3d(
    bbox_3d_world: torch.Tensor,
    cam_pos: str
):    

    projection_cam = fisheye_cams[cam_pos]
    # print("bbox_3d_world: ", bbox_3d_world)
    camera_points = projection_cam.project_world_to_cam(bbox_3d_world)

    fisheye_points, _ = projection_cam.project_cam_to_fe_image(camera_points)
    # print("translation vector: ", translation_vector)
    # print("camera_points: ", camera_points)
    # print("fisheye_points: ", fisheye_points)
    return fisheye_points