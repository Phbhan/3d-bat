from typing import Union, List, Tuple

import numpy as np
import cv2
import torch
from src.od3d.fisheye_devkit.Camera import Camera
from torch import Tensor
from src.od3d.fisheye_devkit.cam_distortion_handler import DistortionModel


class FisheyeCam(Camera):
    def __init__(self, batch_size=32, num_objects=64, num_concern_kps=9, device: torch.device=torch.device("cpu"), cam="", calib_path=""):
        super().__init__(batch_size, num_objects, num_concern_kps, device, cam, calib_path)
        self.batch_size = batch_size
        self.distortion = DistortionModel(device)


    def project_image_to_norm_cam(self, points_img: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inverse project image points to normal camera coordinates.

        Reference:
            - https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html
            - https://oulu3dvision.github.io/calibgeneric/Kannala_Brandt_calibration.pdf

        Note: B: batch size; K: number of objects;

        Args:
             points_img: Points tensor in pixel coordinates of image. (B, K, 2, 9)
        Returns:
            - Points tensor in a normal camera coordinates.
            - The distorted angle between the incoming ray and the optical axis of fisheye camera.
            - Theta_ray
        """
        # Converts pixel coordinates to normalized image coordinates
        x_img = (points_img[:, :, 0, :] - self.intrinsics.matrix_k[0, 2]) / self.intrinsics.matrix_k[0, 0]
        y_img = (points_img[:, :, 1, :] - self.intrinsics.matrix_k[1, 2]) / self.intrinsics.matrix_k[1, 1]

        # Calculate the distorted angle between the incoming ray and the optical axis of fisheye camera.
        theta_distorted = torch.sqrt(x_img**2 + y_img**2)

        # Calculate the undistorted angle the incoming ray and the optical axis before distortion.
        theta = self.distortion.solve_distortion(theta_distorted.reshape(-1), self.intrinsics.matrix_d)
        theta = theta.reshape(theta_distorted.shape)

        # Calculate coordinates in normal camera plane
        radial_dist = torch.tan(theta)
        x_norm_cam = x_img * radial_dist / theta_distorted
        y_norm_cam = y_img * radial_dist / theta_distorted
        z_norm_cam = torch.ones_like(x_norm_cam)

        theta_ray = torch.atan2(x_norm_cam, z_norm_cam)
        points_norm_cam = torch.stack([x_norm_cam, y_norm_cam, z_norm_cam], dim=2)

        return points_norm_cam, theta, theta_ray

    def project_cam_to_fe_image(
        self, 
        points_cam: torch.Tensor,
        intrinsic_matrix=None,
        distortion_coeffs=None,
    ):
        """
        Project points from 3D camera coordinates to 2D fisheye image pixel coordinates.

        Args:
            points_cam: points in camera coordinates with shape (B, K, 3, N)
                where B is batch size, K is number of detected objects, and N is number considered points.

        Returns:
            tuple:
                - projected_points: 2D fisheye image pixel coordinates of shape (B, K, 3, N)
                - rotation_y: Y-axis rotation angles of shape (B, K, 3, N)
        """
        x_cam, y_cam, z_cam = [points_cam[:, :, i, :].reshape(-1) for i in range(3)]

        K = intrinsic_matrix if intrinsic_matrix is not None else self.intrinsics.matrix_k
        D = distortion_coeffs if distortion_coeffs is not None else self.intrinsics.matrix_d

        # print("K: ", K)
        # print("D: ", D)

        # Calculate radial distances and distorted angles
        radial_dist_2d = torch.sqrt(x_cam**2 + y_cam**2)
        radial_dist_3d = torch.sqrt(x_cam**2 + y_cam**2 + z_cam**2)

        theta = torch.asin(radial_dist_2d / radial_dist_3d)
        theta[z_cam < 0] = torch.tensor(np.pi) - theta[z_cam < 0]
        theta_distorted = self.distortion.apply_distortion(theta, D)

        scale_factor = theta_distorted / radial_dist_2d
        x_distorted_cam = x_cam * scale_factor
        y_distorted_cam = y_cam * scale_factor

        # Project to image coordinates
        x_img = K[0][0] * x_distorted_cam + K[0][2]
        y_img = K[1][1] * y_distorted_cam + K[1][2]
        points_img = torch.stack([x_img, y_img])

        # Calculate rotation angles
        x_norm = x_distorted_cam * torch.tan(theta) / theta_distorted
        z_norm = torch.ones_like(x_norm)
        rotation_y = torch.atan2(x_norm, z_norm)

        output_shape = points_cam.shape
        rotation_y = rotation_y.reshape((output_shape[0], output_shape[1], output_shape[3]))
        projected_points = torch.zeros((output_shape[0], output_shape[1], 2, output_shape[3]),
                                       dtype=points_img.dtype,
                                       device=points_img.device)
        for i in range(2):
            points = points_img[i].reshape((output_shape[0], output_shape[1], 1, output_shape[3]))
            projected_points[:, :, i:i + 1, :] = points

        return projected_points, rotation_y

