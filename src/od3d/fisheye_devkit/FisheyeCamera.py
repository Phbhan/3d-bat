from typing import Union, List, Tuple, Optional

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
        # Largest theta for which the fitted distortion polynomial is still
        # monotonic (i.e. a valid, invertible fisheye mapping). Points whose
        # true angle from the optical axis exceeds this are outside the
        # lens's calibrated FOV and must not be projected with the raw
        # polynomial (see cam_distortion_handler.compute_theta_max).
        self.theta_max = self.distortion.compute_theta_max(self.intrinsics.matrix_d)


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

    def project_cam_to_fe_image(self, points_cam: torch.Tensor):
        """
        Project points from 3D camera coordinates to 2D fisheye image pixel coordinates.

        Args:
            points_cam: points in camera coordinates with shape (B, K, 3, N)
                where B is batch size, K is number of detected objects, and N is number considered points.

        Returns:
            tuple:
                - projected_points: 2D fisheye image pixel coordinates of shape (B, K, 2, N)
                - rotation_y: Y-axis rotation angles of shape (B, K, N)
                - valid_mask: bool tensor of shape (B, K, N). True where the
                  point's true angle from the optical axis is within the
                  lens's calibrated FOV (theta <= self.theta_max), i.e. the
                  pixel coordinates are trustworthy. False means the point
                  is outside the valid mapping range: apply_distortion was
                  clamped for it and the returned pixel is *not* a faithful
                  projection (it should not be treated as "just off-frame").
        """
        x_cam, y_cam, z_cam = [points_cam[:, :, i, :].reshape(-1) for i in range(3)]

        # Calculate radial distances and distorted angles
        radial_dist_2d = torch.sqrt(x_cam**2 + y_cam**2)
        radial_dist_3d = torch.sqrt(x_cam**2 + y_cam**2 + z_cam**2)

        theta = torch.asin(radial_dist_2d / radial_dist_3d)
        theta[z_cam < 0] = torch.tensor(np.pi) - theta[z_cam < 0]

        # Points whose true angle exceeds theta_max are outside the range
        # where the fitted polynomial is monotonic/invertible. Clamp before
        # distorting so we never evaluate the polynomial past its turning
        # point (which would fold far-outside points back near the center).
        valid_mask = theta <= self.theta_max
        theta_clamped = torch.clamp(theta, max=self.theta_max)

        theta_distorted = self.distortion.apply_distortion(theta_clamped, self.intrinsics.matrix_d)

        # Guard against blow-up as a ray approaches the optical axis
        # (x_cam, y_cam -> 0), where radial_dist_2d -> 0.
        safe_radial_dist_2d = torch.clamp(radial_dist_2d, min=1e-6)
        scale_factor = theta_distorted / safe_radial_dist_2d
        x_distorted_cam = x_cam * scale_factor
        y_distorted_cam = y_cam * scale_factor

        # Project to image coordinates
        x_img = self.intrinsics.matrix_k[0][0] * x_distorted_cam + self.intrinsics.matrix_k[0][2]
        y_img = self.intrinsics.matrix_k[1][1] * y_distorted_cam + self.intrinsics.matrix_k[1][2]
        points_img = torch.stack([x_img, y_img])

        # Calculate rotation angles
        x_norm = x_distorted_cam * torch.tan(theta_clamped) / theta_distorted
        z_norm = torch.ones_like(x_norm)
        rotation_y = torch.atan2(x_norm, z_norm)

        output_shape = points_cam.shape
        rotation_y = rotation_y.reshape((output_shape[0], output_shape[1], output_shape[3]))
        valid_mask = valid_mask.reshape((output_shape[0], output_shape[1], output_shape[3]))
        projected_points = torch.zeros((output_shape[0], output_shape[1], 2, output_shape[3]),
                                       dtype=points_img.dtype,
                                       device=points_img.device)
        for i in range(2):
            points = points_img[i].reshape((output_shape[0], output_shape[1], 1, output_shape[3]))
            projected_points[:, :, i:i + 1, :] = points

        return projected_points, rotation_y, valid_mask

    def theta_of_camera_points(self, points_cam: torch.Tensor) -> torch.Tensor:
        """
        True (undistorted) angle between each point's ray and the optical
        axis, using the same convention as project_cam_to_fe_image
        (angles beyond pi/2 for points behind the camera get folded to
        pi - theta). Shape in, shape out (no batching/flattening assumed
        beyond broadcasting on the last dim).

        Args:
            points_cam: (3, N) or (..., 3, N) camera-frame points.

        Returns:
            theta: (..., N)
        """
        x_cam, y_cam, z_cam = points_cam[..., 0, :], points_cam[..., 1, :], points_cam[..., 2, :]
        radial_dist_2d = torch.sqrt(x_cam**2 + y_cam**2)
        radial_dist_3d = torch.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
        theta = torch.asin(torch.clamp(radial_dist_2d / torch.clamp(radial_dist_3d, min=1e-12), max=1.0))
        theta = torch.where(z_cam < 0, torch.tensor(np.pi, dtype=theta.dtype, device=theta.device) - theta, theta)
        return theta

    def clip_box_edges_to_fov(
        self,
        corners_cam: torch.Tensor,
        edges: List[Tuple[int, int]],
        num_bisect: int = 30,
    ):
        """
        Project a box's edges into the fisheye image, clipping any edge
        that crosses out of the calibrated FOV instead of letting it
        fold/distort back into the frame.

        For each edge:
            - both endpoints inside the FOV  -> project both normally.
            - one endpoint outside           -> bisect along the edge in
              camera space for the point where theta == theta_max, project
              *that* point instead of the raw out-of-FOV corner.
            - both endpoints outside         -> drop the edge (None).

        Args:
            corners_cam: (3, num_corners) camera-frame corner points for a
                single box (e.g. the 8 bbox corners, not the center).
            edges: list of (i, j) index pairs into corners_cam's last dim.
            num_bisect: bisection iterations used to locate the FOV crossing.

        Returns:
            List parallel to `edges`. Each entry is either
            (pixel_xy_i, pixel_xy_j) as (2,) numpy arrays, or None if the
            whole edge falls outside the FOV and should not be drawn.
        """
        thetas = self.theta_of_camera_points(corners_cam)  # (num_corners,)
        valid = thetas <= self.theta_max

        def project_point(p: torch.Tensor) -> np.ndarray:
            p_batched = p.view(1, 1, 3, 1).to(corners_cam.dtype)
            img_pt, _, _ = self.project_cam_to_fe_image(p_batched)
            return img_pt.view(2).detach().cpu().numpy()

        def theta_at(p: torch.Tensor) -> torch.Tensor:
            return self.theta_of_camera_points(p.view(3, 1)).view(())

        def bisect_to_boundary(p_in: torch.Tensor, p_out: torch.Tensor) -> torch.Tensor:
            lo, hi = 0.0, 1.0
            for _ in range(num_bisect):
                mid = (lo + hi) / 2.0
                p_mid = p_in + mid * (p_out - p_in)
                if theta_at(p_mid) <= self.theta_max:
                    lo = mid
                else:
                    hi = mid
            return p_in + lo * (p_out - p_in)

        results = []
        for (i, j) in edges:
            pi, pj = corners_cam[:, i], corners_cam[:, j]
            vi, vj = bool(valid[i]), bool(valid[j])

            if vi and vj:
                results.append((project_point(pi), project_point(pj)))
            elif not vi and not vj:
                results.append(None)
            else:
                if vi:
                    p_in, p_out = pi, pj
                else:
                    p_in, p_out = pj, pi
                p_clip = bisect_to_boundary(p_in, p_out)
                p_in_img = project_point(p_in)
                p_clip_img = project_point(p_clip)
                results.append((p_in_img, p_clip_img) if vi else (p_clip_img, p_in_img))

        return results