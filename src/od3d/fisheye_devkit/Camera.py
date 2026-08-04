import torch
import numpy as np
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import json

from src.od3d.fisheye_devkit.intrinsics_extrinsics import IntrinsicParameters, ExtrinsicParameters


class CameraPosition(Enum):
    """Enumeration of possible camera positions."""
    LEFT = 0
    FRONT = 1
    REAR = 2
    RIGHT = 3

    @property
    def rotation_y_offset(self) -> float:
        """Get the Y-rotation offset for this camera position."""
        offsets = {
            self.LEFT: 0,
            self.FRONT: -np.pi / 2,
            self.REAR: np.pi / 2,
            self.RIGHT: np.pi
        }
        return offsets[self]


@dataclass
class CameraCalibration:
    matrix_d: np.ndarray  # Distortion coefficients
    matrix_k: np.ndarray  # Camera intrinsic matrix
    matrix_r: np.ndarray  # Rotation matrix
    vector_t: np.ndarray  # Translation vector


class Camera:
    """Main camera class integrating all camera-related functionality."""

    def __init__(
            self,
            batch_size: int = 32,
            num_objects: int = 64,
            num_concern_kps: int = 9,
            device: torch.device=torch.device("cpu"),
            camera_pos: str = "",
            calib_path: str = ""
    ):
        self.device = device
        self.batch_size = batch_size
        self.num_objects = num_objects
        self.num_concern_kps = num_concern_kps

        self.rot_y_offset = None
        self.identity_3d_bbox_matrix = None

        self.intrinsics = IntrinsicParameters(device)
        self.extrinsics = ExtrinsicParameters(device)

        if calib_path:
            self.init_from_calibration_file(calib_path, camera_pos)
        else:
            self._init_defaults()

        self._init_transformation_matrices()

    def init_from_calibration_file(self, path: str, camera_position: str) -> None:
        calibration = self._load_calibration(path)
        pos = CameraPosition[camera_position.upper()]

        self.intrinsics.set_from_calibration(calibration.matrix_k[pos.value], calibration.matrix_d[pos.value])
        self.extrinsics.set_from_calibration(calibration.matrix_r[pos.value], calibration.vector_t[pos.value])
        self.rot_y_offset = pos.rotation_y_offset

    def _init_defaults(self) -> None:
        print("Using default camera (rear cam)")
        self.intrinsics = IntrinsicParameters.create_default_rear(self.device)
        self.extrinsics = ExtrinsicParameters.create_default_rear(self.device)

    def _init_transformation_matrices(self) -> None:
        identity = torch.eye(4, 4).to(self.device).to(torch.float64)
        self.identity_matrix = identity.view(1, 4, 4, 1)
        self.identity_3d_bbox_matrix = self.identity_matrix.repeat(
            self.batch_size * self.num_concern_kps, 1, 1, self.num_concern_kps
        )

    @staticmethod
    def _load_calibration(path: str) -> CameraCalibration:
        with open(Path(path), 'r') as f:
            data = json.load(f)

        num_cameras = 4
        matrices = {
            'matrixD': np.zeros((num_cameras,  len(data['Items'][0]['matrixD']))),
            'matrixK': np.zeros((num_cameras, len(data['Items'][0]['matrixK']))),
            'matrixR': np.zeros((num_cameras,  len(data['Items'][0]['matrixR']))),
            'vectT': np.zeros((num_cameras,  len(data['Items'][0]['vectT'])))
        }

        for i in range(num_cameras):
            for key in matrices:
                matrices[key][i] = data['Items'][i][key]

        return CameraCalibration(
            matrices['matrixD'],
            matrices['matrixK'],
            matrices['matrixR'],
            matrices['vectT']
        )

    def creat_homo_bbox_3d_corners(
            self, bbox_3d_corners: torch.Tensor, batch_size: int, num_objects: int
    ) -> torch.Tensor:
        """Process 3D bounding box corners matrix to create homogeneous matrix."""
        if num_objects != self.num_objects or batch_size != self.batch_size:
            self.batch_size = batch_size
            self.num_objects = num_objects
            self.identity_3d_bbox_matrix = self.identity_matrix.repeat(
                batch_size * num_objects, 1, 1, self.num_concern_kps
            )

        corners = self.identity_3d_bbox_matrix.clone()
        offsets = bbox_3d_corners.view(batch_size * num_objects, 1, 3, self.num_concern_kps)
        corners[:, :3, 3, :] = offsets.squeeze(1)

        return corners.permute(0, 3, 1, 2).contiguous().view(batch_size * num_objects * self.num_concern_kps, 4, 4)

    def project_world_to_cam(
        self, 
        world_kps: torch.Tensor, 
        rotation: torch.Tensor = None, 
        translation: torch.Tensor = None
    ) -> torch.Tensor:
        """
            Project 3D world coordinates to normal camera coordinates.

            Args:
            world_kps: Array-like object of shape (B, K, 3, N) containing the x, y, z coordinates of world points.
                 B is batch size, K is number of detected objects, and N is number considered points.

        Returns:
            torch.Tensor: Transformed coordinates of shape (B, K, 3, N) .
        """
        # Extract shape information
        batch_size, num_objects, _, num_points = world_kps.shape

        # Reshape world_kps to match the expected format for matrix multiplication
        points = world_kps.permute(0, 1, 3, 2).reshape(-1, 3).t()  # Shape: (3, B * K * N)

        if rotation is None:
            rotation = self.extrinsics.matrix_r

        if translation is None:
            translation = self.extrinsics.vector_t

        # print("points: ", points.shape, points)
        # print("rotation: ", rotation.shape, rotation)
        # print("translation: ", translation.shape, translation)

        # Apply transformation
        camera_points = (
                rotation @ points.to(torch.float64) + translation.unsqueeze(1)
        )  # Shape: (3, B * K * N)

        # Reshape back to original format
        camera_points = camera_points.t().reshape(batch_size, num_objects, num_points, 3).permute(0, 1, 3, 2)

        return camera_points


    def convert_3d_bbox_local2world(
            self, world_position: torch.Tensor, bbox_3d_local: torch.Tensor, dim: torch.Tensor
    ) -> torch.Tensor:
        bboox_3d_world = torch.zeros_like(bbox_3d_local).to(self.device)
        world_position[:, :, 2, 0] = abs(dim[:, :, 0] / 2)     # scaling z

        for point_idx in range(self.num_concern_kps):
            bboox_3d_world[:, :, 0, point_idx] = bbox_3d_local[:, :, 0, point_idx] + world_position[:, :, 0, 0]
            bboox_3d_world[:, :, 1, point_idx] = bbox_3d_local[:, :, 1, point_idx] + world_position[:, :, 1, 0]
            bboox_3d_world[:, :, 2, point_idx] = bbox_3d_local[:, :, 2, point_idx] + world_position[:, :, 2, 0]

        return bboox_3d_world
