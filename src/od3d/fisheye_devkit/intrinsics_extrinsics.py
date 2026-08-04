import torch
import numpy as np


def generate_rotation_matrix(euler_angles: torch.Tensor) -> torch.Tensor:
    """
    Generate a batch of 3D rotation matrices from euler angles.

    Args:
        euler_angles: Tensor of shape (batch_size, 3) containing rotation angles
                     for x, y, z axes in radians.

    Returns:
        Tensor of shape (batch_size, 3, 3) containing rotation matrices.
    """
    # Extract individual rotation angles
    rot_x = euler_angles[:, 0]  # roll
    rot_y = euler_angles[:, 1]  # pitch
    rot_z = euler_angles[:, 2]  # yaw

    # Compute trigonometric functions once
    cos_x, sin_x = torch.cos(rot_x), torch.sin(rot_x)
    cos_y, sin_y = torch.cos(rot_y), torch.sin(rot_y)
    cos_z, sin_z = torch.cos(rot_z), torch.sin(rot_z)

    # Initialize rotation matrix
    batch_size = euler_angles.shape[0]
    rot_matrix = torch.zeros(batch_size, 3, 3, device=euler_angles.device)

    # Fill rotation matrix elements
    # First row
    rot_matrix[:, 0, 0] = cos_y * cos_z
    rot_matrix[:, 0, 1] = -cos_y * sin_z
    rot_matrix[:, 0, 2] = sin_y

    # Second row
    rot_matrix[:, 1, 0] = sin_x * sin_y * cos_z + cos_x * sin_z
    rot_matrix[:, 1, 1] = -sin_x * sin_y * sin_z + cos_x * cos_z
    rot_matrix[:, 1, 2] = -sin_x * cos_y

    # Third row
    rot_matrix[:, 2, 0] = -cos_x * sin_y * cos_z + sin_x * sin_z
    rot_matrix[:, 2, 1] = cos_x * sin_y * sin_z + sin_x * cos_z
    rot_matrix[:, 2, 2] = cos_x * cos_y

    return rot_matrix

class IntrinsicParameters:
    """Handles camera intrinsic parameters and related operations."""

    def __init__(self, device: torch.device):
        self.device = device
        self.matrix_k = None
        self.matrix_d = None

    def set_from_calibration(self, calib_k: np.ndarray, calib_d: np.ndarray) -> None:
        """Initialize from calibration data."""
        self.matrix_k = torch.eye(3, 3).to(self.device)
        if len(calib_k) == 4:
            self.matrix_k[0, 0] = calib_k[0]  # fx
            self.matrix_k[0, 2] = calib_k[1]  # cx
            self.matrix_k[1, 1] = calib_k[2]  # fy
            self.matrix_k[1, 2] = calib_k[3]  # cy
        else:
            self.matrix_k[0, 0] = calib_k[0]
            self.matrix_k[0, 1] = calib_k[1]
            self.matrix_k[0, 2] = calib_k[2]
            self.matrix_k[1, 0] = calib_k[3]
            self.matrix_k[1, 1] = calib_k[4]
            self.matrix_k[1, 2] = calib_k[5]
            self.matrix_k[2, 0] = calib_k[6]
            self.matrix_k[2, 1] = calib_k[7]
            self.matrix_k[2, 2] = calib_k[8]
        self.matrix_d = torch.tensor(calib_d).to(self.device)

    @classmethod
    def create_default_rear(cls, device: torch.device) -> 'IntrinsicParameters':
        """Create default rear camera intrinsic parameters."""
        params = cls(device)
        matrix_k = np.array([
            [304.007121, 0.0, 638.469054],
            [0.0, 304.078429, 399.956311],
            [0.0, 0.0, 1.0]
        ])
        matrix_d = np.array([0.138281, 0.025172, -0.030963, 0.005019])
        params.matrix_k = torch.tensor(matrix_k).to(device)
        params.matrix_d = torch.tensor(matrix_d).to(device)
        return params



class ExtrinsicParameters:
    """Handles camera extrinsic parameters and related operations."""

    def __init__(self, device: torch.device):
        self.device = device
        self.matrix_r = None
        self.vector_t = None
        self.rt_matrix = None

    def set_from_calibration(self, matrix_r: np.ndarray, vector_t: np.ndarray) -> None:
        """Initialize from calibration data."""
        self.matrix_r = torch.tensor(matrix_r).view(3, 3).to(self.device)
        self.vector_t = torch.tensor(vector_t).to(self.device)
        self._create_rt_matrix()

    def _create_rt_matrix(self) -> None:
        """Create the 4x4 transformation matrix."""
        self.rt_matrix = torch.eye(4, 4).to(self.device)
        self.rt_matrix[:3, :3] = self.matrix_r
        self.rt_matrix[:3, 3] = self.vector_t

    @classmethod
    def create_default_rear(cls, device: torch.device) -> 'ExtrinsicParameters':
        """Create default rear camera extrinsic parameters."""
        params = cls(device)
        matrix_r = np.array([
            [-0.015964651480317116, 0.99987155199050903, 0.0014207595959305763],
            [0.58267718553543091, 0.010458142496645451, -0.81263637542724609],
            [-0.81254684925079346, -0.012145611457526684, -0.58276933431625366]
        ])
        vector_t = np.array([-0.037088312208652496, 1.5999122858047485, -1.2545629739761353])
        params.set_from_calibration(matrix_r, vector_t)
        return params
