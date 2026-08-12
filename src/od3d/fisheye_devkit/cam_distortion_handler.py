import numpy as np
import torch


class DistortionModel:
    """
    Inverse fisheye distortion model.

    Solves:

        cdist = theta * (
            1 + k1*theta² + k2*theta⁴ +
                k3*theta⁶ + k4*theta⁸
        )

    for theta.
    """

    def __init__(
        self,
        device: torch.device,
        max_iterations: int = 15,
        tolerance: float = 1e-8,
    ):
        self.device = device
        self._max_iterations = max_iterations
        self._tolerance = tolerance

    @staticmethod
    def apply_distortion(
        theta: torch.Tensor,
        coeffs: torch.Tensor
    ) -> torch.Tensor:
        """
        Distortion polynomial using Horner's method.

        Args:
            theta: (...,)
            coeffs: (4,) = [k1,k2,k3,k4]

        Returns:
            distorted radius
        """

        t2 = theta * theta

        poly = (
            ((coeffs[3] * t2 + coeffs[2]) * t2 + coeffs[1])
            * t2 + coeffs[0]
        ) * t2 + 1.0

        return theta * poly

    @staticmethod
    def distortion_derivative(
        theta: torch.Tensor,
        coeffs: torch.Tensor
    ) -> torch.Tensor:
        """
        First derivative:

        f(theta)
        = theta
          + k1*theta³
          + k2*theta⁵
          + k3*theta⁷
          + k4*theta⁹

        f'(theta)
        = 1
          + 3*k1*theta²
          + 5*k2*theta⁴
          + 7*k3*theta⁶
          + 9*k4*theta⁸
        """

        t2 = theta * theta
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4

        return (
            1.0
            + 3.0 * coeffs[0] * t2
            + 5.0 * coeffs[1] * t4
            + 7.0 * coeffs[2] * t6
            + 9.0 * coeffs[3] * t8
        )

    def compute_theta_max(
        self,
        coeffs: torch.Tensor,
        search_max: float = np.pi,
        num_samples: int = 4096,
    ) -> torch.Tensor:
        """
        Largest theta for which `apply_distortion` is still monotonically
        increasing given the fitted `coeffs`.

        The fisheye polynomial f(theta) = theta*(1 + k1*theta^2 + ... )
        is only a valid, invertible mapping while f'(theta) > 0. Past that
        point f can turn over and start decreasing, which means a ray far
        outside the calibrated FOV can alias onto a small/near-center pixel
        instead of projecting far off-frame. Callers should clamp theta to
        this bound before calling apply_distortion, and treat rays with a
        raw (unclamped) theta beyond this bound as outside the valid FOV.

        Args:
            coeffs: (4,) = [k1,k2,k3,k4]
            search_max: upper bound of the angle range to search, in
                radians. Defaults to pi (matches the theta clamp already
                used in solve_distortion) since real fisheye lenses can
                exceed a 180 deg FOV — i.e. theta from the optical axis
                approaching pi/2 or beyond is not automatically invalid,
                only the point past which the fitted polynomial itself
                stops being monotonic is.
            num_samples: resolution of the search grid.

        Returns:
            0-d tensor: theta_max, in radians.
        """
        thetas = torch.linspace(
            0.0, search_max, num_samples,
            device=coeffs.device, dtype=coeffs.dtype,
        )
        deriv = self.distortion_derivative(thetas, coeffs)

        invalid = (deriv <= 0).nonzero(as_tuple=True)[0]
        if invalid.numel() == 0:
            return thetas[-1]

        # Back off one sample so theta_max is still comfortably monotonic.
        first_invalid = invalid[0]
        idx = torch.clamp(first_invalid - 1, min=0)
        return thetas[idx]

    def solve_distortion(
        self,
        cdist: torch.Tensor,
        distortion_coeffs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Solve for theta given distorted radius.

        Newton-Raphson:
            theta_{n+1} = theta_n - f/f'

        Args:
            cdist:
                distorted radius tensor

            distortion_coeffs:
                [k1,k2,k3,k4]

        Returns:
            theta
        """

        # Good initial guess for fisheye models
        theta = cdist.clone()

        for _ in range(self._max_iterations):

            distorted = self.apply_distortion(
                theta,
                distortion_coeffs,
            )

            f = distorted - cdist

            if torch.max(torch.abs(f)) < self._tolerance:
                break

            fp = self.distortion_derivative(
                theta,
                distortion_coeffs,
            )

            theta_update = f / (fp + 1e-12)

            theta = theta - theta_update

            # Optional safety clamp
            theta = torch.clamp(
                theta,
                min=0.0,
                max=np.pi,
            )

            if torch.max(torch.abs(theta_update)) < self._tolerance:
                break

        return theta