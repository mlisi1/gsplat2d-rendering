"""Camera pose -> view/projection matrices, in the exact convention
diff-surfel-rasterization's CUDA kernel expects: row-major matrices,
transposed relative to the textbook column-vector form
(`world_view_transform = w2c.T`, `full_proj_transform = (w2c @ P).T`) --
this is the standard Gaussian-Splatting-family camera convention, required
for the kernel to interpret the matrices correctly, not a choice made here.

Cameras use the optical-frame axis convention (x-right, y-down, z-forward),
matching OpenCV/COLMAP -- the convention every Gaussian-Splatting training
pipeline stores its poses in. Whatever produces a pose for `Camera` must
resolve it to *this* axis convention itself (e.g. a ROS z-forward/x-right/
y-down optical frame lookup, or a robot base-frame pose composed with a
fixed camera-mount transform) -- this module has no opinion on where poses
come from, only what it does with one once it has it.

Principal-point offset is NOT modeled: the projection matrix is built from
fov alone (derived from fx/fy), implicitly assuming cx=width/2, cy=height/2.
This is a limitation of the upstream CUDA rasterizer itself (its projection
matrix has no cx/cy term).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

ZNEAR_DEFAULT = 0.01
ZFAR_DEFAULT = 100.0


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics. `fx`/`fy` are focal lengths in pixels."""
    width: int
    height: int
    fx: float
    fy: float

    @property
    def fov_x(self) -> float:
        return 2.0 * math.atan(self.width / (2.0 * self.fx))

    @property
    def fov_y(self) -> float:
        return 2.0 * math.atan(self.height / (2.0 * self.fy))

    @classmethod
    def from_fov(cls, width: int, height: int, fov_x: float, fov_y: float | None = None) -> "Intrinsics":
        """fov_y defaults to fov_x scaled by the aspect ratio (square pixels)."""
        if fov_y is None:
            fov_y = fov_x * height / width
        fx = width / (2.0 * math.tan(fov_x * 0.5))
        fy = height / (2.0 * math.tan(fov_y * 0.5))
        return cls(width=width, height=height, fx=fx, fy=fy)


@dataclass
class Camera:
    width: int
    height: int
    fov_x: float
    fov_y: float
    world_view_transform: torch.Tensor   # [4, 4]
    full_proj_transform: torch.Tensor    # [4, 4]
    camera_center: torch.Tensor          # [3]

    @staticmethod
    def from_w2c(
        r_w2c: np.ndarray, t_w2c: np.ndarray, intrinsics: Intrinsics,
        znear: float = ZNEAR_DEFAULT, zfar: float = ZFAR_DEFAULT, device: str = "cuda",
    ) -> "Camera":
        """r_w2c: [3, 3] world-to-camera rotation. t_w2c: [3] world-to-camera
        translation (i.e. p_cam = r_w2c @ p_world + t_w2c). Both already in
        the optical-frame axis convention -- see module docstring."""
        camera_center = -r_w2c.T @ t_w2c
        return Camera._build(r_w2c, t_w2c, camera_center, intrinsics, znear, zfar, device)

    @staticmethod
    def from_c2w(
        r_c2w: np.ndarray, t_c2w: np.ndarray, intrinsics: Intrinsics,
        znear: float = ZNEAR_DEFAULT, zfar: float = ZFAR_DEFAULT, device: str = "cuda",
    ) -> "Camera":
        """r_c2w: [3, 3] camera-to-world rotation. t_c2w: [3] camera position
        in world coordinates. Both already in the optical-frame axis
        convention -- see module docstring."""
        r_w2c = r_c2w.T
        t_w2c = -r_w2c @ t_c2w
        return Camera._build(r_w2c, t_w2c, t_c2w, intrinsics, znear, zfar, device)

    @staticmethod
    def from_c2w_matrix(
        c2w: np.ndarray, intrinsics: Intrinsics,
        znear: float = ZNEAR_DEFAULT, zfar: float = ZFAR_DEFAULT, device: str = "cuda",
    ) -> "Camera":
        """c2w: [4, 4] camera-to-world matrix (rotation in [:3,:3],
        translation in [:3,3]), the common NeRF/3DGS pose-file layout."""
        return Camera.from_c2w(c2w[:3, :3], c2w[:3, 3], intrinsics, znear, zfar, device)

    @staticmethod
    def look_at(
        eye: np.ndarray, target: np.ndarray, intrinsics: Intrinsics,
        up: np.ndarray = np.array([0.0, 0.0, 1.0]),
        znear: float = ZNEAR_DEFAULT, zfar: float = ZFAR_DEFAULT, device: str = "cuda",
    ) -> "Camera":
        """Convenience constructor for orbit/free-fly viewers: camera at
        `eye` looking toward `target`, with `up` (default +Z, override to
        (0,1,0) for a Y-up scene) resolving the remaining roll freedom."""
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)

        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        down = np.cross(forward, right)

        r_c2w = np.stack([right, down, forward], axis=1)  # columns = camera axes in world coords
        return Camera.from_c2w(r_c2w, eye, intrinsics, znear, zfar, device)

    @staticmethod
    def _build(
        r_w2c: np.ndarray, t_w2c: np.ndarray, camera_center: np.ndarray,
        intrinsics: Intrinsics, znear: float, zfar: float, device: str,
    ) -> "Camera":
        world_view = _world_to_view_transposed(r_w2c, t_w2c)
        full_proj = world_view @ _projection_transposed(intrinsics.fov_x, intrinsics.fov_y, znear, zfar)
        return Camera(
            width=intrinsics.width,
            height=intrinsics.height,
            fov_x=intrinsics.fov_x,
            fov_y=intrinsics.fov_y,
            world_view_transform=torch.tensor(world_view, dtype=torch.float32, device=device),
            full_proj_transform=torch.tensor(full_proj, dtype=torch.float32, device=device),
            camera_center=torch.tensor(camera_center, dtype=torch.float32, device=device),
        )


def _world_to_view_transposed(r_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = r_w2c
    m[:3, 3] = t_w2c
    return m.T


def _projection_transposed(fov_x: float, fov_y: float, znear: float, zfar: float) -> np.ndarray:
    tan_half_x = math.tan(fov_x * 0.5)
    tan_half_y = math.tan(fov_y * 0.5)
    top, bottom = tan_half_y * znear, -tan_half_y * znear
    right, left = tan_half_x * znear, -tan_half_x * znear

    p = np.zeros((4, 4), dtype=np.float64)
    p[0, 0] = 2.0 * znear / (right - left)
    p[1, 1] = 2.0 * znear / (top - bottom)
    p[0, 2] = (right + left) / (right - left)
    p[1, 2] = (top + bottom) / (top - bottom)
    p[3, 2] = 1.0
    p[2, 2] = zfar / (zfar - znear)
    p[2, 3] = -(zfar * znear) / (zfar - znear)
    return p.T
