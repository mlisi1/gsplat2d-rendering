"""Depth map -> world-space points -> per-pixel surface normal, via central
differences on the unprojected point grid. Ported from 2D-GS's reference
implementation (Huang et al. 2024, utils/point_utils.py) so callers don't
need the training-only 2d_gaussian_splatting repo just for this -- it's
pinhole unprojection + a cross product, not project-specific.
"""
from __future__ import annotations

import torch

from gsplat2d_rendering.camera import Camera


def depth_to_points(camera: Camera, depth: torch.Tensor) -> torch.Tensor:
    """depth: any shape with camera.height * camera.width elements (e.g. the
    [1, H, W] depth SplatRenderer/Renderer produce). Returns [H, W, 3]
    world-space points, one per pixel."""
    device = camera.world_view_transform.device
    W, H = camera.width, camera.height
    c2w = camera.world_view_transform.T.inverse()

    # NDC -> pixel-space, so intrins below comes out as a standard pixel
    # intrinsics matrix rather than an NDC-space one.
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, W / 2],
        [0, H / 2, 0, H / 2],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=device).T
    projection_matrix = c2w.T @ camera.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T

    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device=device).float(),
        torch.arange(H, device=device).float(),
        indexing="xy",
    )
    pixels = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = pixels @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depth.reshape(-1, 1) * rays_d + rays_o
    return points.reshape(H, W, 3)


def depth_to_normal(camera: Camera, depth: torch.Tensor) -> torch.Tensor:
    """Per-pixel surface normal from a depth map. Border pixels (no
    both-side neighbor for the central difference) are left zero rather than
    one-sided-differenced -- matches the reference implementation, not an
    oversight."""
    points = depth_to_points(camera, depth)
    output = torch.zeros_like(points)
    dx = points[2:, 1:-1] - points[:-2, 1:-1]
    dy = points[1:-1, 2:] - points[1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output
