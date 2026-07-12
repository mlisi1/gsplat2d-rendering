"""Leaf-level LOD selection + coarse-leaf proxy-splat blending for
SplatRenderer. Split out from rasterizer.py to keep that file focused on
the core per-frame render flow -- these are free functions operating on the
proxy tensors SplatRenderer loads from Octree.proxy_* at construction time,
not methods, since they don't need any other renderer state.
"""
from __future__ import annotations

import math

import torch

from gsplat2d_rendering.camera import Camera
from gsplat2d_rendering.culling import visible_point_mask_screen_size_torch
from gsplat2d_rendering.sh import C0


def lod_split(
    leaf_vis: torch.Tensor | None, camera: Camera,
    proxy_xyz: torch.Tensor | None, leaf_center: torch.Tensor | None, leaf_radius: torch.Tensor | None,
    lod_leaf_pixel_threshold: float,
):
    """Splits frustum-visible leaves into (leaf_fine, leaf_coarse) based on
    each leaf's own projected screen size -- same pinhole approximation as
    screen-size culling, applied to the whole leaf's extent instead of one
    splat's. Returns (leaf_vis, None) unchanged if LOD isn't active this
    frame (no octree, LOD off, or no proxies built)."""
    if leaf_vis is None or proxy_xyz is None:
        return leaf_vis, None
    focal_x = camera.width / (2.0 * math.tan(camera.fov_x * 0.5))
    focal_y = camera.height / (2.0 * math.tan(camera.fov_y * 0.5))
    # cutoff=1.0: leaf_radius is already a literal spatial extent (half the
    # leaf's AABB diagonal), not a per-splat log-scale Gaussian sigma -- the
    # function's default cutoff=3.0 is for the latter.
    leaf_full_detail = visible_point_mask_screen_size_torch(
        leaf_center, leaf_radius, camera.world_view_transform,
        focal_x, focal_y, cutoff=1.0, min_pixel_radius=lod_leaf_pixel_threshold,
    )
    leaf_coarse = leaf_vis & ~leaf_full_detail
    leaf_fine = leaf_vis & leaf_full_detail
    return leaf_fine, leaf_coarse


def append_proxies(
    leaf_coarse: torch.Tensor | None,
    means3D: torch.Tensor, opacity: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor,
    proxy_xyz: torch.Tensor, proxy_opacity: torch.Tensor, proxy_scale: torch.Tensor, proxy_rotation: torch.Tensor,
):
    """Concatenates coarse leaves' proxy splats onto the already-activated
    full-detail tensors -- proxies are pre-activated (see lod.py's
    build_leaf_proxies), so they skip GaussianModel._activate entirely,
    just a gather. Returns the (possibly extended) tensors plus proxy_idx
    (None if LOD isn't active this frame, an index tensor -- possibly
    empty -- otherwise), which the caller needs again for the matching
    color merge."""
    if leaf_coarse is None:
        return means3D, opacity, scales, rotations, None
    proxy_idx = torch.nonzero(leaf_coarse, as_tuple=True)[0]
    if proxy_idx.numel() == 0:
        return means3D, opacity, scales, rotations, proxy_idx
    means3D = torch.cat([means3D, proxy_xyz[proxy_idx]], dim=0)
    opacity = torch.cat([opacity, proxy_opacity[proxy_idx]], dim=0)
    scales = torch.cat([scales, proxy_scale[proxy_idx]], dim=0)
    rotations = torch.cat([rotations, proxy_rotation[proxy_idx]], dim=0)
    return means3D, opacity, scales, rotations, proxy_idx


def blend_proxy_colors(
    colors: torch.Tensor, proxy_idx: torch.Tensor | None, proxy_features_dc: torch.Tensor,
) -> torch.Tensor:
    """Real splats' colors (already SH-evaluated by the caller) get
    proxies' colors appended -- proxies are pre-collapsed to a single DC
    term (build_leaf_proxies), so they use the same degree-0 color formula
    a real degree-0 model would, skipping SH eval/view-direction math
    entirely."""
    if proxy_idx is not None and proxy_idx.numel() > 0:
        proxy_colors = torch.clamp_min(C0 * proxy_features_dc[proxy_idx, 0, :] + 0.5, 0.0)
        colors = torch.cat([colors, proxy_colors], dim=0)
    return colors
