# SPDX-License-Identifier: Apache-2.0
"""Leaf-level LOD selection + coarse-leaf proxy splats for SplatRenderer. Split out from rasterizer.py to keep that file focused on
the core per-frame render flow -- these are free functions operating on the
proxy tensors SplatRenderer loads from Octree.proxy_* at construction time,
not methods, since they don't need any other renderer state.
"""
from __future__ import annotations

import torch

from gsplat2d_rendering.camera import Camera
from gsplat2d_rendering.culling import visible_point_mask_screen_size_torch


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
    focal_x, focal_y = camera.fx, camera.fy
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


def raw_proxy_fields(proxy_xyz, proxy_opacity, proxy_scale, proxy_rotation, proxy_features_dc,
                     sh_rest: int, dtype: torch.dtype):
    """Converts lod.py's activated proxies (opacity in [0, 1], world-unit
    scales) into the raw form the rasterizer takes, once at construction:
    log scales, opacity logits, and SH rows with only the DC term set
    (proxies render view-independent). Same dtype as the model, so the
    per-frame concatenation is a plain cat."""
    opacity = proxy_opacity.clamp(1e-6, 1.0 - 1e-6)
    return (
        proxy_xyz,
        torch.log(opacity / (1.0 - opacity)).to(dtype),
        torch.log(proxy_scale.clamp_min(1e-12)).to(dtype),
        proxy_rotation.to(dtype),
        proxy_features_dc.to(dtype),
        proxy_xyz.new_zeros((proxy_xyz.shape[0], sh_rest, 3), dtype=dtype),
    )


def append_proxies(leaf_coarse: torch.Tensor | None, raw: tuple, proxies: tuple | None) -> tuple:
    """Concatenates coarse leaves' raw proxy splats (see raw_proxy_fields)
    onto the gathered raw candidate tensors."""
    if leaf_coarse is None or proxies is None:
        return raw
    idx = torch.nonzero(leaf_coarse, as_tuple=True)[0]
    if idx.numel() == 0:
        return raw
    return tuple(torch.cat([field, proxy[idx].to(field.dtype)], dim=0)
                 for field, proxy in zip(raw, proxies))
