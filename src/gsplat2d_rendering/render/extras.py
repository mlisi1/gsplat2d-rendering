# SPDX-License-Identifier: Apache-2.0
"""Depth blending and the optional per-pixel extras (alpha, normal, median
depth) taken from the rasterizer's named outputs.

`depth_ratio` blends expected depth toward median depth, the 2D-GS paper's
own quality/robustness knob (Huang et al. 2024); 0 keeps expected depth
only. It needs the kernel's aux outputs, so the renderer requests them
whenever `depth_ratio > 0` or extras are wanted. `with_extras` only gates
whether the raw tensors are attached to RenderOutput (memory kept on GPU
and, for Renderer callers, a per-frame CPU copy).

Normals are raw camera-space accumulations (weighted sums of unit normals
facing the camera, not renormalized); rotating them to world space is left
to the caller.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from gsplat2d_rendering._raster import RasterOutput


@dataclass
class RenderExtras:
    alpha: torch.Tensor        # [H, W]
    normal: torch.Tensor       # [3, H, W], camera-space, un-rotated
    middepth: torch.Tensor     # [H, W], median depth


def extract_depth_and_extras(
    out: RasterOutput, depth_ratio: float, with_extras: bool,
) -> tuple[torch.Tensor, RenderExtras | None]:
    depth = out.depth
    if depth_ratio > 0.0:
        depth = depth * (1.0 - depth_ratio) + depth_ratio * out.median_depth
    if not with_extras:
        return depth, None
    return depth, RenderExtras(alpha=out.alpha, normal=out.normal, middepth=out.median_depth)
