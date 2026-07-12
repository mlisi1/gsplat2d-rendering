"""Point-cloud / disk display modes: not real rasterizer kernel modes (the
CUDA kernel only ever evaluates a Gaussian falloff -- see rasterizer.py's
module docstring), so both are approximated by overriding scale/opacity on
the already-gathered candidate set right before the rasterizer call, same
"narrow-phase transform on the candidate set" shape as candidate_filters.py.
"""
from __future__ import annotations

from typing import Literal

import torch

RenderMode = Literal["gaussian", "point", "disk"]


def apply_render_mode(
    mode: RenderMode, scales: torch.Tensor, opacity: torch.Tensor, point_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """"gaussian" (default): scales/opacity untouched, current behavior.
    "disk": opacity forced to 1.0, scale left at the splat's trained
    extent -- shows each surfel's full footprint opaquely instead of
    alpha-blended, useful for inspecting coverage/overlap.
    "point": scale forced to a small isotropic `point_size` (world units --
    scene-scale dependent, tune per model) and opacity forced to 1.0,
    approximating a point-cloud view.

    Applied after LOD proxy splats are appended, so it affects proxies too
    -- a global display toggle, not a fine/coarse-specific one."""
    if mode == "gaussian":
        return scales, opacity
    opacity = torch.ones_like(opacity)
    if mode == "point":
        scales = torch.full_like(scales, point_size)
    return scales, opacity
