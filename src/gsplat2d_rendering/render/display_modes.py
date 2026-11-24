# SPDX-License-Identifier: Apache-2.0
"""Point-cloud / disk display modes: not rasterizer modes (the kernel only
ever evaluates a Gaussian falloff), so both override the raw scale/opacity
of the already-gathered candidate set right before the rasterizer call,
same "narrow-phase transform on the candidate set" shape as
candidate_filters.py.
"""
from __future__ import annotations

import math
from typing import Literal

import torch

RenderMode = Literal["gaussian", "point", "disk"]

# sigmoid(12) ~= 0.999994: above the rasterizer's alpha_max clamp, so the
# disc renders as opaque as the compositor allows.
_OPAQUE_LOGIT = 12.0


def apply_render_mode(
    mode: RenderMode, log_scales: torch.Tensor, opacity_logits: torch.Tensor, point_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Works on raw values (log scales, opacity logits).
    "gaussian" (default): untouched.
    "disk": opaque, scale left at the splat's trained extent -- shows each
    surfel's full footprint instead of alpha-blended.
    "point": scale forced to a small isotropic `point_size` (world units --
    scene-scale dependent, tune per model), opaque, approximating a
    point-cloud view.

    Applied after LOD proxy splats are appended, so it affects proxies too."""
    if mode == "gaussian":
        return log_scales, opacity_logits
    opacity_logits = torch.full_like(opacity_logits, _OPAQUE_LOGIT)
    if mode == "point":
        log_scales = torch.full_like(log_scales, math.log(point_size))
    return log_scales, opacity_logits
