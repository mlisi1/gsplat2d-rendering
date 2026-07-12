"""allmap channel extraction beyond depth/alpha, and the 2D-GS
median/expected depth blend -- split out of rasterizer.py's core render()
flow (see that file's module docstring for the fixed 7-channel layout this
reads).

`depth_ratio` blending is applied regardless of `with_extras`: the kernel
computes every channel every frame no matter what (see rasterizer.py's
module docstring), so reading channel 5 to blend into the primary `depth`
output costs nothing extra. `with_extras` only gates whether the *raw*
per-channel tensors (alpha/normal/middepth/distortion) get attached to
RenderOutput at all -- that's the real, non-free cost (memory retained on
GPU, and for Renderer callers, copied to CPU every frame).

Both the world-space rotation of `normal` and any further depth blending
beyond the 2D-GS default are left to the caller -- these fields are raw,
un-rotated camera-space kernel output, matching how RenderOutput.rgb/.depth
already leave clamping/byte-casting to Renderer instead of SplatRenderer.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

_DEPTH_OFFSET = 0
_ALPHA_OFFSET = 1
_NORMAL_OFFSET = 2
_MIDDEPTH_OFFSET = 5
_DISTORTION_OFFSET = 6


@dataclass
class RenderExtras:
    alpha: torch.Tensor        # [H, W]
    normal: torch.Tensor       # [3, H, W], camera-space, un-rotated
    middepth: torch.Tensor     # [H, W], median depth
    distortion: torch.Tensor   # [H, W]


def extract_depth_and_extras(
    allmap: torch.Tensor, depth_ratio: float, with_extras: bool,
) -> tuple[torch.Tensor, RenderExtras | None]:
    """allmap: the rasterizer's raw [7, H, W] second output tensor. Returns
    (depth [H, W], extras-or-None)."""
    alpha = allmap[_ALPHA_OFFSET:_ALPHA_OFFSET + 1]
    depth = torch.nan_to_num(
        allmap[_DEPTH_OFFSET:_DEPTH_OFFSET + 1] / alpha, nan=0.0, posinf=0.0, neginf=0.0
    )
    if depth_ratio > 0.0:
        # 2D-GS's own median/expected depth blend (Huang et al. 2024) -- a
        # real quality/robustness tradeoff the paper itself treats as a
        # knob, not a Kestrel-specific concept. depth_ratio=0 (default)
        # reproduces this library's previous expected-depth-only behavior
        # exactly.
        middepth = allmap[_MIDDEPTH_OFFSET:_MIDDEPTH_OFFSET + 1]
        depth = depth * (1.0 - depth_ratio) + depth_ratio * middepth
    depth = depth.squeeze(0)

    if not with_extras:
        return depth, None
    extras = RenderExtras(
        alpha=alpha.squeeze(0),
        normal=allmap[_NORMAL_OFFSET:_NORMAL_OFFSET + 3],
        middepth=allmap[_MIDDEPTH_OFFSET],
        distortion=allmap[_DISTORTION_OFFSET],
    )
    return depth, extras
