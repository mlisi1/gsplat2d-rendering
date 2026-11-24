# SPDX-License-Identifier: Apache-2.0
"""Python entry point to the library's own render-only surfel rasterizer
(CUDA core in csrc/core, torch binding in csrc/torch_binding.cpp).

Takes *raw* splat attributes (log scales, opacity logits, unnormalized
quaternions, SH coefficients): activation and SH evaluation happen inside
the kernel, only for splats that survive culling. Attributes other than
positions may be float16; positions are always float32, since half
precision visibly moves splats in large scenes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from gsplat2d_rendering._raster import _C
except ImportError as exc:  # pragma: no cover - depends on the build
    _C = None
    _IMPORT_ERROR = exc


@dataclass
class RasterOutput:
    rgb: torch.Tensor                    # [3, H, W] float32, background composited
    depth: torch.Tensor                  # [H, W] expected camera z, 0 where alpha == 0
    alpha: torch.Tensor                  # [H, W]
    normal: torch.Tensor | None          # [3, H, W] camera space, aux only
    median_depth: torch.Tensor | None    # [H, W], aux only
    rgb8: torch.Tensor | None            # [H, W, 3] uint8, when requested
    num_pairs: int                       # splat-tile pairs, a cost measure


def rasterize(
    means: torch.Tensor, quats: torch.Tensor, log_scales: torch.Tensor,
    opacity_logits: torch.Tensor, features_dc: torch.Tensor, features_rest: torch.Tensor,
    sh_degree: int, *,
    w2c, fx: float, fy: float, cx: float, cy: float, width: int, height: int, campos,
    near_z: float = 0.01, lowpass: bool = True, aux: bool = False,
    exact_tiles: bool = False, rgb8: bool = False, background=(0.0, 0.0, 0.0),
    alpha_min: float = 1.0 / 255.0, alpha_max: float = 0.99, t_min: float = 1e-4,
    max_sigma: float = 0.0, near_crossing: bool = False,
) -> RasterOutput:
    """`w2c`: [3, 4] world-to-camera (x_cam = R x_world + t, OpenCV axes),
    any array-like; `campos`: world camera centre (SH view directions).
    Both are read on the host, so pass CPU values to avoid a sync.
    `sh_degree=-1` means `features_dc` holds final RGB colours.
    `exact_tiles` bins each splat to the tiles its ellipse actually touches
    instead of its bounding box: same image, fewer pairs. `max_sigma > 0`
    also cuts each splat off at that many standard deviations (fewer pairs,
    slightly different image); 0 keeps the exact alpha_min contour.
    `near_crossing` draws the in-front part of discs whose footprint crosses
    the near plane; off (default) culls them, which is faster and matches
    the previous renderer."""
    if _C is None:
        raise ImportError(
            "gsplat2d_rendering._raster._C is not built; run "
            "`pip install --no-build-isolation -e .` in the gsplat2d-rendering checkout"
        ) from _IMPORT_ERROR
    if log_scales.dim() != 2 or log_scales.shape[1] != 2:
        raise ValueError(
            f"log_scales must be [N, 2] (the surfel rasterizer takes exactly 2 per splat), "
            f"got {tuple(log_scales.shape)}"
        )
    w2c_flat = [float(v) for v in torch.as_tensor(w2c, dtype=torch.float64).reshape(-1).tolist()]
    campos_l = [float(v) for v in torch.as_tensor(campos, dtype=torch.float64).reshape(-1).tolist()]
    dtype = quats.dtype
    if sh_degree > 0 and features_rest.shape[1] < (sh_degree + 1) ** 2 - 1:
        raise ValueError(f"sh_degree={sh_degree} needs {(sh_degree + 1) ** 2 - 1} rest "
                         f"coefficients, features_rest has {features_rest.shape[1]}")
    ins = [means.float(), quats, log_scales.to(dtype), opacity_logits.to(dtype),
           features_dc.to(dtype), features_rest.to(dtype)]
    ins = [t.contiguous() for t in ins]
    rgb, depth, alpha, normal, median, out8, pairs = _C.rasterize(
        *ins, int(sh_degree), w2c_flat, [float(fx), float(fy), float(cx), float(cy)],
        int(width), int(height), float(near_z), campos_l, bool(lowpass), bool(aux),
        bool(exact_tiles), bool(near_crossing), bool(rgb8), [float(alpha_min), float(alpha_max), float(t_min), float(max_sigma)],
        [float(b) for b in background],
    )
    return RasterOutput(rgb=rgb, depth=depth, alpha=alpha,
                        normal=normal if aux else None, median_depth=median if aux else None,
                        rgb8=out8 if rgb8 else None, num_pairs=int(pairs.item()))
