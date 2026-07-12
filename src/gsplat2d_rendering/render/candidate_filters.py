"""Narrow-phase filters applied to the already leaf-gathered (or brute-force
full-model) candidate set, right before activation. Split out of
rasterizer.py's core render() loop -- the filter list has grown from two
(culling_narrow_phase, screen_size_culling) to five (+ sparsity, crop-box
bounds, live opacity threshold), and inlining all of them was pushing that
file past this repo's soft line-count cap.

Each filter is opt-in (its disabled value is a no-op) and independent of
the others; `apply_candidate_filters` runs them in a fixed order --
cheapest/most-reducing first -- and laps the profiler after each one under
the same stage names rasterizer.py has always used, so existing
get_last_timings()/get_profiling_stats() consumers see no renamed/merged
stages from this split.

`culling_narrow_phase`/`screen_size_culling` stay gated on `has_leaf_gather`
(only meaningful once an octree broad phase has already reduced N -> K --
see rasterizer.py's class docstring); `sparsity`/`bounds`/`min_opacity` are
plain display/debug filters with no such precondition, so they run
unconditionally, octree or not.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

from gsplat2d_rendering.camera import Camera
from gsplat2d_rendering.culling import (
    visible_point_mask_bounds_torch,
    visible_point_mask_exact_torch,
    visible_point_mask_screen_size_torch,
)

_XYZ, _OPACITY, _SCALING, _ROTATION, _FEATURES_DC, _FEATURES_REST = range(6)

Bounds = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def apply_candidate_filters(
    raw: tuple[torch.Tensor, ...], camera: Camera, has_leaf_gather: bool, lap: Callable[[str], None], *,
    narrow_phase: bool, margin: float,
    screen_size: bool, min_pixel_radius: float,
    sparsity: int,
    bounds: Bounds | None,
    min_opacity: float,
) -> tuple[torch.Tensor, ...]:
    if sparsity > 1:
        raw = tuple(field[::sparsity] for field in raw)
    lap("sparsity")

    if narrow_phase and has_leaf_gather:
        # Exact per-point frustum test on the already leaf-gathered
        # candidates (K-sized, not N). Operates on raw (possibly fp16) xyz
        # upcast just for the test, same reasoning as
        # GaussianModel.render_fields: touch only the candidates.
        keep = visible_point_mask_exact_torch(raw[_XYZ].float(), camera.full_proj_transform, margin=margin)
        raw = tuple(field[keep] for field in raw)
    lap("narrow_cull")

    if screen_size and has_leaf_gather:
        focal_x = camera.width / (2.0 * math.tan(camera.fov_x * 0.5))
        focal_y = camera.height / (2.0 * math.tan(camera.fov_y * 0.5))
        keep = visible_point_mask_screen_size_torch(
            raw[_XYZ].float(), torch.exp(raw[_SCALING].float()), camera.world_view_transform,
            focal_x, focal_y, min_pixel_radius=min_pixel_radius,
        )
        raw = tuple(field[keep] for field in raw)
    lap("screen_size_cull")

    if bounds is not None:
        keep = visible_point_mask_bounds_torch(raw[_XYZ].float(), bounds)
        raw = tuple(field[keep] for field in raw)
    lap("bounds_cull")

    if min_opacity > 0.0:
        # Non-destructive per-frame equivalent of compression.py's
        # prune_low_opacity -- that one rebuilds the model and invalidates
        # any octree cache built at a different threshold; this is just a
        # mask, safe to change every frame (e.g. a GUI slider).
        keep = torch.sigmoid(raw[_OPACITY].float().squeeze(-1)) >= min_opacity
        raw = tuple(field[keep] for field in raw)
    lap("opacity_cull")

    return raw
