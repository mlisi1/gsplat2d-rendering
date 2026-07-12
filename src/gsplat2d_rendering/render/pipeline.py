"""Narrow public entry point: a built Camera in, RGB (+ depth) out as plain
numpy arrays. No project-specific types -- callers own pose/intrinsics
conventions and image-processing decisions (JPEG quality, distortion,
publishing, etc.); this layer only runs the rasterizer and moves the
result to CPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gsplat2d_rendering.camera import Camera
from gsplat2d_rendering.culling import Octree
from gsplat2d_rendering.model import GaussianModel
from gsplat2d_rendering.render.rasterizer import SplatRenderer


@dataclass
class RenderResult:
    rgb: np.ndarray             # (H, W, 3) uint8
    depth: np.ndarray | None    # (H, W) float32, model-space units
    num_rendered: int           # splats actually rendered this frame (post-culling)


class Renderer:
    """One instance per loaded model. Holds the model + culling/LOD
    configuration; `render()` is the per-frame entry point, taking an
    already-built `Camera` (see camera.py)."""

    def __init__(self, model: GaussianModel, device: str = "cuda",
                 with_depth: bool = True, octree: Octree | None = None,
                 culling_enabled: bool = True, culling_narrow_phase: bool = False,
                 culling_margin: float = 0.0, screen_size_culling: bool = False,
                 screen_size_min_pixels: float = 1.0, octree_lod: bool = False,
                 lod_leaf_pixel_threshold: float = 16.0):
        self.device = device
        self.with_depth = with_depth
        self._rasterizer = SplatRenderer(
            model, device=device, octree=octree, culling_enabled=culling_enabled,
            culling_narrow_phase=culling_narrow_phase,
            culling_margin=culling_margin, screen_size_culling=screen_size_culling,
            screen_size_min_pixels=screen_size_min_pixels, octree_lod=octree_lod,
            lod_leaf_pixel_threshold=lod_leaf_pixel_threshold)

    @property
    def last_visible_count(self) -> int:
        return self._rasterizer.last_visible_count

    def enable_profiling(self) -> None:
        """Turn on per-stage timing collection. Off by default -- see
        get_last_timings()/get_profiling_stats() to read results, and
        profiling.py for why this is opt-in rather than a render() kwarg."""
        self._rasterizer.enable_profiling()

    def disable_profiling(self) -> None:
        self._rasterizer.disable_profiling()

    def get_last_timings(self) -> dict[str, float] | None:
        """Per-stage ms for the most recent render() call (including this
        layer's CPU copy-back), or None if profiling is off."""
        return self._rasterizer.get_last_timings()

    def get_profiling_stats(self) -> dict[str, dict[str, float]]:
        """Per-stage {count, mean_ms, min_ms, max_ms, total_ms} accumulated
        since enable_profiling() or the last reset_profiling()."""
        return self._rasterizer.get_profiling_stats()

    def reset_profiling(self) -> None:
        self._rasterizer.reset_profiling()

    def render(self, camera: Camera) -> RenderResult:
        import torch

        with torch.no_grad():
            output = self._rasterizer.render(camera)
            rgb = (output.rgb.clamp(0., 1.)
                   .permute(1, 2, 0).mul(255).byte().cpu().numpy())
            depth = None
            if self.with_depth:
                depth = output.depth.cpu().numpy().astype(np.float32)
            # .cpu() above already blocks until the copy lands, so this lap
            # is timed honestly without an extra synchronize().
            self._rasterizer.profiler.lap("copy_to_cpu")
        return RenderResult(rgb=rgb, depth=depth, num_rendered=output.num_rendered)
