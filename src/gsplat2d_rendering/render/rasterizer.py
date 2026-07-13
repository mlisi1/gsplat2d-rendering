"""Wraps diff-surfel-rasterization directly: SH evaluation + the CUDA
tile-sort/alpha-composite kernel + extraction of depth (and, opt-in, the
kernel's other extra channels) from its extra-channels output buffer.

The channel layout of the rasterizer's second output tensor (7 channels) is
fixed by the CUDA kernel itself (cuda_rasterizer/auxiliary.h): DEPTH_OFFSET=0
(accumulated depth*weight), ALPHA_OFFSET=1, NORMAL_OFFSET=2..4,
MIDDEPTH_OFFSET=5, DISTORTION_OFFSET=6 -- confirmed against that header, not
assumed. Every channel is computed by the kernel every frame regardless of
what gets extracted (see render/extras.py); `with_extras` only controls
whether normal/middepth/distortion get attached to RenderOutput.

See lod_blend.py for the leaf-level LOD selection / coarse-leaf proxy
blending, candidate_filters.py for the sparsity/crop-box/opacity narrow-
phase filters, and display_modes.py for the point/disk render modes this
class delegates to.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from gsplat2d_rendering._log import info, verbose
from gsplat2d_rendering.camera import Camera
from gsplat2d_rendering.culling import Octree, visible_leaf_mask_torch
from gsplat2d_rendering.model import GaussianModel
from gsplat2d_rendering.render.candidate_filters import Bounds, apply_candidate_filters
from gsplat2d_rendering.render.display_modes import RenderMode, apply_render_mode
from gsplat2d_rendering.render.extras import extract_depth_and_extras
from gsplat2d_rendering.render.lod_blend import append_proxies, blend_proxy_colors, lod_split
from gsplat2d_rendering.render.profiling import Profiler
from gsplat2d_rendering.sh import C0, eval_sh

# Field order for the raw-tensor tuples passed around render() -- matches
# GaussianModel._activate's positional args and GaussianModel's own
# dataclass field order.
_XYZ, _OPACITY, _SCALING, _ROTATION, _FEATURES_DC, _FEATURES_REST = range(6)


@dataclass
class RenderOutput:
    rgb: torch.Tensor       # [3, H, W], float, ~[0, 1]
    depth: torch.Tensor     # [H, W], float, model-space units
    num_rendered: int       # splats actually passed to the rasterizer this frame
    # Only populated when SplatRenderer(with_extras=True); raw, un-rotated/
    # un-blended kernel output -- see render/extras.py.
    alpha: torch.Tensor | None = None        # [H, W]
    normal: torch.Tensor | None = None       # [3, H, W], camera-space
    middepth: torch.Tensor | None = None     # [H, W]
    distortion: torch.Tensor | None = None   # [H, W]


class SplatRenderer:
    """One instance per loaded model -- the model is loaded once, this is
    reused every frame with a new camera. `octree`/`culling_enabled` are
    optional: with no octree, every splat is rendered every frame.

    PRECONDITION when an octree is supplied: `model` must already be
    permuted into that octree's leaf-contiguous order via
    `model.reorder_(torch.from_numpy(octree.flat_indices))` -- the caller's
    job, not this class's, since the model is loaded before the octree
    exists. Without this, leaf j's points would not actually be at
    `model.xyz[node_offsets[j]:node_offsets[j+1]]` and the contiguous-slice
    gather below would silently gather the wrong splats.

    Per-frame gather (_gather_leaf_slices) builds one index tensor covering
    every leaf that passes the frustum test and does a single indexed
    gather per field, instead of boolean-mask indexing the full model --
    `tensor[bool_mask]` has to scan/compact the entire N-length mask
    regardless of how few points survive, a real fixed cost that no amount
    of tightening *what* got included ever reduces. Whether a single big
    gather or a per-leaf loop of smaller gathers wins depends on typical
    leaf count per frame (itself driven by `leaf_max` -- see
    culling/octree.py) -- measure on your own scene before assuming either
    direction is faster."""

    def __init__(self, model: GaussianModel, device: str = "cuda",
                 octree: Octree | None = None, culling_enabled: bool = True,
                 culling_narrow_phase: bool = False, culling_margin: float = 0.0,
                 screen_size_culling: bool = False, screen_size_min_pixels: float = 1.0,
                 octree_lod: bool = False, lod_leaf_pixel_threshold: float = 16.0,
                 with_extras: bool = False, verbose_log: bool = False):
        from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        self._settings_cls = GaussianRasterizationSettings
        self._rasterizer_cls = GaussianRasterizer

        self.model = model
        self.device = device
        self.octree = octree
        self.culling_enabled = culling_enabled
        self.culling_narrow_phase = culling_narrow_phase
        self.culling_margin = culling_margin
        self.screen_size_culling = screen_size_culling
        self.screen_size_min_pixels = screen_size_min_pixels
        self.octree_lod = octree_lod
        self.lod_leaf_pixel_threshold = lod_leaf_pixel_threshold
        # Off by default: alpha/normal/middepth/distortion are computed by
        # the kernel every frame regardless (see module docstring), but
        # attaching them to RenderOutput keeps them alive on GPU and, for
        # Renderer callers, pays a real per-frame CPU copy -- a cost most
        # callers (e.g. a sensor sim only wanting RGB+depth) shouldn't pay.
        self.with_extras = with_extras
        self.background = torch.zeros(3, dtype=torch.float32, device=device)
        self.last_visible_count = model.num_points
        # A plain closure over `device` (the local parameter, not
        # `self.device`) rather than the bound method `self._sync` --
        # `self._sync` would hold a reference back to this very instance via
        # its `__self__`, creating SplatRenderer -> profiler -> sync_fn ->
        # SplatRenderer, a genuine reference cycle that plain refcounting can
        # never free (only Python's cyclic GC can, and only if/when it
        # happens to run). _sync()'s only real dependency is the device
        # string, which is immutable for this instance's lifetime, so
        # capturing it directly avoids the cycle at its root instead of
        # working around it with periodic gc.collect() calls -- verified: a
        # caller that discards and reconstructs many SplatRenderer instances
        # in a tight loop (e.g. Kestrel's chunk streaming) now frees each one
        # via ordinary refcounting the instant the last external reference
        # drops, with no gc involvement needed at all.
        def _sync_fn(_device: str = device) -> None:
            if _device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
        self.profiler = Profiler(sync_fn=_sync_fn)

        self._has_octree = culling_enabled and octree is not None
        self._node_aabbs_gpu = None
        self._node_offsets_gpu = None
        if self._has_octree:
            self._node_aabbs_gpu = torch.from_numpy(octree.node_aabbs).to(device)
            self._node_offsets_gpu = torch.from_numpy(octree.node_offsets).to(device)

        self._proxy_xyz_gpu = None
        self._proxy_scale_gpu = None
        self._proxy_rotation_gpu = None
        self._proxy_opacity_gpu = None
        self._proxy_features_dc_gpu = None
        self._leaf_center_gpu = None
        self._leaf_radius_gpu = None
        if self._has_octree and octree_lod and octree.has_lod:
            self._proxy_xyz_gpu = torch.from_numpy(octree.proxy_xyz).to(device)
            self._proxy_scale_gpu = torch.from_numpy(octree.proxy_scale).to(device)
            self._proxy_rotation_gpu = torch.from_numpy(octree.proxy_rotation).to(device)
            self._proxy_opacity_gpu = torch.from_numpy(octree.proxy_opacity).to(device)
            self._proxy_features_dc_gpu = torch.from_numpy(octree.proxy_features_dc).to(device)
            self._leaf_center_gpu = (self._node_aabbs_gpu[:, :3] + self._node_aabbs_gpu[:, 3:]) * 0.5
            self._leaf_radius_gpu = (
                (self._node_aabbs_gpu[:, 3:] - self._node_aabbs_gpu[:, :3]) * 0.5
            ).amax(dim=-1, keepdim=True)

        # verbose_log routes these startup-summary lines through verbose()
        # instead of info(): constructing one SplatRenderer for a model load
        # is a rare, notable event worth NORMAL visibility, but a caller
        # rebuilding this instance repeatedly (e.g. chunk streaming's
        # per-composition-change rebuild) would otherwise flood NORMAL-level
        # output with one line per rebuild -- SplatRenderer's own
        # constructor-only octree/culling_enabled is exactly why such a
        # caller has to reconstruct rather than mutate in place, see this
        # class's own docstring.
        log_fn = verbose if verbose_log else info
        if self._has_octree:
            log_fn(__name__, f"Culling enabled: {len(octree.node_aabbs):,} leaf nodes")
        elif not culling_enabled:
            log_fn(__name__, "Culling disabled: culling_enabled=False")
        else:
            log_fn(__name__, "Culling disabled: no octree provided")
        if octree_lod:
            if self._proxy_xyz_gpu is not None:
                verbose(__name__, f"LOD enabled: {len(octree.node_aabbs):,} leaf proxies "
                        f"(pixel threshold {lod_leaf_pixel_threshold})")
            else:
                verbose(__name__, "LOD requested but unavailable (no octree, or octree has no proxies)")

    def _gather_leaf_slices(self, leaf_mask: torch.Tensor):
        """Builds one index tensor covering every True leaf's contiguous
        point range in the model's own (reorder_-permuted) order, and does
        a single `tensor[index]` gather per raw field -- touches only the
        visible K points, never the full N.

        `.item()` on the total visible-point count is a deliberate, small
        sync -- building a variable-length index tensor is inherently
        data-dependent, there's no way to avoid some sync here. What
        matters is its payload is O(visible leaf count), a few thousand at
        most, not O(N) -- the thing this whole approach replaces was an
        O(N) sync/scan on *every* frame regardless of visibility."""
        visible = torch.nonzero(leaf_mask, as_tuple=True)[0]
        if visible.numel() == 0:
            z = lambda t: t.new_zeros((0,) + t.shape[1:])
            return (z(self.model.xyz), z(self.model.raw_opacity), z(self.model.raw_scaling),
                    z(self.model.raw_rotation), z(self.model.features_dc), z(self.model.features_rest))
        starts = self._node_offsets_gpu[visible]
        ends = self._node_offsets_gpu[visible + 1]
        lengths = ends - starts
        total = int(lengths.sum().item())
        idx = torch.repeat_interleave(starts, lengths) + (
            torch.arange(total, device=starts.device)
            - torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
        )
        return (self.model.xyz[idx], self.model.raw_opacity[idx], self.model.raw_scaling[idx],
                self.model.raw_rotation[idx], self.model.features_dc[idx], self.model.features_rest[idx])

    def _compute_colors(self, means3D: torch.Tensor, shs: torch.Tensor, camera: Camera) -> torch.Tensor:
        degree = self.model.active_sh_degree
        if degree > 0:
            dirs = means3D - camera.camera_center
            dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
            sh_dim = (degree + 1) ** 2
            colors = eval_sh(degree, shs.transpose(1, 2)[:, :, :sh_dim], dirs)
            return torch.clamp_min(colors + 0.5, 0.0)
        return torch.clamp_min(C0 * shs[:, 0, :] + 0.5, 0.0)

    def _sync(self) -> None:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def enable_profiling(self) -> None:
        """Turn on per-stage timing collection (see RenderResult and
        get_last_timings/get_profiling_stats). Off by default: profiling
        forces a torch.cuda.synchronize() per stage, which is real overhead
        the hot path shouldn't pay just to have profiling available."""
        self.profiler.enable()

    def disable_profiling(self) -> None:
        self.profiler.disable()

    def get_last_timings(self) -> dict[str, float] | None:
        """Per-stage ms for the most recent render() call, or None if
        profiling is off or nothing has rendered yet."""
        return self.profiler.last()

    def get_profiling_stats(self) -> dict[str, dict[str, float]]:
        """Per-stage {count, mean_ms, min_ms, max_ms, total_ms} accumulated
        since enable_profiling() or the last reset_profiling()."""
        return self.profiler.stats()

    def reset_profiling(self) -> None:
        self.profiler.reset()

    def render(
        self, camera: Camera, *, render_camera: Camera | None = None,
        render_mode: RenderMode = "gaussian", point_size: float = 0.01,
        sparsity: int = 1, bounds: Bounds | None = None, min_opacity: float = 0.0,
        depth_ratio: float = 0.0,
    ) -> RenderOutput:
        """`render_mode`/`point_size`: see render/display_modes.py.
        `sparsity`: render every Nth candidate splat (1 = disabled).
        `bounds`: axis-aligned world-space crop box, see
        culling/frustum.py's visible_point_mask_bounds_torch (None =
        disabled). `min_opacity`: non-destructive per-frame opacity-
        threshold mask, distinct from compression.py's load-time
        prune_low_opacity (0.0 = disabled). `depth_ratio`: blend fraction
        toward the kernel's median depth, see render/extras.py (0.0 =
        expected-depth-only, this library's original behavior).
        `render_camera`: optional second camera to project/rasterize from,
        decoupled from `camera` -- which always drives selection (octree
        cull, LOD, candidate filters) and view-dependent SH color -- while
        `render_camera` (defaulting to `camera` when omitted, identical to
        the pre-existing single-camera behavior) supplies the rasterizer's
        viewmatrix/projmatrix/campos/resolution/fov. Lets a caller render
        the splat set one camera would select from a different camera's
        viewpoint, e.g. an external/debug view for auditing culling."""
        self.profiler.start()
        lap = self.profiler.lap
        model = self.model
        proj_camera = render_camera if render_camera is not None else camera

        leaf_vis = None
        if self._has_octree:
            leaf_vis = visible_leaf_mask_torch(self._node_aabbs_gpu, camera.full_proj_transform)
        lap("cull")

        leaf_fine, leaf_coarse = lod_split(
            leaf_vis, camera, self._proxy_xyz_gpu, self._leaf_center_gpu, self._leaf_radius_gpu,
            self.lod_leaf_pixel_threshold,
        ) if self.octree_lod else (leaf_vis, None)
        lap("lod_select")

        # Brute force (no octree, or culling disabled) falls back to the
        # raw model tensors directly -- the only place a full N-sized
        # tensor is still touched on the per-frame hot path.
        raw = (
            self._gather_leaf_slices(leaf_fine) if leaf_fine is not None else
            (model.xyz, model.raw_opacity, model.raw_scaling,
             model.raw_rotation, model.features_dc, model.features_rest)
        )

        raw = apply_candidate_filters(
            raw, camera, has_leaf_gather=leaf_fine is not None, lap=lap,
            narrow_phase=self.culling_narrow_phase, margin=self.culling_margin,
            screen_size=self.screen_size_culling, min_pixel_radius=self.screen_size_min_pixels,
            sparsity=sparsity, bounds=bounds, min_opacity=min_opacity,
        )

        # render_fields' underlying activation (sigmoid/exp/normalize) runs
        # on this already-culled candidate set, not the full model -- see
        # GaussianModel._activate.
        means3D, opacity, scales, rotations, shs = model._activate(*raw)
        n_full = means3D.shape[0]
        means3D, opacity, scales, rotations, proxy_idx = append_proxies(
            leaf_coarse, means3D, opacity, scales, rotations,
            self._proxy_xyz_gpu, self._proxy_opacity_gpu, self._proxy_scale_gpu, self._proxy_rotation_gpu,
        )
        lap("gather")

        self.last_visible_count = int(means3D.shape[0])
        means2D = torch.zeros_like(means3D)
        colors = self._compute_colors(means3D[:n_full], shs, proj_camera)
        colors = blend_proxy_colors(colors, proxy_idx, self._proxy_features_dc_gpu)
        lap("sh_eval")

        scales, opacity = apply_render_mode(render_mode, scales, opacity, point_size)
        lap("render_mode")

        raster_settings = self._settings_cls(
            image_height=int(proj_camera.height),
            image_width=int(proj_camera.width),
            tanfovx=math.tan(proj_camera.fov_x * 0.5),
            tanfovy=math.tan(proj_camera.fov_y * 0.5),
            bg=self.background,
            scale_modifier=1.0,
            viewmatrix=proj_camera.world_view_transform,
            projmatrix=proj_camera.full_proj_transform,
            sh_degree=model.active_sh_degree,
            campos=proj_camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = self._rasterizer_cls(raster_settings=raster_settings)

        rendered_image, _radii, allmap = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=colors,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        lap("rasterize")

        depth, extras = extract_depth_and_extras(allmap, depth_ratio, self.with_extras)
        lap("depth_extract")
        return RenderOutput(
            rgb=rendered_image, depth=depth,
            num_rendered=self.last_visible_count,
            alpha=extras.alpha if extras else None,
            normal=extras.normal if extras else None,
            middepth=extras.middepth if extras else None,
            distortion=extras.distortion if extras else None,
        )
