"""GPU-native frustum/screen-size visibility tests, all torch (not numpy):
render/rasterizer.py's per-frame gather works from leaf-contiguous index
ranges (GaussianModel.reorder_ + Octree.node_offsets), not a per-point
boolean mask, so there's never a reason to materialize an N-length mask on
CPU.

Gribb & Hartmann's plane-extraction-from-clip-matrix method, a standard
generic graphics technique -- the matrix convention matches camera.py
(`clip_row = point_row @ full_proj_transform`, so planes are extracted from
*columns* of `full_proj_transform`, not rows).

Only 5 planes are tested (left/right/top/bottom/near); the far plane is
deliberately omitted because the CUDA rasterizer itself doesn't hard-clip at
zfar, so culling against it would incorrectly drop splats it would have
still rendered.
"""
from __future__ import annotations

import torch


def _frustum_planes(full_proj_transform: torch.Tensor):
    m = full_proj_transform
    planes = torch.stack([
        m[:, 0] + m[:, 3],  # left
        m[:, 3] - m[:, 0],  # right
        m[:, 1] + m[:, 3],  # bottom
        m[:, 3] - m[:, 1],  # top
        m[:, 2],            # near
    ], dim=0)
    return planes[:, :3], planes[:, 3]  # normals [5,3], d_vals [5]


def visible_leaf_mask_torch(node_aabbs: torch.Tensor, full_proj_transform: torch.Tensor) -> torch.Tensor:
    """Gribb-Hartmann p-vertex test, vectorized over all leaves -> [L] bool,
    entirely in torch on `full_proj_transform`'s own device. `node_aabbs`
    must already be a torch tensor on that device. This is the frustum
    broad-phase test -- see module docstring."""
    normals, d_vals = _frustum_planes(full_proj_transform)

    aabb_min = node_aabbs[:, :3]
    aabb_max = node_aabbs[:, 3:]

    pos_mask = normals.unsqueeze(1) >= 0  # [5, 1, 3]
    p_vertex = torch.where(pos_mask, aabb_max.unsqueeze(0), aabb_min.unsqueeze(0))  # [5, L, 3]
    dots = (p_vertex * normals.unsqueeze(1)).sum(dim=2) + d_vals.unsqueeze(1)
    return (dots >= 0).all(dim=0)


def visible_point_mask_exact_torch(
    xyz: torch.Tensor, full_proj_transform: torch.Tensor, margin: float = 0.0,
) -> torch.Tensor:
    """Exact per-point Gribb-Hartmann test against the same 5 planes as
    `visible_leaf_mask_torch` (no far plane, same rationale), but on point
    centers directly instead of a leaf's AABB -- a narrow-phase refinement
    meant to run only on the (much smaller) candidate set that already
    passed the leaf-level broad phase, not the full point cloud, so it
    stays cheap.

    `margin`: this tests splat *centers*, which have zero screen-space
    extent, unlike the splats actually being rendered -- at margin=0 a
    splat can visibly pop out right as its center crosses the frustum edge,
    before its rendered footprint has actually left the screen. Inflates
    the plane test by this amount to compensate; tune from what you
    actually see at the frame edges, this isn't derived from splat scale."""
    normals, d_vals = _frustum_planes(full_proj_transform)
    dots = xyz @ normals.T + d_vals  # [K, 5]
    return (dots >= -margin).all(dim=1)


def visible_point_mask_screen_size_torch(
    xyz: torch.Tensor, scales: torch.Tensor, world_view_transform: torch.Tensor,
    focal_x: float, focal_y: float, cutoff: float = 3.0, min_pixel_radius: float = 1.0,
) -> torch.Tensor:
    """Screen-space size test: culls candidates whose projected footprint
    is smaller than `min_pixel_radius` pixels -- a splat that small
    contributes negligible unique detail. Meant to run only on points that
    already passed the frustum broad phase, same reasoning as
    `visible_point_mask_exact_torch`. Also reusable (with cutoff=1.0) for a
    leaf-level LOD fine/coarse decision, applied to a leaf's own
    center/radius instead of one splat's.

    Deliberately a coarse, conservative proxy rather than replicating the
    CUDA kernel's own exact anisotropic footprint math (compute_transmat/
    compute_aabb in forward.cu, which projects the splat's local tangent
    frame through the projection matrix) -- reimplementing that exactly in
    Python is real surface area for a subtle mismatch bug. Instead this
    uses the standard real-time-rendering pinhole approximation, projected
    radius ~= focal * world_radius / depth, with `world_radius` taken as
    the *larger* in-plane scale axis (worst case, so this never
    underestimates and over-culls) times `cutoff` to match the ~3-standard-
    deviation effective radius the kernel itself uses by default (its own
    `cutoff = 3.0f` in forward.cu). Points behind the camera (depth <= 0)
    are never culled here -- that's frustum culling's job, not this test's;
    a non-positive depth just means 'not testable, keep it'."""
    n = xyz.shape[0]
    ones = xyz.new_ones((n, 1))
    xyz_h = torch.cat([xyz, ones], dim=-1)
    p_view = xyz_h @ world_view_transform
    depth = p_view[:, 2]

    world_radius = scales.amax(dim=-1) * cutoff
    focal = (focal_x + focal_y) * 0.5
    safe_depth = torch.clamp(depth, min=1e-6)
    pixel_radius = focal * world_radius / safe_depth

    return (depth <= 0) | (pixel_radius >= min_pixel_radius)
