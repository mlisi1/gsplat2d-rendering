"""Two-level octree LOD: precomputes one merged 'proxy' Gaussian per leaf
at index-build time, swapped in for a whole leaf's individual splats when
the leaf's projected screen size is small (see render/rasterizer.py's
leaf_fine/leaf_coarse split). Split out from culling/ because proxy
construction (moment-matching, quaternion math) is a distinct concern from
octree building/frustum culling and only touches Octree's proxy_* fields,
not its spatial-partitioning core.
"""
from __future__ import annotations

import numpy as np

from gsplat2d_rendering.culling.octree import Octree


def _quat_to_rotmat_batch(q_wxyz: np.ndarray) -> np.ndarray:
    """Vectorized (w, x, y, z) quaternion -> rotation matrix, [K, 4] -> [K,
    3, 3] -- this is the splat rotation field's own convention (see
    model.py), not math_utils.rotations' (x, y, z, w) camera-pose
    convention, and not the same thing. Same formula as
    math_utils.rotations.quat_to_rotmat given a reordered input, written
    separately here only because that one isn't batched and this needs to
    run over many points at once without a Python-level loop."""
    w, x, y, z = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    n = q_wxyz.shape[0]
    r = np.empty((n, 3, 3), dtype=np.float64)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - z * w)
    r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w)
    r[:, 2, 1] = 2 * (y * z + x * w)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def build_leaf_proxies(
    octree: Octree,
    xyz: np.ndarray,
    opacity: np.ndarray,
    scale: np.ndarray,
    rotation: np.ndarray,
    features_dc: np.ndarray,
    keep_normal_axis: bool = False,
):
    """Precomputes one merged 'proxy' Gaussian per octree leaf -- a
    moment-matched single-Gaussian reduction of the leaf's full splat set.
    Runs once at index-build time, not per-frame.

    Inputs are already-activated values (opacity in [0, 1], scale in world
    units, rotation a normalized (w, x, y, z) quaternion, features_dc the
    raw per-splat SH-DC coefficient) -- these are GaussianModel's own
    get_opacity/get_scaling/get_rotation outputs, reused rather than
    reimplementing sigmoid/exp/normalize here.

    Per leaf, with per-point weight = opacity (a near-transparent splat
    should barely move the proxy's position/shape):
    - position: weighted centroid.
    - opacity: 1 - prod(1 - opacity_i), the standard alpha-compositing
      'over' formula treating the group as roughly co-located -- not a
      naive average, which would understate coverage for many low-opacity
      splats that together read as solid.
    - color: weighted average of the DC (view-independent) term only;
      proxies always render at SH degree 0 -- a merged region can't
      coherently represent several splats' different view-dependent
      behavior, the same tradeoff aggressive SH-truncation compression
      makes for the whole model (see compression.py).
    - shape: moment-matched covariance (standard Gaussian-mixture-to-
      single-Gaussian reduction -- combine each splat's own covariance
      with its offset from the merged mean), eigendecomposed back into
      scale+rotation so the proxy stays correctly oriented (e.g. a flat
      wall segment collapses to a flat, not spherical, proxy) instead of
      an axis-aligned blob.

    `scale` is genuinely 2D for the surfel rasterizer this library targets,
    not 3D: its CUDA kernel's own scale_to_mat (auxiliary.h) builds
    S = diag(scale.x, scale.y, 1.0) -- the third entry is a fixed 1.0
    placeholder purely so the rotation matrix's third column can be read
    off as the surfel's normal, not a real spatial extent (the normal
    direction carries zero variance). `scale` may arrive here with 2 or 3
    stored columns (some PLYs keep a vestigial third column) -- either way,
    the covariance computed below zero-pads to a true (not
    kernel-placeholder) zero variance along the normal. By default
    (`keep_normal_axis=False`) the *output* `proxy_scale` is 2 columns,
    matching that surfel kernel's per-splat float layout (a 3-column
    scales tensor would desync its flat per-splat read, corrupting every
    splat after the first -- silently, not an error). Pass
    `keep_normal_axis=True` only if pairing this with a genuinely-3D-GS
    kernel instead, whose splats have no surfel/degenerate-normal
    assumption, to get all 3 eigenvalues back, already in the same column
    order as `proxy_rotation`'s axes (see below). After eigendecomposition,
    the two largest-variance eigenvectors become the proxy's in-plane
    tangent axes and the smallest becomes its normal (matching the surfel
    kernel's own column convention, not eigh's default ascending order) --
    kept as the compute order regardless of `keep_normal_axis`, since it's
    also a reasonable, consistent axis choice for a genuine 3D ellipsoid.

    Fully vectorized over all N points via `flat_indices`'s leaf-contiguous
    ordering + `np.add.reduceat`/`np.multiply.reduceat` per-leaf segment
    reductions -- the only Python-level loop is rotmat_to_quat, called once
    per *leaf* (L, not N; L is small, a few thousand at most)."""
    from gsplat2d_rendering.math_utils.rotations import rotmat_to_quat

    offsets = octree.node_offsets
    order = octree.flat_indices
    n_leaves = len(offsets) - 1
    cuts = offsets[:-1]

    xyz_o = xyz[order].astype(np.float64)
    opacity_clamped = np.clip(opacity[order, 0], 0.0, 1.0).astype(np.float64)
    # +epsilon: opacity is the merge weight -- only matters for the
    # pathological case of a leaf where every point has opacity exactly 0,
    # keeping weighted sums well-defined without a special-cased branch.
    weights = opacity_clamped + 1e-6
    scale_o = scale[order].astype(np.float64)
    if scale_o.shape[1] == 2:
        scale_o = np.concatenate([scale_o, np.zeros((scale_o.shape[0], 1))], axis=1)
    elif scale_o.shape[1] != 3:
        raise ValueError(f"scale must have 2 or 3 columns, got {scale_o.shape[1]}")
    rotation_o = rotation[order].astype(np.float64)
    features_dc_o = features_dc[order, 0, :].astype(np.float64)

    w_sum = np.add.reduceat(weights, cuts)  # [L]

    mean = np.add.reduceat(weights[:, None] * xyz_o, cuts, axis=0) / w_sum[:, None]  # [L, 3]
    mean_per_point = np.repeat(mean, np.diff(offsets), axis=0)  # [N, 3]
    delta = xyz_o - mean_per_point  # [N, 3]

    r_all = _quat_to_rotmat_batch(rotation_o)  # [N, 3, 3]
    cov_i = np.einsum('nij,nj,nkj->nik', r_all, scale_o ** 2, r_all)  # R diag(s^2) R^T, [N, 3, 3]
    outer_delta = np.einsum('ni,nj->nij', delta, delta)  # [N, 3, 3]
    weighted_term = weights[:, None, None] * (cov_i + outer_delta)  # [N, 3, 3]
    cov_sum = (
        np.add.reduceat(weighted_term.reshape(-1, 9), cuts, axis=0).reshape(n_leaves, 3, 3)
        / w_sum[:, None, None]
    )

    eigvals, eigvecs = np.linalg.eigh(cov_sum)  # ascending order: [L,3] smallest->largest, [L,3,3]
    eigvals = np.clip(eigvals, 1e-12, None)

    # Reorder [smallest, mid, largest] -> [mid, largest, smallest] so
    # columns 0,1 (in-plane) get the two largest-variance directions and
    # column 2 (normal) gets the smallest -- see docstring.
    reorder = np.array([1, 2, 0])
    eigvals = eigvals[:, reorder]
    eigvecs = eigvecs[:, :, reorder].copy()

    dets = np.linalg.det(eigvecs)
    eigvecs[dets < 0, :, -1] *= -1  # fix reflections (det=-1) into proper rotations (det=+1)

    proxy_scale = np.sqrt(eigvals if keep_normal_axis else eigvals[:, :2]).astype(np.float32)

    proxy_rotation = np.empty((n_leaves, 4), dtype=np.float32)
    for leaf in range(n_leaves):
        q_xyzw = rotmat_to_quat(eigvecs[leaf])
        proxy_rotation[leaf] = (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])  # (x,y,z,w) -> (w,x,y,z)

    prod_1_minus_p = np.multiply.reduceat(1.0 - opacity_clamped, cuts)  # [L]
    proxy_opacity = np.clip(1.0 - prod_1_minus_p, 0.0, 0.99).astype(np.float32)[:, None]  # [L, 1]

    proxy_features_dc = (
        np.add.reduceat(weights[:, None] * features_dc_o, cuts, axis=0) / w_sum[:, None]
    ).astype(np.float32)[:, None, :]  # [L, 1, 3]

    proxy_xyz = mean.astype(np.float32)

    return proxy_xyz, proxy_scale, proxy_rotation, proxy_opacity, proxy_features_dc
