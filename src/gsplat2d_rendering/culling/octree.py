"""Octree spatial index over splat centers: build, save, load.

Standard axis-aligned octree over point positions -- a generic spatial
partitioning structure, not tied to any particular renderer. See
frustum.py for the GPU-native frustum tests that consume it, and cache.py
for the on-disk cache wrapper most callers actually want.

`leaf_max` is the key tuning knob: bigger leaves mean fewer, larger
per-frame gathers (cheaper index-building work, coarser culling), smaller
leaves mean tighter culling at the cost of more per-frame gather overhead.
The right value is scene- and hardware-dependent -- there is no universally
correct default, so measure on your own model rather than trusting any one
number carried over from another project.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gsplat2d_rendering._log import info, verbose


@dataclass
class Octree:
    node_aabbs: np.ndarray     # [L, 6] float32 (xmin,ymin,zmin,xmax,ymax,zmax) per leaf
    node_offsets: np.ndarray   # [L+1] int64, into flat_indices
    flat_indices: np.ndarray   # [N] int64, permutation of point indices, leaf-ordered

    # Two-level LOD (lod.py's build_leaf_proxies): one merged "proxy"
    # Gaussian per leaf, precomputed at index-build time, swapped in for a
    # whole leaf's individual splats when the leaf's projected screen size
    # is small (see render/rasterizer.py's leaf_fine/leaf_coarse split).
    # None unless the index was built with compute_lod enabled -- an older
    # cached index without these fields degrades gracefully to "LOD
    # unavailable", not an error.
    proxy_xyz: np.ndarray | None = None              # [L, 3], world position
    proxy_scale: np.ndarray | None = None            # [L, 2 or 3], already activated (world units)
    proxy_rotation: np.ndarray | None = None         # [L, 4], normalized quaternion (w, x, y, z)
    proxy_opacity: np.ndarray | None = None          # [L, 1], already activated ([0, 1])
    proxy_features_dc: np.ndarray | None = None      # [L, 1, 3], raw SH-DC space (pre C0/+0.5)

    @property
    def has_lod(self) -> bool:
        return self.proxy_xyz is not None


def build_octree(xyz: np.ndarray, leaf_max: int = 5000, max_depth: int = 8,
                  verbose_log: bool = False) -> Octree:
    """verbose_log routes the "Built octree" summary line through verbose()
    instead of info(): building one octree over a whole model is a rare,
    notable event worth NORMAL visibility, but a caller building many small
    per-partition octrees (e.g. chunk streaming building one per chunk, up
    to hundreds per session) would otherwise flood NORMAL-level output with
    one line per chunk."""
    t0 = time.perf_counter()
    n = xyz.shape[0]
    leaves_indices: list[np.ndarray] = []
    leaves_aabb: list[np.ndarray] = []

    stack: list[tuple[np.ndarray, int]] = [(np.arange(n, dtype=np.int64), 0)]
    while stack:
        indices, depth = stack.pop()
        if indices.size == 0:
            continue
        pts = xyz[indices]
        aabb_min = pts.min(axis=0)
        aabb_max = pts.max(axis=0)
        if indices.size <= leaf_max or depth >= max_depth:
            leaves_indices.append(indices)
            leaves_aabb.append(np.concatenate([aabb_min, aabb_max]))
            continue

        center = (aabb_min + aabb_max) * 0.5
        octant = (
            (pts[:, 0] >= center[0]).astype(np.int64)
            + (pts[:, 1] >= center[1]).astype(np.int64) * 2
            + (pts[:, 2] >= center[2]).astype(np.int64) * 4
        )
        for o in range(8):
            child = indices[octant == o]
            if child.size:
                stack.append((child, depth + 1))

    if leaves_indices:
        node_offsets = np.zeros(len(leaves_indices) + 1, dtype=np.int64)
        for i, idx in enumerate(leaves_indices):
            node_offsets[i + 1] = node_offsets[i] + idx.size
        flat_indices = np.concatenate(leaves_indices).astype(np.int64)
        node_aabbs = np.stack(leaves_aabb).astype(np.float32)
    else:
        node_offsets = np.zeros(1, dtype=np.int64)
        flat_indices = np.zeros(0, dtype=np.int64)
        node_aabbs = np.zeros((0, 6), dtype=np.float32)

    elapsed = time.perf_counter() - t0
    log_fn = verbose if verbose_log else info
    log_fn(__name__, f"Built octree: {n:,} points -> {len(node_aabbs):,} leaf nodes "
                      f"(leaf_max={leaf_max:,}) in {elapsed:.2f}s")
    _log_leaf_stats(node_aabbs, node_offsets)
    return Octree(node_aabbs=node_aabbs, node_offsets=node_offsets, flat_indices=flat_indices)


def _log_leaf_stats(node_aabbs: np.ndarray, node_offsets: np.ndarray) -> None:
    """Diagnostic-only (VERBOSE level): splats-per-leaf spread and a
    per-depth leaf histogram, both purely derived from what build_octree
    already computed -- no extra structural work, just reporting."""
    if len(node_aabbs) == 0:
        return
    counts = np.diff(node_offsets)
    verbose(__name__, f"Splats/leaf: avg={counts.mean():.0f} min={counts.min():,} max={counts.max():,}")

    edge = (node_aabbs[:, 3:] - node_aabbs[:, :3]).max(axis=1)
    root_edge = float(edge.max())
    if root_edge <= 0:
        return
    depths = np.round(np.log2(root_edge / np.maximum(edge, 1e-12))).astype(int)
    for d in sorted(set(depths.tolist())):
        cnt = int((depths == d).sum())
        verbose(__name__, f"  depth {d}: {cnt:,} leaf nodes")


def save_octree(path: str | Path | object, octree: Octree) -> None:
    """`path` may be a str/Path, or an already-open file object (anything
    with `.write`). `np.savez_compressed` appends `.npz` to a string path
    that doesn't already end in it -- surprising for a caller with their
    own extension convention (e.g. `.idx`) -- but leaves an open file
    handle's name alone, so file-like `path` is passed through untouched
    rather than `str()`-ed."""
    kwargs = dict(
        node_aabbs=octree.node_aabbs,
        node_offsets=octree.node_offsets,
        flat_indices=octree.flat_indices,
    )
    if octree.has_lod:
        kwargs.update(
            proxy_xyz=octree.proxy_xyz,
            proxy_scale=octree.proxy_scale,
            proxy_rotation=octree.proxy_rotation,
            proxy_opacity=octree.proxy_opacity,
            proxy_features_dc=octree.proxy_features_dc,
        )
    target = path if hasattr(path, "write") else str(path)
    np.savez_compressed(target, **kwargs)
    dest = getattr(path, "name", path)
    verbose(__name__, f"Saved octree ({len(octree.node_aabbs):,} leaf nodes) to {dest}")


def load_octree(path: str | Path | object) -> Octree:
    """See save_octree's docstring re: file-like `path`."""
    data = np.load(path if hasattr(path, "read") else str(path))
    has_lod = "proxy_xyz" in data.files
    octree = Octree(
        node_aabbs=data["node_aabbs"],
        node_offsets=data["node_offsets"],
        flat_indices=data["flat_indices"],
        proxy_xyz=data["proxy_xyz"] if has_lod else None,
        proxy_scale=data["proxy_scale"] if has_lod else None,
        proxy_rotation=data["proxy_rotation"] if has_lod else None,
        proxy_opacity=data["proxy_opacity"] if has_lod else None,
        proxy_features_dc=data["proxy_features_dc"] if has_lod else None,
    )
    dest = getattr(path, "name", path)
    info(__name__, f"Loaded octree ({len(octree.node_aabbs):,} leaf nodes) from {dest}")
    return octree
