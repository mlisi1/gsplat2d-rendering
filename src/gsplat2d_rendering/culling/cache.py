"""On-disk cache wrapper around octree.py's build_octree -- the entry point
most callers actually want: load a cached index if present, build (and
cache) one on request, or return None to mean "culling disabled, render
every splat every frame."
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from gsplat2d_rendering.culling.octree import Octree, build_octree, load_octree, save_octree


def index_cache_path(ply_path: str | Path, opacity_threshold: float = 0.0) -> Path:
    """<ply_dir>/.gsplat2d_rendering/<ply_stem>[_opacityN].idx.npz

    `opacity_threshold` changes *which* splats exist (see
    compression.py's prune_low_opacity), not just how they're stored --
    an octree built at one threshold is structurally invalid for a model
    loaded at a different one (wrong point count, wrong point identities).
    Suffixing the cache filename only when nonzero keeps every existing
    cache built before opacity pruning existed valid and reachable at its
    original path (opacity_threshold=0.0 is still the default)."""
    ply_path = Path(ply_path)
    suffix = f"_opacity{opacity_threshold:g}" if opacity_threshold > 0.0 else ""
    return ply_path.parent / ".gsplat2d_rendering" / f"{ply_path.stem}{suffix}.idx.npz"


def load_or_build_octree(
    ply_path: str | Path,
    xyz: np.ndarray,
    leaf_max: int = 5000,
    max_depth: int = 8,
    build_index: bool = False,
    compute_lod: bool = False,
    opacity: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    rotation: np.ndarray | None = None,
    features_dc: np.ndarray | None = None,
    opacity_threshold: float = 0.0,
    keep_normal_axis: bool = False,
) -> Octree | None:
    """Loads a cached index if present; builds (and caches) one if
    `build_index` is set and no cache exists; otherwise returns None
    (culling disabled -- the caller renders every splat every frame).

    `compute_lod`: also builds (and caches) per-leaf LOD proxies via
    lod.py's build_leaf_proxies -- requires opacity/scale/rotation/
    features_dc (already-activated arrays, see that function's docstring)
    when actually building a fresh index. If a *cached* index is loaded
    that predates LOD (no proxy data in the .npz), this degrades
    gracefully to "LOD unavailable this run" with a printed notice rather
    than an error -- rebuild with build_index=True to add it.

    `opacity_threshold` must be the same value the model was actually
    loaded with (see index_cache_path) -- passed through here only to pick
    the right cache file, this function doesn't prune anything itself.
    A loaded cache whose point count doesn't match `xyz` (e.g. a stale
    cache, or `xyz` itself changed on disk) fails loudly here rather than
    crashing confusingly later in GaussianModel.reorder_ with an
    out-of-range index.

    `keep_normal_axis`: passed straight through to `build_leaf_proxies` --
    only relevant when `compute_lod=True` for a genuinely-3D-GS-family
    model (rather than the 2D surfel model this library targets by
    default). Not part of the cache filename (unlike `opacity_threshold`):
    a cached index's own `proxy_scale` column count already reveals which
    mode built it (2 vs 3 columns), so mixing this flag across runs
    against the same cache fails loudly via a real shape mismatch at the
    call site rather than silently, same spirit as the point-count check
    above."""
    cache_path = index_cache_path(ply_path, opacity_threshold)
    if cache_path.is_file():
        octree = load_octree(cache_path)
        print(f"[gsplat2d_rendering] Loaded octree index from {cache_path}")
        if octree.flat_indices.shape[0] != xyz.shape[0]:
            raise ValueError(
                f"Cached octree at {cache_path} covers {octree.flat_indices.shape[0]:,} points, "
                f"but the loaded model has {xyz.shape[0]:,} -- stale cache (e.g. from a different "
                f"opacity_threshold, or the PLY changed on disk). Delete {cache_path} or pass "
                "build_index=True to regenerate it."
            )
        needs_lod_rebuild = compute_lod and not octree.has_lod
        if not needs_lod_rebuild:
            return octree
        if not build_index:
            print(f"[gsplat2d_rendering] Cached index at {cache_path} has no LOD proxies -- "
                  "pass build_index=True to regenerate it with compute_lod enabled; "
                  "LOD disabled for this run")
            return octree
        print(f"[gsplat2d_rendering] Cached index at {cache_path} has no LOD proxies -- "
              "rebuilding it with LOD since build_index=True ...")
        # falls through to the build below, deliberately not returning here
    elif not build_index:
        print(f"[gsplat2d_rendering] No octree index at {cache_path} and build_index=False -- culling disabled")
        return None

    print(f"[gsplat2d_rendering] Building octree index (leaf_max={leaf_max:,}) ...")
    octree = build_octree(xyz, leaf_max=leaf_max, max_depth=max_depth)
    if compute_lod:
        if opacity is None or scale is None or rotation is None or features_dc is None:
            raise ValueError("compute_lod=True requires opacity/scale/rotation/features_dc")
        from gsplat2d_rendering.lod import build_leaf_proxies
        print(f"[gsplat2d_rendering] Building LOD proxies ({len(octree.node_aabbs):,} leaves) ...")
        (octree.proxy_xyz, octree.proxy_scale, octree.proxy_rotation,
         octree.proxy_opacity, octree.proxy_features_dc) = build_leaf_proxies(
            octree, xyz, opacity, scale, rotation, features_dc,
            keep_normal_axis=keep_normal_axis)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_octree(cache_path, octree)
    print(f"[gsplat2d_rendering] Built {len(octree.node_aabbs):,} leaf nodes, saved to {cache_path}")
    return octree
