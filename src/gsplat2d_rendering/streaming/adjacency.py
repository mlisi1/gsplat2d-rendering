"""Spatial adjacency between disk chunks (coarse Octree leaves -- see
streaming/manager.py's module docstring for the chunk-streaming feature this
serves), plus BFS hop-expansion over that graph.

Chunks are Octree leaves at chunk-streaming granularity: tight AABBs around
each chunk's own points, not padded to the octant they were split from -- so
there are real gaps between chunks that are genuinely spatial neighbors.
`build_chunk_adjacency` below is a padded-AABB-overlap test rather than a
fixed-K-nearest-centroid graph, which was tried first and turned out to be
wrong two different ways (both reproduced against real scenes as a
resident chunk bordering a non-resident one with no buffer chunk between
them -- exactly the invariant `ChunkManager`'s tier residency depends on):
a fixed K undercounts whenever a chunk genuinely has more real neighbors
than K, and even after symmetrizing the raw K-NN output (adding a reverse
edge whenever only one side's own top-K ranked the other), a neighbor pair
where *neither* side ranked the other in its own top-K -- both had enough
other, closer centroids to fill their list -- still had no edge in either
direction to symmetrize. Padded-AABB overlap has neither failure mode: it's
symmetric by construction (interval overlap doesn't depend on evaluation
order) and scales adjacency degree with each chunk's own local geometry
instead of an arbitrary neighbor count.
"""
from __future__ import annotations

import numpy as np

DEFAULT_PAD_FRACTION = 0.3  # each chunk AABB is grown by this fraction of its
                            # own extent (per axis) before the overlap test
DEFAULT_FALLBACK_K = 3      # nearest-centroid fallback for a chunk the padded-
                            # overlap test still leaves with zero neighbors


def build_chunk_adjacency(
    node_aabbs: np.ndarray,
    pad_fraction: float = DEFAULT_PAD_FRACTION,
    fallback_k: int = DEFAULT_FALLBACK_K,
) -> dict[int, list[int]]:
    """Adjacency graph over `node_aabbs` ([L, 6] float, xmin/ymin/zmin/xmax/
    ymax/zmax per chunk) -- an `Octree.node_aabbs` at chunk-streaming
    granularity. Each AABB is grown by `pad_fraction` of its own extent (per
    axis, not a fixed absolute distance) before a pairwise overlap test:
    growing proportionally to each chunk's own size closes genuine gaps at
    both a large sparse chunk's scale and a small dense chunk's scale at
    once, where a fixed absolute epsilon either misses the former or
    over-joins the latter (a naive "AABBs touch within an epsilon" test
    measured 35% of chunks with zero neighbors on a real scene, because
    that epsilon was too small relative to larger/sparser chunks' own gaps).

    `fallback_k` nearest-centroid edges are added only for a chunk the
    overlap test still leaves completely isolated (a genuine spatial
    outlier, e.g. a handful of points isolated from all denser geometry) --
    a safety net so BFS expansion always has *something* to reach from
    every chunk, not the primary source of adjacency.

    O(N^2) pairwise overlap test: fine for the tens-to-low-hundreds of
    chunks a sane chunk size produces, not meant to scale to the thousands
    of chunks an extremely fine chunk size would produce."""
    n = len(node_aabbs)
    mins, maxs = node_aabbs[:, :3], node_aabbs[:, 3:]
    extents = np.maximum(maxs - mins, 1e-9)
    pad = extents * pad_fraction
    pmins, pmaxs = mins - pad, maxs + pad

    overlap = np.ones((n, n), dtype=bool)
    for ax in range(3):
        overlap &= (pmins[:, None, ax] <= pmaxs[None, :, ax])
        overlap &= (pmaxs[:, None, ax] >= pmins[None, :, ax])
    np.fill_diagonal(overlap, False)

    adjacency: dict[int, set[int]] = {
        i: set(np.nonzero(overlap[i])[0].tolist()) for i in range(n)
    }

    isolated = [i for i in range(n) if not adjacency[i]]
    if isolated:
        centers = 0.5 * (mins + maxs)
        k = min(fallback_k, n - 1)
        for i in isolated:
            if k <= 0:
                continue
            dists = np.linalg.norm(centers - centers[i], axis=1)
            dists[i] = np.inf
            nearest = np.argpartition(dists, k - 1)[:k]
            for j in nearest:
                adjacency[i].add(int(j))
                adjacency[int(j)].add(i)

    return {i: sorted(neighbors) for i, neighbors in adjacency.items()}


def bfs_expand(adjacency: dict[int, list[int]], seed: set[int], hops: int) -> set[int]:
    """All chunk ids reachable from any chunk in `seed` within `hops`
    adjacency steps, including `seed` itself. `hops<=0` returns `seed`
    unchanged (0 hops = "don't expand at all", a well-defined off state for
    a caller layering an optional margin on top of this)."""
    visited = set(seed)
    frontier = set(seed)
    for _ in range(max(hops, 0)):
        next_frontier: set[int] = set()
        for cid in frontier:
            next_frontier.update(adjacency.get(cid, ()))
        next_frontier -= visited
        if not next_frontier:
            break
        visited |= next_frontier
        frontier = next_frontier
    return visited


def nearest_chunk_id(node_aabbs: np.ndarray, anchor: np.ndarray) -> int:
    """The single chunk whose centroid is closest to `anchor` (a world-space
    xyz point) -- a fallback seed for a caller whose primary visibility test
    (e.g. a frustum test) finds nothing at all, so residency/reachability
    never goes fully empty on a degenerate starting pose.

    `anchor` must be a point the caller actually cares about being near --
    e.g. an orbit camera's look-at point, not its raw eye position. This
    matters more than it looks: an orbit camera's eye sits some distance
    *away* from what it's actually pointed at, often well outside the
    scene's own bounding volume entirely, so "nearest chunk to the camera's
    raw position" can land in a part of the adjacency graph with zero
    hop-overlap with the chunks actually in front of the camera, even when
    the centroids involved are only a few world units apart -- this exact
    mistake (seeding a reachability BFS from a nearest-to-eye-position
    lookup instead of nearest-to-look-at) silently evicted an entire
    resident set on the first frame after startup in one real caller, before
    being traced back to this function being fed the wrong anchor."""
    centers = 0.5 * (node_aabbs[:, :3] + node_aabbs[:, 3:])
    dists = ((centers - np.asarray(anchor)) ** 2).sum(axis=1)
    return int(dists.argmin())
