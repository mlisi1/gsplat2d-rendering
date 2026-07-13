"""Per-chunk fine-grained culling octree cache + stitching, for a caller
that keeps a large model resident as separate spatial chunks (see
manager.py) but still wants per-frame octree culling over whatever
composition is currently resident, without re-running the full recursive
octree split over the *entire* resident set every time that composition
changes.

A chunk's point *set* never changes for the life of a session (chunks are a
fixed, disjoint partition of one source file), so its own local octree only
ever needs building once -- `FineOctreeCache.ensure()` builds and caches it
the first time a chunk's data is read, then reuses the cached structure on
every later call for that chunk id, however many times the chunk itself
gets evicted and re-read over the session. `stitch()` then assembles a
combined octree over any subset of resident chunks purely by concatenating
each one's already-built local `node_aabbs`/`flat_indices` (shifting indices
by a running point offset) -- cost scales with total *leaf-node* count
across resident chunks (typically hundreds), not total point count
(typically millions), and doesn't require rebuilding anything when the
resident *set* changes, only when a chunk id is seen for the very first
time.
"""
from __future__ import annotations

import numpy as np
import torch

from gsplat2d_rendering.culling import Octree, build_octree
from gsplat2d_rendering.model import GaussianModel, concat_gaussian_models


class FineOctreeCache:
    """`leaf_max` is this cache's own per-chunk octree granularity -- almost
    always much smaller than the chunk size itself (chunks are a coarse,
    disk-residency partition; this is the fine, per-frame-culling partition
    within each chunk), matching whatever leaf size a caller would otherwise
    use for a single-file octree."""

    def __init__(self, leaf_max: int):
        self.leaf_max = leaf_max
        self._octrees: dict[int, Octree] = {}
        # The permutation each cached octree implies, kept separately from
        # the octree itself (whose own flat_indices gets rewritten to
        # identity once the permutation has actually been applied -- see
        # ensure()) -- re-applied on every call for a given chunk id, since
        # each read from disk hands back a fresh, still-unpermuted model
        # even though the split computation itself never needs redoing.
        self._reorder_perm: dict[int, torch.Tensor] = {}

    def ensure(self, chunk_id: int, model: GaussianModel, verbose_log: bool = True) -> None:
        """Builds and caches `chunk_id`'s own local octree the first time
        this is called for it, then permutes `model` in place into that
        octree's leaf-contiguous order (on every call, not just the first --
        see module docstring: the split is cached, but the permutation must
        be re-applied to every fresh read). `verbose_log` defaults True
        since this typically fires once per chunk read, up to hundreds of
        times per session -- see build_octree's own `verbose_log` param."""
        if chunk_id not in self._octrees:
            xyz_np = model.xyz.float().cpu().numpy()
            local_octree = build_octree(xyz_np, leaf_max=self.leaf_max, verbose_log=verbose_log)
            self._reorder_perm[chunk_id] = torch.from_numpy(
                local_octree.flat_indices.copy()
            ).to(model.xyz.device)
            # node_aabbs/node_offsets are unaffected by row order (per-leaf
            # bounds/counts, not row indices) -- only flat_indices needs
            # rewriting, to identity, since every future call for this
            # chunk_id will already be permuted by the reorder_ below before
            # stitch() ever sees it.
            local_octree.flat_indices = np.arange(model.xyz.shape[0], dtype=np.int64)
            self._octrees[chunk_id] = local_octree

        model.reorder_(self._reorder_perm[chunk_id], verbose_log=verbose_log)

    def clear(self) -> None:
        """Drops every cached octree/permutation -- call this whenever
        something changes a chunk's own point identities/order wholesale
        (e.g. a compression-level change: different precision can shift a
        borderline point across an octant boundary, so a stale split isn't
        safe to keep for any resident chunk, not just the ones currently
        loaded)."""
        self._octrees.clear()
        self._reorder_perm.clear()

    def stitch(self, resident: dict[int, GaussianModel], device: str) -> tuple[GaussianModel, Octree]:
        """Assembles one combined `(GaussianModel, Octree)` over `resident`
        (chunk_id -> CPU-or-GPU-resident model, each already permuted via a
        prior `ensure()` call for that chunk_id) by concatenating each
        chunk's cached local octree -- every chunk's own cached
        `flat_indices` is already the identity permutation post-`ensure()`,
        so the concatenation (`flat_indices + point_offset`) naturally
        produces a globally leaf-contiguous result with no extra reordering
        work here.

        The final model is moved to `device` as a single transfer at the
        end -- the only per-call device transfer this performs, so a caller
        keeping per-chunk data CPU-resident until composited (see
        manager.py's own reasoning for why) never pays for more than one
        GPU upload per stitch, regardless of how many chunks are resident."""
        ids = list(resident.keys())
        models = [resident[cid] for cid in ids]
        merged_cpu = concat_gaussian_models(models)

        aabb_parts, idx_parts, size_parts = [], [], []
        point_offset = 0
        for cid, model in zip(ids, models):
            octree = self._octrees[cid]
            aabb_parts.append(octree.node_aabbs)
            idx_parts.append(octree.flat_indices + point_offset)
            size_parts.append(np.diff(octree.node_offsets))
            point_offset += model.xyz.shape[0]

        node_aabbs = np.concatenate(aabb_parts, axis=0) if aabb_parts else np.zeros((0, 6), dtype=np.float32)
        flat_indices = np.concatenate(idx_parts, axis=0) if idx_parts else np.zeros(0, dtype=np.int64)
        node_sizes = np.concatenate(size_parts, axis=0) if size_parts else np.zeros(0, dtype=np.int64)
        node_offsets = np.zeros(len(node_sizes) + 1, dtype=np.int64)
        np.cumsum(node_sizes, out=node_offsets[1:])

        stitched = Octree(node_aabbs=node_aabbs, node_offsets=node_offsets, flat_indices=flat_indices)
        merged = merged_cpu.to(device)
        del merged_cpu
        return merged, stitched
