"""ChunkManager: per-frame VRAM/RAM/disk residency for load-time chunk
streaming of a model too large to load (or keep resident) all at once.

A second, coarser optimization layered *on top of* per-frame octree
culling (`culling/frustum.py`), not a replacement for it: culling only cuts
*rasterizer* cost -- the whole point cloud still has to be parsed into host
RAM and uploaded to VRAM first. Chunk streaming instead partitions a model
into disk chunks (Octree leaves at chunk-streaming granularity -- much
coarser than a per-frame culling octree's `leaf_max`, see `fine_octree.py`
for the fine-grained one this class builds *within* each resident chunk)
and keeps only chunks near the current camera actually resident in RAM/VRAM
at all -- the rest stays on disk. Per-frame octree culling keeps working
unmodified on top of whatever's currently resident.

Requires a chunk-contiguous PLY (rows for chunk `i` occupy one contiguous
byte range) and its manifest (an `Octree` at chunk granularity: AABB + row
range per chunk) -- producing both is the caller's job (`build_octree` +
`write_gaussian_model` over the model reordered by the octree's own
`flat_indices`; see `culling/octree.py`/`io/ply.py`), since the on-disk
path/naming convention a project uses for these two artifacts is exactly
the kind of project-specific choice this library stays out of.

**Chunk source data is always CPU-resident, in both tiers -- neither tier
holds a persistent per-chunk GPU tensor.** A renderer needs one single
composited model, so per-chunk data has to be concatenated into one tensor
set before it's useful regardless of tier; keeping each VRAM-tier chunk as
its own standalone GPU tensor *on top of* that composited copy would mean
two full copies of the resident working set live in VRAM at once at high
residency -- measured directly as the cause of an otherwise-unexplained
"VRAM keeps climbing" report against a model that fits fine loaded whole.
The only tensor that's ever GPU-resident is the one merged tensor
`fine_octree.FineOctreeCache.stitch` produces and a caller installs into
its renderer. A useful side effect: since both tiers are CPU-resident,
VRAM<->RAM tier reclassification (a chunk crossing from "prefetched" to
"actually needed" or back) is a synchronous, zero-cost dict move in
`update()` -- no disk read, no GPU copy, no background thread needed.

Composes the rest of this package: `adjacency` for the spatial graph,
`tiers` for the pure residency math, `transitions.TransitionPool` for the
bounded background disk-read pool, `fine_octree.FineOctreeCache` for the
per-chunk culling-octree cache/stitch, and `rebuild.RebuildScheduler` for
the throttled composited-model rebuild. This class's own job is gluing
those together and owning the two resident-chunk dicts (`_vram`/`_ram`)."""
from __future__ import annotations

import concurrent.futures

import numpy as np
import torch

from gsplat2d_rendering.culling import Octree
from gsplat2d_rendering.culling.frustum import visible_leaf_mask_torch
from gsplat2d_rendering.io.chunked_ply import ChunkedPlyReader
from gsplat2d_rendering.model import GaussianModel
from gsplat2d_rendering.streaming.adjacency import build_chunk_adjacency, nearest_chunk_id
from gsplat2d_rendering.streaming.fine_octree import FineOctreeCache
from gsplat2d_rendering.streaming.rebuild import RebuildScheduler
from gsplat2d_rendering.streaming.tiers import compute_desired_tiers
from gsplat2d_rendering.streaming.transitions import TransitionPool, spawn_tier_transitions


class ChunkManager:
    def __init__(self, chunked_ply_path: str, manifest: Octree, device: str,
                 vram_margin_hops: int = 0, ram_margin_hops: int = 0,
                 fine_leaf_max: int = 5000, rebuild_throttle_s: float = 0.4,
                 compression_level: int = 0):
        self.chunked_ply_path = chunked_ply_path
        self.manifest         = manifest
        self.device           = device
        # Adjacency-hop-count residency -- see tiers.py's module docstring
        # for what each of these means. 0 hops means "off" for either tier,
        # a well-defined state, not a special case.
        self.vram_margin_hops = vram_margin_hops
        self.ram_margin_hops  = ram_margin_hops
        # gsplat2d_rendering's own in-memory compression_level (0=none,
        # 1=fp16, 2=fp16+SH-degree-1, 3+=fp16+SH-degree-0), applied per
        # chunk read via ChunkedPlyReader.read_range -- see set_compression
        # for why changing this forces a full re-read of every resident chunk.
        self.compression_level = compression_level

        # Depends only on the manifest, fixed for the session -- see its own
        # module docstring for why this is a padded-AABB-overlap test rather
        # than a fixed-K-nearest-centroid graph.
        self._adjacency = build_chunk_adjacency(manifest.node_aabbs)
        self._fine = FineOctreeCache(fine_leaf_max)

        self._chunk_offsets = manifest.node_offsets
        self._aabbs_gpu = torch.from_numpy(manifest.node_aabbs).to(device)
        # One open mmap for the whole session instead of re-parsing the PLY
        # header and re-establishing the mmap on every single chunk read.
        self._reader = ChunkedPlyReader(chunked_ply_path)

        # Tier classification only -- both dicts hold CPU-resident
        # GaussianModels (see module docstring). Mutated only from whatever
        # thread calls update() (inside its drain/sync-unload/reclassify
        # steps) -- background transition workers never touch these dicts
        # directly, only what TransitionPool hands back through drain().
        self._vram: dict[int, GaussianModel] = {}
        self._ram: dict[int, GaussianModel] = {}

        self.last_oom_chunk_id: int | None = None
        self._transitions = TransitionPool(
            self._load_and_prepare_chunk,
            on_oom=lambda chunk_id: setattr(self, "last_oom_chunk_id", chunk_id),
        )
        self._rebuild = RebuildScheduler(
            lambda snapshot: self._fine.stitch(snapshot, self.device),
            throttle_s=rebuild_throttle_s,
            on_oom=lambda: setattr(self, "last_oom_chunk_id", -1),
        )

        # Populated by update()/initial_sync_load() each time they run --
        # chunk_states() reads these instead of recomputing frustum+BFS
        # itself, since a caller's own debug overlay typically wants this
        # right after update() in the same frame.
        self._last_desired_vram: frozenset[int] = frozenset()
        self._last_desired_vram_margin: frozenset[int] = frozenset()

    def set_margins(self, vram_margin_hops: int, ram_margin_hops: int) -> None:
        """Live reconfiguration -- changes only affect which chunks
        `update()` asks for next call; already-resident chunks are left
        alone rather than eagerly evicted."""
        self.vram_margin_hops = vram_margin_hops
        self.ram_margin_hops = ram_margin_hops

    def set_compression(self, level: int) -> None:
        """Live compression-level change. Unlike set_margins, this can't
        leave already-resident chunks alone: their cached tensors are at
        the *old* level's precision/SH-degree, and
        `gsplat2d_rendering.concat_gaussian_models` deliberately raises on a
        mismatched SH degree across inputs -- its own guard against exactly
        this "chunks from different compression levels" scenario. So this
        evicts every resident chunk outright (both tiers) and clears the
        per-chunk fine-octree cache (conservative but correct: fp16
        rounding could in principle shift a borderline point across an
        octant boundary, so a stale split isn't safe to keep either) --
        `update()`'s normal transition machinery then re-reads everything
        fresh at the new level, exactly like a newly-panned-in chunk that
        was never seen before."""
        if level == self.compression_level:
            return
        self.compression_level = level
        self._vram.clear()
        self._ram.clear()
        self._fine.clear()
        self._rebuild.reset()

    # ── chunk row-range I/O ──────────────────────────────────────────────────

    def _read_chunk(self, chunk_id: int) -> GaussianModel:
        """Always reads to CPU -- see module docstring; the only GPU upload
        this whole class performs happens once, in FineOctreeCache.stitch,
        for the final composited model."""
        start = int(self._chunk_offsets[chunk_id])
        end = int(self._chunk_offsets[chunk_id + 1])
        return self._reader.read_range(
            start, end - start, device="cpu",
            compression_level=self.compression_level, target_sh_degree=1,
        )

    def _load_and_prepare_chunk(self, chunk_id: int) -> GaussianModel:
        """The `load_fn` handed to `TransitionPool`: a disk read plus the
        one-time-per-chunk fine-octree build/permutation -- bundled so that
        cost (typically the more expensive of the two) lands inside the
        same background worker call that already pays for the chunk's disk
        read, instead of on whatever thread calls update()/
        initial_sync_load()."""
        model = self._read_chunk(chunk_id)
        self._fine.ensure(chunk_id, model)
        return model

    @staticmethod
    def _resolve_anchor(camera, anchor: np.ndarray | None) -> np.ndarray:
        if anchor is not None:
            return np.asarray(anchor)
        return camera.camera_center.detach().cpu().numpy()

    # ── startup ──────────────────────────────────────────────────────────────

    def initial_sync_load(self, camera, anchor: np.ndarray | None = None):
        """Blocking -- meant for a caller's own startup path (or when chunk
        streaming is toggled on mid-session), the same category as an
        offline/synchronous index build. Falls back to the single
        nearest-to-`anchor` chunk if the starting camera pose doesn't
        intersect any (so the scene is never blank at launch) -- see
        `adjacency.nearest_chunk_id`'s docstring for why `anchor` should be
        a look-at point, not the camera's raw position, if the caller has
        one. Reads the initial chunk set in parallel, bounded by the same
        concurrency cap `update()`'s transition pool uses -- a wide
        starting FOV can cover several chunks at once, and reading them one
        at a time would directly extend startup latency.

        Returns `(model, octree)` ready to install into a renderer."""
        mask = visible_leaf_mask_torch(self._aabbs_gpu, camera.full_proj_transform)
        ids = torch.nonzero(mask, as_tuple=True)[0].tolist()
        if not ids:
            ids = [nearest_chunk_id(self.manifest.node_aabbs, self._resolve_anchor(camera, anchor))]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._transitions.max_concurrent
        ) as pool:
            models = list(pool.map(self._load_and_prepare_chunk, ids))
        for cid, model in zip(ids, models):
            self._vram[cid] = model

        self._last_desired_vram = frozenset(ids)
        self._last_desired_vram_margin = frozenset()
        return self._rebuild.sync(dict(self._vram))

    # ── per-frame tier diff ─────────────────────────────────────────────────

    def update(self, camera, anchor: np.ndarray | None = None,
               max_load_hops: int | None = None) -> None:
        """Non-blocking. Call once per frame/tick with the current render
        camera. Recomputes desired VRAM/RAM sets (see `tiers.
        compute_desired_tiers` for exactly what `vram_margin_hops`/
        `ram_margin_hops`/`max_load_hops` each do), applies free synchronous
        tier reclassification and full-unloads, and tops up the disk-read
        transition pool. Also kicks a time-throttled fine-octree rebuild
        whenever the VRAM-tier set has changed since the last landed
        rebuild.

        `anchor` is only used as a fallback seed when the frustum test
        finds nothing at all (so residency never goes fully empty on a
        degenerate camera pose) -- see `adjacency.nearest_chunk_id`."""
        for result in self._transitions.drain():
            self._apply_transition_result(result)

        proj = camera.full_proj_transform
        frustum_mask = visible_leaf_mask_torch(self._aabbs_gpu, proj)
        frustum_visible = set(torch.nonzero(frustum_mask, as_tuple=True)[0].tolist())
        if not frustum_visible:
            frustum_visible = {nearest_chunk_id(self.manifest.node_aabbs, self._resolve_anchor(camera, anchor))}

        tiers = compute_desired_tiers(
            self._adjacency, frustum_visible,
            self.vram_margin_hops, self.ram_margin_hops, max_load_hops,
        )
        desired_vram, desired_ram = tiers.vram, tiers.ram

        # Read by chunk_states() for a caller's own debug overlay --
        # computed here, once per frame, rather than recomputed redundantly
        # there.
        self._last_desired_vram = tiers.strict
        self._last_desired_vram_margin = tiers.margin

        # Instant, synchronous tier reclassification for chunks whose data
        # is already CPU-resident in the other tier -- no disk I/O, no GPU
        # copy (chunk source data lives on CPU in both tiers, see module
        # docstring), so there's no reason to route this through the
        # background worker pool at all.
        for cid in list(desired_vram & self._ram.keys()):
            self._vram[cid] = self._ram.pop(cid)
        for cid in list((set(self._vram.keys()) - desired_vram) & desired_ram):
            self._ram[cid] = self._vram.pop(cid)

        # Free, synchronous full-unload: a chunk that fell out of both the
        # strict frustum and the margin ring needs no thread, just a dict
        # drop -- GC reclaims its (CPU) tensors. Chunks currently in flight
        # are excluded since a disk-read worker may be about to land a
        # result for them.
        inflight = self._transitions.inflight
        for cid in set(self._vram.keys()) - desired_vram - desired_ram - inflight:
            del self._vram[cid]
        for cid in set(self._ram.keys()) - desired_ram - desired_vram - inflight:
            del self._ram[cid]

        self._spawn_needed_transitions(desired_vram, desired_ram)

        # Purely time-throttled, not gated on the transition pool being idle:
        # under continuous camera motion a freed worker slot is immediately
        # refilled by the next newly-visible chunk, so the pool can stay
        # non-empty indefinitely -- gating on "settled" here would mean
        # chunks finish loading into VRAM-tier in the background but never
        # get folded into what's actually rendered until motion stops.
        if set(self._vram.keys()) != self._rebuild.last_rebuilt_ids:
            self._rebuild.maybe_kick(self._vram)

    def _apply_transition_result(self, result) -> None:
        if result.payload is None:
            return  # failed (OOM or otherwise) -- leave prior state as-is
        if result.tag != self.compression_level:
            return  # stale: set_compression() ran while this read was in flight
        if result.kind == "to_vram_from_disk":
            self._vram[result.chunk_id] = result.payload
        elif result.kind == "to_ram_from_disk":
            self._ram[result.chunk_id] = result.payload

    def _spawn_needed_transitions(self, desired_vram: set[int], desired_ram: set[int]) -> None:
        """Only genuine disk reads reach here -- tier reclassification
        between already-CPU-resident chunks is handled synchronously in
        update() above, before this is called. VRAM promotions are
        prioritized over RAM prefetch, with at least one slot always
        reserved for RAM demand -- see `spawn_tier_transitions`."""
        current_vram, current_ram = set(self._vram.keys()), set(self._ram.keys())
        inflight = self._transitions.inflight
        vram_candidates = {
            cid for cid in desired_vram - current_vram - inflight
            if self._transitions.cooled_down(cid)
        }
        ram_candidates = {
            cid for cid in desired_ram - current_ram - current_vram - inflight
            if self._transitions.cooled_down(cid)
        }
        spawn_tier_transitions(
            self._transitions, vram_candidates, ram_candidates,
            "to_vram_from_disk", "to_ram_from_disk", tag=self.compression_level,
        )

    def drain_pending_swap(self):
        """Call once per frame/tick. Returns `(gaussian_model, octree)` for
        a caller to install into its renderer, or `None` if no rebuild has
        landed since the last call."""
        return self._rebuild.drain()

    # ── debug / introspection ────────────────────────────────────────────────

    def visible_chunk_ids(self, camera) -> set[int]:
        """Which chunks intersect `camera`'s frustum -- independent of
        residency. Useful for a debug overlay that wants to bound which
        disk-tier chunk boxes are even worth drawing, keeping overlay cost
        independent of total chunk count."""
        mask = visible_leaf_mask_torch(self._aabbs_gpu, camera.full_proj_transform)
        return set(torch.nonzero(mask, as_tuple=True)[0].tolist())

    def chunk_states(self) -> dict[int, str]:
        """chunk_id -> 'vram' | 'vram_margin' | 'ram'. Absence means
        disk-only. 'vram' = strictly frustum-visible (the composition
        actually rendered); 'vram_margin' = adjacency-promoted into that
        same GPU-resident composition despite being outside the frustum
        right now (see tiers.py's `vram_margin_hops`) -- per-frame GPU
        culling hides it until the camera turns enough to bring it back
        into frame, with zero rebuild needed; 'ram' = CPU-only prefetch.
        Reflects the most recent update()/initial_sync_load() call's
        classification, not literal current GPU/CPU residency of that
        chunk's own tensor (see module docstring)."""
        states = {}
        for cid in self._vram:
            states[cid] = "vram_margin" if cid in self._last_desired_vram_margin else "vram"
        for cid in self._ram:
            states.setdefault(cid, "ram")
        return states

    def stats(self) -> tuple[int, int, int]:
        return len(self._vram), len(self._ram), len(self.manifest.node_aabbs)
