# gsplat2d-rendering

General-purpose rendering for trained **2D Gaussian Splatting** (surfel) models:
load a PLY, render it from any camera pose, and keep it fast at splat counts
that would otherwise choke a laptop GPU — via octree frustum culling, a
two-level LOD, and in-memory fp16/SH compression.

This library owns the "pose in, image out" rendering core shared across
several unrelated projects (viewers, robot sensor simulators, offline
tools). It has no opinion on where a project's camera poses come from, what
UI it has, or what it does with the rendered image — those stay in the
downstream project.

## Install

```bash
git submodule update --init --recursive
pip install -e .                                          # this library
pip install -e third_party/diff-surfel-rasterization      # the CUDA kernel
```

The CUDA kernel build needs `nvcc` and a matching PyTorch/CUDA install, same
as any other Gaussian-Splatting CUDA extension.

### License note

This repo's own code has no license restriction beyond what you and your
downstream projects need. `third_party/diff-surfel-rasterization`
(vendored as a git submodule, not copied) is Inria's Gaussian-Splatting
research kernel and carries **its own non-commercial research license** —
see that submodule's `LICENSE.md`. Anything built on top of this library
that renders through that kernel inherits that restriction.

## Quickstart

```python
import numpy as np
import torch
import gsplat2d_rendering as gs2d

model = gs2d.load_gaussian_model("scene.ply", device="cuda")

# Optional but recommended for big scenes: build/cache an octree and put
# the model into its leaf-contiguous order for fast per-frame culling.
xyz_np = model.xyz.float().cpu().numpy()
octree = gs2d.load_or_build_octree("scene.ply", xyz_np, build_index=True)
model.reorder_(torch.from_numpy(octree.flat_indices).to("cuda"))

renderer = gs2d.Renderer(model, octree=octree, culling_enabled=True)

intrinsics = gs2d.Intrinsics.from_fov(1280, 720, fov_x=np.radians(60))
camera = gs2d.Camera.look_at(eye=[0, -5, 2], target=[0, 0, 0], intrinsics=intrinsics)

result = renderer.render(camera)
# result.rgb: (H, W, 3) uint8 -- result.depth: (H, W) float32 -- result.num_rendered: int
```

See `examples/render_ply.py` for a runnable end-to-end script.

## Camera poses

`Camera` is deliberately decoupled from any one project's pose convention.
Build one from whatever you have:

| Constructor | Use it when you have |
|---|---|
| `Camera.from_c2w(r_c2w, t_c2w, intrinsics)` | a rotation matrix + camera position in world coords |
| `Camera.from_c2w_matrix(c2w_4x4, intrinsics)` | a single 4x4 camera-to-world matrix (common NeRF/3DGS pose-file layout) |
| `Camera.from_w2c(r_w2c, t_w2c, intrinsics)` | a world-to-camera matrix (e.g. straight from COLMAP) |
| `Camera.look_at(eye, target, intrinsics)` | an orbit/free-fly viewer camera |

All of them expect the pose already resolved to the optical-frame axis
convention (x-right, y-down, z-forward) — the one every Gaussian-Splatting
training pipeline stores poses in. If your pose source uses a different
convention (e.g. a ROS `base_link`, or a Z-up robotics frame), resolve that
in your own project before building a `Camera`; this library has no
opinion on it.

## Rendering large scenes

Three independent techniques, each opt-in and each with a real cost/quality
tradeoff — tune against your own scene, none of the defaults below are
universally correct:

- **Octree culling** (`culling_enabled=True` + an `Octree`): only fetches
  and rasterizes splats whose octree leaf intersects the view frustum.
  `leaf_max` (see `build_octree`) is the key knob — bigger leaves mean
  cheaper index-building and coarser culling, smaller leaves mean tighter
  culling at more per-frame gather overhead.
- **Two-level LOD** [STILL NOT FULLY SUPPORTED](`octree_lod=True`, built via `compute_lod=True` in
  `load_or_build_octree`): leaves whose projected screen size falls below
  `lod_leaf_pixel_threshold` render as one precomputed, moment-matched
  "proxy" Gaussian instead of their full splat set.
- **In-memory compression** (`compression_level=0..3` in
  `load_gaussian_model`): fp16 storage, then progressively truncated
  spherical harmonics. Reduces VRAM and per-frame gather bandwidth, not
  rasterizer cost directly.
- **Opacity pruning** (`opacity_threshold` in `load_gaussian_model`):
  permanently drops splats at or under a given activated opacity at load
  time — on real trained models this is often a majority of all splats.

Every technique above is documented in more depth in its own module's
docstring (`culling/octree.py`, `lod.py`, `compression.py`).

## Chunk streaming for a scene too large to load at once

The techniques above all still assume the whole model fits in host RAM and
VRAM at once — `streaming.ChunkManager` is for when it doesn't. It's a
second, coarser optimization layered *on top of* octree culling above (that
still cuts rasterizer cost; this cuts the load/upload cost that happens
before a single splat is even rasterized): partition a model into disk
chunks once, offline, then keep only the chunks near the current camera
resident in RAM/VRAM at all, streaming the rest in/out as the camera moves.

```python
import torch
import gsplat2d_rendering as gs2d

# Offline, once: partition + physically reorder into chunk-contiguous order.
model = gs2d.load_gaussian_model("scene.ply", device="cpu")
xyz_np = model.xyz.float().cpu().numpy()
manifest = gs2d.build_octree(xyz_np, leaf_max=500_000)   # coarse -- target splats/chunk
model.reorder_(torch.from_numpy(manifest.flat_indices.astype("int64")))
gs2d.write_gaussian_model("scene_chunked.ply", model)
with open("scene_chunks.idx", "wb") as fh:
    gs2d.save_octree(fh, manifest)

# At runtime:
manifest = gs2d.load_octree("scene_chunks.idx")
chunks = gs2d.ChunkManager("scene_chunked.ply", manifest, device="cuda",
                            vram_margin_hops=1, ram_margin_hops=1)
model, octree = chunks.initial_sync_load(camera)          # blocking, once at startup
renderer = gs2d.Renderer(model, octree=octree, culling_enabled=True)

# Per frame:
chunks.update(camera, max_load_hops=3)                      # non-blocking
swap = chunks.drain_pending_swap()
if swap is not None:
    model, octree = swap
    # install into your renderer the same way you would after a fresh load
```

Key ideas, each documented in depth in its own module:

- **Two residency tiers, both CPU-resident** (`manager.py`): "VRAM tier"
  (composited into the model you actually render, `vram_margin_hops` beyond
  the camera's strict frustum) and "RAM tier" (`ram_margin_hops` further
  out, prefetched but not yet composited). Neither tier holds a persistent
  per-chunk GPU tensor — only the one merged/composited model ever touches
  the GPU, so promoting a chunk from RAM to VRAM tier is a synchronous,
  zero-cost dict move, not a `.to("cuda")` copy.
- **Adjacency is a padded-AABB-overlap graph, not K-nearest-centroids**
  (`adjacency.py`): margin/prefetch reach is defined in adjacency-graph hops
  (`streaming.tiers.compute_desired_tiers`), which only gives the intended
  "every resident chunk's neighbors are also resident (or one tier out)"
  guarantee if the graph actually contains every genuine spatial neighbor
  pair — a fixed-K graph turned out not to, on real scenes with irregular
  chunk sizes; see `adjacency.py`'s own docstring for what broke and why.
- **Per-chunk fine-grained culling octrees are cached and stitched, not
  rebuilt from scratch on every composition change** (`fine_octree.py`): a
  chunk's own point set never changes for the life of a session, so its
  local octree only needs building once; `FineOctreeCache.stitch()`
  assembles a combined octree over any resident subset by concatenating
  already-built per-chunk pieces, at a cost that scales with leaf-node
  count (hundreds), not point count (millions).
- **Bounded background I/O** (`transitions.py`): up to a fixed number of
  chunk reads run concurrently, so a single newly-exposed frustum edge
  uncovering several chunks at once doesn't serialize into several seconds
  of visibly gray, not-yet-loaded chunks.
- **Throttled, backgrounded recompositing** (`rebuild.py`): re-stitching the
  actually-rendered model is time-throttled and runs off the calling
  thread, since it can be a real cost at high chunk counts.

### Profiling

Per-stage timings (cull, LOD select, gather, SH eval, rasterize, depth
extract, CPU copy) are off by default — measuring them forces a
`torch.cuda.synchronize()` per stage, which is real overhead the hot path
shouldn't pay just to have profiling available. Turn it on/off and pull
results with plain methods on `Renderer` (or `SplatRenderer`), no `render()`
kwarg needed:

```python
renderer.enable_profiling()
renderer.render(camera)
renderer.get_last_timings()      # {"cull": 0.03, "rasterize": 44.8, ...} for that one call
renderer.get_profiling_stats()   # {"rasterize": {"count": 1, "mean_ms": 44.8, "min_ms": ..., "max_ms": ..., "total_ms": ...}, ...}
renderer.reset_profiling()       # clear accumulated stats, keep profiling enabled
renderer.disable_profiling()
```

## Architecture

```
src/gsplat2d_rendering/
├── model.py              GaussianModel: raw tensors + activation functions
├── compression.py        fp16 storage, SH truncation, opacity pruning
├── sh.py                 Spherical-harmonics basis evaluation (degree 0-3)
├── camera.py             Camera + Intrinsics: pose/intrinsics -> render-ready matrices
├── math_utils/
│   └── rotations.py       Quaternion <-> rotation-matrix helpers
├── io/
│   ├── ply.py             load_gaussian_model, detect_sh_degree
│   ├── chunked_ply.py     ChunkedPlyReader: cheap repeated row-range reads of one PLY
│   └── paths.py           resolve_ply_path (direct .ply or training-output dir)
├── culling/
│   ├── octree.py          Octree: build / save / load
│   ├── frustum.py         GPU-native frustum + screen-size visibility tests
│   └── cache.py           On-disk octree index cache wrapper
├── lod.py                 Two-level LOD: per-leaf moment-matched proxy splats
├── render/
│   ├── rasterizer.py       SplatRenderer: direct diff-surfel-rasterization wrapper
│   ├── lod_blend.py        Leaf LOD selection + coarse-leaf proxy blending
│   └── pipeline.py         Renderer: Camera in, RGB/depth numpy arrays out
└── streaming/              Load-time chunk streaming for scenes too large to load whole
    ├── manager.py           ChunkManager: the public entry point, composes the below
    ├── adjacency.py         Padded-AABB-overlap chunk adjacency graph + BFS hop expansion
    ├── tiers.py             Pure VRAM/RAM residency math over that graph
    ├── transitions.py       Bounded background disk-read worker pool
    ├── fine_octree.py       Per-chunk culling-octree cache + stitch into one composited octree
    └── rebuild.py           Throttled, backgrounded recompositing scheduler
```

Each file stays focused on one concern; when one starts covering two, split
it rather than letting it grow (see `render/lod_blend.py`, split out of
`render/rasterizer.py` for exactly this reason).

## Design history

This library generalizes the rendering core originally built independently
for `gs_sensors` (a ROS 2 Gaussian-Splat sensor simulator) — dropping that
project's ROS/ROS-pose-specific bits
(camera YAML profiles, a Sim(3) world<->training-frame transform) in favor
of the plain `Camera`/`Intrinsics` API above — informed by prior art in
[`Kestrel`](https://github.com/mlisi1/Kestrel) (a PyQt 2D-GS viewer), which
independently validated reordering a model into octree leaf-contiguous
order for fast per-frame gathers at large scale, and later validated
`streaming.ChunkManager` end to end (built there first as project-specific
code, then generalized back into this library once proven).
