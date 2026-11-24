# The surfel rasterizer (`gsplat2d_rendering._raster`)

## Provenance

Written for this project from public descriptions only:

- the 2DGS paper (Huang et al., SIGGRAPH 2024): the surfel model, the
  ray–splat intersection, the depth and normal definitions, the
  screen-space low-pass
- the 3DGS paper (Kerbl et al., 2023): tile binning, depth sorting,
  front-to-back compositing
- the CUB documentation (radix sort, scan)

No code was taken or translated from `diff-surfel-rasterization`,
`gaussian-splatting`, or any GPL implementation. The real-SH constants
come from PlenOctree (BSD-2-Clause; see `NOTICE`) by way of `sh.py`.
The library is Apache-2.0.

## Layout

```
_raster/
├── __init__.py              rasterize(...) -> RasterOutput
└── csrc/
    ├── torch_binding.cpp    torch tensors <-> core; scratch from torch's allocator
    └── core/                CUDA C++, no torch
        ├── surfel_raster.h  public API: SurfelInputs/Camera/Options/Outputs, surfel_render()
        ├── math.cuh         vector helpers, quaternion columns, real SH
        ├── preprocess.cuh   tile coverage shared by count and emit
        ├── preprocess.cu    per-splat transform, cull, colour, footprint
        ├── binning.cu       pair emission, tile ranges
        ├── raster.cu        per-tile compositing
        └── render.cu        surfel_render: preprocess -> scan -> emit -> sort -> ranges -> raster
```

Render only: no backward pass, no gradients, no training buffers.

## Per-splat math

Camera: `x_c = R x_w + t` (OpenCV axes), pinhole `fx, fy, cx, cy`, pixel
`(i, j)` centred at `(i + 0.5, j + 0.5)`. Pixel ray `d = ((x - cx)/fx, (y - cy)/fy, 1)`.

For a splat with mean `μ`, quaternion `q`, log scales `(ls_u, ls_v)`,
opacity logit `ℓ`:

- `p = R μ + t`; `t_u, t_v` = first two columns of `R · rot(q/|q|)`;
  `n = t_u × t_v`; `s = exp(ls)`; `o = sigmoid(ℓ)`.
- Culled if `o < alpha_min`, `p.z <= near_z`, or the splat is degenerate
  (zero or non-finite quaternion, non-finite position or scale). Some real
  PLYs carry millions of zero-filled rows, which would otherwise bin to
  the whole image while drawing nothing.
- `c = p · n`; if `c > 0`, flip `n` and `c`, so the normal faces the camera.
- `a_u = t_u / s_u`, `b_u = p · a_u` (same for `v`).
- Ray hit with the splat's plane: `τ = c / (d · n)`, which is camera z
  because `d.z = 1`. Disc coordinates `u = τ (d · a_u) - b_u`,
  `v = τ (d · a_v) - b_v`, `ρ = u² + v²`.
- Colour: `max(SH(μ - campos) + 0.5, 0)`, or the stored RGB when `sh_degree = -1`.

The splat contributes `α = min(alpha_max, o·exp(-ρ/2))` where
`α >= alpha_min`, i.e. exactly inside `ρ <= r²`, `r² = 2 ln(o / alpha_min)`.

### Low-pass

Models trained with the 2DGS low-pass need it at render time:
`ρ = min(ρ_disc, 2·|pixel - centre_px|²)`, a screen-space blob of about
0.7 px that keeps sub-pixel splats from vanishing. Where the blob term
wins, or the ray misses the plane (`|d · n| < 1e-12` or `τ <= near_z`),
**the depth is the splat centre's `p.z`**, not the plane hit. A nearly
edge-on disc's plane hit can be arbitrarily far from the disc, and using
it put depths of ~100 into a scene 3 units away during testing.

`GaussianModel.surfel_lowpass` says which a model needs. It defaults to
true for every PLY layout, overridable with a `comment surfel_lowpass 0|1`
header line (which `write_gaussian_model` writes). The plan this kernel was
built from assumed spirula's 3-scale models render without it, but spirula's
own eval renders say otherwise. Against them, bonsai views 3 and 10 score
49.0 / 49.2 dB with low-pass on and 39.3 / 42.0 dB with it off; garden
views score 37.7 / 38.5 dB on and 28.7 / 29.7 dB off.

## Footprint

`H = K · [s_u t_u | s_v t_v | p]` maps disc coordinates `(u, v, 1)` to
pixels, so the cutoff disc projects to the conic with dual matrix
`D = H · diag(r², r², -1) · Hᵀ`.

When the whole cutoff disc is in front of the near plane
(`p.z - r·sqrt((s_u t_u.z)² + (s_v t_v.z)²) > near_z`, which implies
`D33 < 0`), the image of the disc is an ellipse with

- centre `(D13, D23) / D33`
- centred form `{centre + X : Xᵀ M⁻¹ X <= 1}`, where
  `M_ij = (D_i3 D_j3 - D_ij D33) / D33²`
- half-extents `sqrt(M11)`, `sqrt(M22)`.

Otherwise the disc crosses the near plane. By default such splats are
**culled**, as the previous kernel did (black-box tests show it never drew
them). With `near_crossing=True` the footprint becomes the whole image, and
the raster stage's `τ > near_z` test removes what lies in front of the plane.
That is more faithful but expensive: each such splat is binned to every tile. With low-pass on, the box also covers the circle of radius
`r/√2` around `centre_px`.

`max_sigma > 0` caps the cutoff at that many standard deviations
(`r² = min(r², max_sigma²)`, applied in the raster test too): fewer pairs,
slightly different image.

All analytic bounds are padded by `0.01 + 1e-4·extent` px, so float
rounding in the per-pixel test can never put a contributing pixel
outside its binned tiles. The footprint math runs in double precision
(a few dozen operations per splat).

### Exact tiles

`exact_tiles` bins a splat only to the tiles its ellipse touches. For a
tile row whose pixel centres span `Y ∈ [ya, yb]` (relative to the centre,
clipped to `±sqrt(M22)`), the ellipse's x-interval at height `Y` is

```
X = (M12·Y ± sqrt(det M · (M22 - Y²))) / M22
```

The ellipse is convex, so its leftmost point over the band is the left
end at `ya` or `yb`, or the global leftmost point `(-sqrt(M11), -M12/sqrt(M11))`
if that lies in the band. The rightmost point is found the same way.
Count and emit both call `for_each_tile` (`preprocess.cuh`); if they
disagreed, emit would write past the scanned offsets. It is off when
low-pass is on (the union is not one conic) and for near-plane-crossing
splats.

## Binning and raster

1. Preprocess writes 64-byte records `(a_u,b_u) (a_v,b_v) (n,c) (rgb,o)`,
   plus `(centre_px, p.z)` on the low-pass path, and a tile count.
2. `cub::DeviceScan::InclusiveSum` over the counts, then one 8-byte copy
   to the host to size the pair buffers. This is the frame's only sync.
3. Emit: `key = (tile_id << 32) | float_bits(p.z)`, value = splat index.
4. `cub::DeviceRadixSort::SortPairs` over the low `32 + bit_width(num_tiles)`
   bits. It is stable, and emit order follows splat index, so equal
   depths keep index order.
5. Per-tile `[start, end)` ranges over the sorted keys.
6. Raster: one 16×16 block per tile, records staged through shared memory
   256 at a time, front to back:

```
T = 1
for each splat:
    α = ...; skip if α < alpha_min
    if T·(1-α) < t_min: stop (this splat is not added)
    w = α·T;  C += w·rgb;  Dz += w·τ;  [N += w·n;  if T > 0.5: zmed = τ]
    T = T·(1-α)
rgb = C + T·bg;  alpha = 1-T;  depth = Dz/alpha (0 if alpha == 0)
```

`aux` and `lowpass` are template parameters, so the plain RGB+depth path
computes no normals and no blob term. The 2DGS depth-distortion channel
is not produced: it is a training loss, not a viewer output.

Attributes other than positions may be float16 (`compression_level >= 1`).
Preprocess reads them directly, so compression costs no per-frame float32
copy.

## Tests

`tests/reference_render.py` is a brute-force float64 implementation of
the math above: every pixel against every splat, no tiles. The tests
require the kernel to match it to `1e-4`:

- `tests/test_raster.py`: every combination of low-pass, aux,
  exact_tiles and fp16 inputs; every SH degree; background and `rgb8`.
  Also checks that `exact_tiles` gives an identical image with fewer pairs.
- `tests/test_raster_edges.py`: centre just past the near plane, a disc
  crossing it, edge-on and degenerate discs, discs off every image edge
  and on the last row and column, opacity just above and below
  `alpha_min`, an off-centre principal point, equal depths.

Run with `PYTHONPATH=src:tests python3 -m pytest tests`.

## Parity with the previous (Inria) kernel

Measured once, from a separate checkout of the previous commit, on a
7.2M-splat model trained with the original 2DGS code: 5 orbit poses,
1280×720, brute force (no culling), RTX 5070 Laptop.

| | PSNR vs old | depth rel. err p99 | old ms | new ms |
|---|---|---|---|---|
| default | 38–53 dB | ≤ 0.19 | 46–63 | 29–39 |
| default, near-crossing discs removed from both | 52–57 dB | ≤ 0.004 | 46–63 | 16–17 |
| `max_sigma=3` | 39–49 dB | ≤ 0.19 | 46–63 | 16–21 |

The first row was measured before near-crossing culling became the default;
its gap was those discs, which the old kernel never drew. The default now
matches the second row.

Known remaining difference: around the projected centre of strongly tilted
splats, the old kernel's low-pass gives somewhat different alpha (up to
~0.2 on single pixels). Fitting simple variants of the blob term (scale, a
half-pixel offset) did not reproduce it. This kernel follows the formula
above, which the brute-force reference checks.
