// SPDX-License-Identifier: Apache-2.0
//
// Per-splat stage: transform, cull, colour, and the screen footprint that
// binning uses. One thread per splat. Footprint math (cutoff radius, dual
// conic, centred ellipse form) is derived in docs/rasterizer.md.
#include "preprocess.cuh"

namespace surfel_raster {
namespace {

// Analytic bounds are padded so float rounding in the per-pixel test can
// never put a contributing pixel outside the tiles it was binned to.
__device__ __forceinline__ double footprint_pad(double extent) { return 0.01 + 1e-4 * extent; }

template <typename T>
__global__ void preprocess_kernel(SurfelInputs in, SurfelCamera cam, SurfelOptions opt,
                                  int tiles_x, int tiles_y, SplatRecords out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= in.n) return;
  out.count[i] = 0;

  const float* W = cam.w2c;
  Vec3 mu = v3(in.means[3 * i], in.means[3 * i + 1], in.means[3 * i + 2]);
  Vec3 p = v3(W[0] * mu.x + W[1] * mu.y + W[2] * mu.z + W[3],
              W[4] * mu.x + W[5] * mu.y + W[6] * mu.z + W[7],
              W[8] * mu.x + W[9] * mu.y + W[10] * mu.z + W[11]);
  if (!(p.z > cam.near_z)) return;

  const T* op = static_cast<const T*>(in.opacity_logits);
  float o = 1.f / (1.f + expf(-ld(op, i)));
  if (!(o >= opt.alpha_min)) return;

  const T* q = static_cast<const T*>(in.quats);
  Vec3 r0, r1;
  quat_columns(ld(q, 4ll * i), ld(q, 4ll * i + 1), ld(q, 4ll * i + 2), ld(q, 4ll * i + 3), r0, r1);
  Vec3 tu = v3(W[0] * r0.x + W[1] * r0.y + W[2] * r0.z, W[4] * r0.x + W[5] * r0.y + W[6] * r0.z,
               W[8] * r0.x + W[9] * r0.y + W[10] * r0.z);
  Vec3 tv = v3(W[0] * r1.x + W[1] * r1.y + W[2] * r1.z, W[4] * r1.x + W[5] * r1.y + W[6] * r1.z,
               W[8] * r1.x + W[9] * r1.y + W[10] * r1.z);
  Vec3 n = cross(tu, tv);
  const T* ls = static_cast<const T*>(in.log_scales);
  float su = expf(ld(ls, 2ll * i)), sv = expf(ld(ls, 2ll * i + 1));

  float c = dot(p, n);
  if (c > 0.f) {  // make the normal face the camera
    n = scale(n, -1.f);
    c = -c;
  }
  Vec3 au = scale(tu, 1.f / su), av = scale(tv, 1.f / sv);

  // Colour.
  const T* dc = static_cast<const T*>(in.features_dc);
  Vec3 rgb;
  if (in.sh_degree < 0) {
    rgb = v3(ld(dc, 3ll * i), ld(dc, 3ll * i + 1), ld(dc, 3ll * i + 2));
  } else {
    Vec3 dir = v3(mu.x - cam.campos[0], mu.y - cam.campos[1], mu.z - cam.campos[2]);
    dir = scale(dir, rsqrtf(fmaxf(dot(dir, dir), 1e-30f)));
    const T* rest = static_cast<const T*>(in.features_rest);
    long long rbase = 3ll * in.rest_stride * i;
    rgb = eval_sh(in.sh_degree, dir, [&](int k) {
      if (k == 0) return v3(ld(dc, 3ll * i), ld(dc, 3ll * i + 1), ld(dc, 3ll * i + 2));
      long long b = rbase + 3ll * (k - 1);
      return v3(ld(rest, b), ld(rest, b + 1), ld(rest, b + 2));
    });
    rgb = v3(fmaxf(rgb.x + 0.5f, 0.f), fmaxf(rgb.y + 0.5f, 0.f), fmaxf(rgb.z + 0.5f, 0.f));
  }

  // Footprint. The fragment weight o*exp(-rho/2) drops below alpha_min
  // exactly outside rho = r2.
  double r2 = 2.0 * log((double)o / (double)opt.alpha_min);
  double r = sqrt(r2);
  float cx_px = cam.fx * p.x / p.z + cam.cx, cy_px = cam.fy * p.y / p.z + cam.cy;

  int W_px = cam.width, H_px = cam.height;
  double x_lo = 0, x_hi = W_px - 1, y_lo = 0, y_hi = H_px - 1;  // pixel-index bounds
  bool have_conic = false;
  double xc = 0, yc = 0, m11 = 0, m12 = 0, m22 = 0;
  double ez = sqrt((double)su * tu.z * su * tu.z + (double)sv * tv.z * sv * tv.z);
  if (p.z - r * ez > cam.near_z) {
    // H = K [su tu | sv tv | p]; dual conic D = H diag(r2, r2, -1) H^T.
    double h1x = cam.fx * su * tu.x + cam.cx * su * tu.z, h1y = cam.fy * su * tu.y + cam.cy * su * tu.z,
           h1z = su * tu.z;
    double h2x = cam.fx * sv * tv.x + cam.cx * sv * tv.z, h2y = cam.fy * sv * tv.y + cam.cy * sv * tv.z,
           h2z = sv * tv.z;
    double h3x = cam.fx * p.x + cam.cx * p.z, h3y = cam.fy * p.y + cam.cy * p.z, h3z = p.z;
    double d11 = r2 * (h1x * h1x + h2x * h2x) - h3x * h3x;
    double d12 = r2 * (h1x * h1y + h2x * h2y) - h3x * h3y;
    double d22 = r2 * (h1y * h1y + h2y * h2y) - h3y * h3y;
    double d13 = r2 * (h1x * h1z + h2x * h2z) - h3x * h3z;
    double d23 = r2 * (h1y * h1z + h2y * h2z) - h3y * h3z;
    double d33 = r2 * (h1z * h1z + h2z * h2z) - h3z * h3z;
    // Centred form: the ellipse is {c + X : X^T M^-1 X <= 1}.
    xc = d13 / d33;
    yc = d23 / d33;
    double inv = 1.0 / (d33 * d33);
    m11 = (d13 * d13 - d11 * d33) * inv;
    m12 = (d13 * d23 - d12 * d33) * inv;
    m22 = (d23 * d23 - d22 * d33) * inv;
    double hx = sqrt(fmax(m11, 0.0)), hy = sqrt(fmax(m22, 0.0));
    if (isfinite(xc) && isfinite(yc) && isfinite(hx) && isfinite(hy)) {
      double px = footprint_pad(hx), py = footprint_pad(hy);
      x_lo = xc - hx - 0.5 - px;
      x_hi = xc + hx - 0.5 + px;
      y_lo = yc - hy - 0.5 - py;
      y_hi = yc + hy - 0.5 + py;
      have_conic = true;
    }
  }
  if (opt.lowpass) {
    // Screen-space blob 2*|d|^2 <= r2, i.e. a circle of radius r/sqrt(2).
    double rl = r * 0.70710678118654752 + footprint_pad(r);
    if (have_conic) {
      x_lo = fmin(x_lo, cx_px - rl - 0.5);
      x_hi = fmax(x_hi, cx_px + rl - 0.5);
      y_lo = fmin(y_lo, cy_px - rl - 0.5);
      y_hi = fmax(y_hi, cy_px + rl - 0.5);
    }
    have_conic = false;  // the union is not one conic
  }
  int ix0 = (int)fmax(ceil(x_lo), 0.0), ix1 = (int)fmin(floor(x_hi), (double)(W_px - 1));
  int iy0 = (int)fmax(ceil(y_lo), 0.0), iy1 = (int)fmin(floor(y_hi), (double)(H_px - 1));
  if (ix0 > ix1 || iy0 > iy1) return;
  int4 rect = make_int4(ix0 / kTile, iy0 / kTile, ix1 / kTile, iy1 / kTile);

  out.rec[4 * i + 0] = make_float4(au.x, au.y, au.z, dot(p, au));
  out.rec[4 * i + 1] = make_float4(av.x, av.y, av.z, dot(p, av));
  out.rec[4 * i + 2] = make_float4(n.x, n.y, n.z, c);
  out.rec[4 * i + 3] = make_float4(rgb.x, rgb.y, rgb.z, o);
  if (opt.lowpass) out.blob[i] = make_float4(cx_px, cy_px, p.z, 0.f);
  out.zkey[i] = __float_as_uint(p.z);
  out.rect[i] = rect;

  ExactConic ec;
  ec.valid = false;
  if (opt.exact_tiles && have_conic) {
    ec.valid = true;
    ec.xc = xc; ec.yc = yc; ec.m12 = m12; ec.m22 = m22;
    ec.hx = sqrt(fmax(m11, 0.0));
    ec.det = fmax(m11 * m22 - m12 * m12, 0.0);
    ec.pad = footprint_pad(fmax(ec.hx, sqrt(fmax(m22, 0.0))));
    store_conic(out.conic, i, ec);
  } else if (out.conic) {
    store_conic(out.conic, i, ec);
  }
  out.count[i] = count_tiles(rect, ec, cam.width);
}

}  // namespace

cudaError_t launch_preprocess(const SurfelInputs& in, const SurfelCamera& cam,
                              const SurfelOptions& opt, int tiles_x, int tiles_y,
                              const SplatRecords& out, cudaStream_t stream) {
  if (in.n == 0) return cudaSuccess;
  int blocks = (in.n + 255) / 256;
  if (in.attr_dtype == DType::F16)
    preprocess_kernel<__half><<<blocks, 256, 0, stream>>>(in, cam, opt, tiles_x, tiles_y, out);
  else
    preprocess_kernel<float><<<blocks, 256, 0, stream>>>(in, cam, opt, tiles_x, tiles_y, out);
  return cudaGetLastError();
}

}  // namespace surfel_raster
