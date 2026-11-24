// SPDX-License-Identifier: Apache-2.0
//
// Tile coverage shared by the preprocess (count) and binning (emit)
// kernels. Both MUST go through for_each_tile: if they disagreed, the scan
// offsets would not match the pairs written and emit would overrun.
#pragma once

#include "math.cuh"

namespace surfel_raster {

// Centred ellipse {c + X : X^T M^-1 X <= 1} in pixel coordinates.
struct ExactConic {
  float xc, yc, m12, m22, hx, det, pad;
  bool valid;
};

constexpr int kConicFloats = 8;

__device__ __forceinline__ void store_conic(float* buf, int i, const ExactConic& e) {
  float4* b = reinterpret_cast<float4*>(buf) + 2 * i;
  b[0] = make_float4(e.xc, e.yc, e.m12, e.m22);
  b[1] = make_float4(e.hx, e.det, e.pad, e.valid ? 1.f : 0.f);
}

__device__ __forceinline__ ExactConic load_conic(const float* buf, int i) {
  ExactConic e;
  e.valid = false;
  if (!buf) return e;
  const float4* b = reinterpret_cast<const float4*>(buf) + 2 * i;
  float4 a = b[0], c = b[1];
  e.xc = a.x; e.yc = a.y; e.m12 = a.z; e.m22 = a.w;
  e.hx = c.x; e.det = c.y; e.pad = c.z; e.valid = c.w != 0.f;
  return e;
}

// Tile columns [tx0, tx1] of tile row ty that the ellipse touches. The
// ellipse is convex, so over a band of rows its leftmost point is either
// at one end of the band or the global leftmost point (if that lies
// inside the band); likewise on the right.
__device__ __forceinline__ bool row_span(const ExactConic& e, const int4& rect, int ty, int width,
                                         int& tx0, int& tx1) {
  tx0 = rect.x;
  tx1 = rect.z;
  if (!(e.m22 > 1e-12f) || !(e.hx > 1e-6f)) return true;  // degenerate: keep the box row
  float hy = sqrtf(e.m22);
  float ya = fmaxf(ty * kTile + 0.5f - e.yc - e.pad, -hy);
  float yb = fminf(ty * kTile + (kTile - 1) + 0.5f - e.yc + e.pad, hy);
  if (ya > yb) return false;
  auto half = [&](float y) { return sqrtf(fmaxf(e.det * (e.m22 - y * y), 0.f)); };
  float inv = 1.f / e.m22;
  float l = fminf((e.m12 * ya - half(ya)) * inv, (e.m12 * yb - half(yb)) * inv);
  float r = fmaxf((e.m12 * ya + half(ya)) * inv, (e.m12 * yb + half(yb)) * inv);
  float y_left = -e.m12 / e.hx;
  if (y_left >= ya && y_left <= yb) l = -e.hx;
  if (-y_left >= ya && -y_left <= yb) r = e.hx;
  float x0 = ceilf(e.xc + l - 0.5f - e.pad), x1 = floorf(e.xc + r - 0.5f + e.pad);
  x0 = fmaxf(x0, 0.f);
  x1 = fminf(x1, (float)(width - 1));
  if (x0 > x1) return false;
  tx0 = max((int)x0 / kTile, rect.x);
  tx1 = min((int)x1 / kTile, rect.z);
  return tx0 <= tx1;
}

template <typename F>
__device__ __forceinline__ void for_each_tile(const int4& rect, const ExactConic& e, int width,
                                              F f) {
  for (int ty = rect.y; ty <= rect.w; ++ty) {
    int tx0 = rect.x, tx1 = rect.z;
    if (e.valid && !row_span(e, rect, ty, width, tx0, tx1)) continue;
    for (int tx = tx0; tx <= tx1; ++tx) f(tx, ty);
  }
}

__device__ __forceinline__ long long count_tiles(const int4& rect, const ExactConic& e, int width) {
  if (!e.valid) return (long long)(rect.z - rect.x + 1) * (rect.w - rect.y + 1);
  long long n = 0;
  for (int ty = rect.y; ty <= rect.w; ++ty) {
    int tx0, tx1;
    if (row_span(e, rect, ty, width, tx0, tx1)) n += tx1 - tx0 + 1;
  }
  return n;
}

cudaError_t launch_preprocess(const SurfelInputs& in, const SurfelCamera& cam,
                              const SurfelOptions& opt, int tiles_x, int tiles_y,
                              const SplatRecords& out, cudaStream_t stream);

cudaError_t launch_emit(const SplatRecords& rec, const long long* ends, int n, int tiles_x,
                        int width, unsigned long long* keys, uint32_t* vals, cudaStream_t stream);

cudaError_t launch_ranges(const unsigned long long* keys, long long num_pairs, uint2* ranges,
                          cudaStream_t stream);

cudaError_t launch_raster(const SplatRecords& rec, const uint32_t* vals, const uint2* ranges,
                          const SurfelCamera& cam, const SurfelOptions& opt,
                          const SurfelOutputs& out, int tiles_x, int tiles_y, cudaStream_t stream);

}  // namespace surfel_raster
