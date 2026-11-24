// SPDX-License-Identifier: Apache-2.0
//
// The real-SH constants below are from PlenOctree (BSD-2-Clause, copyright
// 2021 The PlenOctree Authors; see NOTICE), the same ones as sh.py.
#pragma once

#include <cuda_fp16.h>

#include "surfel_raster.h"

namespace surfel_raster {

constexpr int kTile = 16;
constexpr int kBlock = kTile * kTile;

struct Vec3 {
  float x, y, z;
};

__device__ __forceinline__ Vec3 v3(float x, float y, float z) { return {x, y, z}; }
__device__ __forceinline__ float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__device__ __forceinline__ Vec3 scale(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
__device__ __forceinline__ Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }

template <typename T>
__device__ __forceinline__ float ld(const T* p, long long i);
template <>
__device__ __forceinline__ float ld<float>(const float* p, long long i) { return p[i]; }
template <>
__device__ __forceinline__ float ld<__half>(const __half* p, long long i) {
  return __half2float(p[i]);
}

// Columns of the rotation matrix of a (w,x,y,z) quaternion, normalized here.
__device__ __forceinline__ void quat_columns(float w, float x, float y, float z, Vec3& c0,
                                             Vec3& c1) {
  float inv = rsqrtf(w * w + x * x + y * y + z * z);
  w *= inv; x *= inv; y *= inv; z *= inv;
  c0 = {1.f - 2.f * (y * y + z * z), 2.f * (x * y + w * z), 2.f * (x * z - w * y)};
  c1 = {2.f * (x * y - w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + w * x)};
}

constexpr float kSH0 = 0.28209479177387814f;
constexpr float kSH1 = 0.4886025119029199f;
__device__ constexpr float kSH2[5] = {1.0925484305920792f, -1.0925484305920792f,
                                      0.31539156525252005f, -1.0925484305920792f,
                                      0.5462742152960396f};
__device__ constexpr float kSH3[7] = {-0.5900435899266435f, 2.890611442640554f,
                                      -0.4570457994644658f, 0.3731763325901154f,
                                      -0.4570457994644658f, 1.445305721320277f,
                                      -0.5900435899266435f};

// coef(k) returns the RGB coefficient k; dir is a unit vector.
template <typename Coef>
__device__ __forceinline__ Vec3 eval_sh(int degree, Vec3 dir, Coef coef) {
  Vec3 r = scale(coef(0), kSH0);
  if (degree < 1) return r;
  float x = dir.x, y = dir.y, z = dir.z;
  r = add(r, scale(coef(1), -kSH1 * y));
  r = add(r, scale(coef(2), kSH1 * z));
  r = add(r, scale(coef(3), -kSH1 * x));
  if (degree < 2) return r;
  float xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
  r = add(r, scale(coef(4), kSH2[0] * xy));
  r = add(r, scale(coef(5), kSH2[1] * yz));
  r = add(r, scale(coef(6), kSH2[2] * (2.f * zz - xx - yy)));
  r = add(r, scale(coef(7), kSH2[3] * xz));
  r = add(r, scale(coef(8), kSH2[4] * (xx - yy)));
  if (degree < 3) return r;
  r = add(r, scale(coef(9), kSH3[0] * y * (3.f * xx - yy)));
  r = add(r, scale(coef(10), kSH3[1] * xy * z));
  r = add(r, scale(coef(11), kSH3[2] * y * (4.f * zz - xx - yy)));
  r = add(r, scale(coef(12), kSH3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy)));
  r = add(r, scale(coef(13), kSH3[4] * x * (4.f * zz - xx - yy)));
  r = add(r, scale(coef(14), kSH3[5] * z * (xx - yy)));
  r = add(r, scale(coef(15), kSH3[6] * x * (xx - 3.f * yy)));
  return r;
}

// Per-splat data the raster stage reads: 4 float4 = 64 bytes.
//   r0 = (a_u, b_u), r1 = (a_v, b_v), r2 = (n, c), r3 = (rgb, opacity)
// plus, only on the low-pass path, (centre_px.x, centre_px.y, p.z, 0).
struct SplatRecords {
  float4* rec;     // [4N]
  float4* blob;    // [N], low-pass only
  uint32_t* zkey;  // [N] float bits of p.z (positive, so bits sort like values)
  int4* rect;      // [N] inclusive tile rectangle
  float* conic;    // [8N] centred ellipse for exact tiles, see preprocess.cuh
  long long* count;  // [N] tiles touched
};

}  // namespace surfel_raster
