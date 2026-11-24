// SPDX-License-Identifier: Apache-2.0
//
// Render-only 2D Gaussian surfel rasterizer. No torch dependency: callers
// pass device pointers, a CUDA stream and a scratch allocator, so the same
// core serves the Python binding and C++ users (e.g. Gazebo plugins).
// The math is documented in docs/rasterizer.md.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace surfel_raster {

enum class DType : int { F32 = 0, F16 = 1 };

struct SurfelInputs {
  int n = 0;
  const float* means = nullptr;          // [N,3] world, always float32
  const void* quats = nullptr;           // [N,4] (w,x,y,z), unnormalized
  const void* log_scales = nullptr;      // [N,2]
  const void* opacity_logits = nullptr;  // [N]
  const void* features_dc = nullptr;     // [N,3]: SH DC, or final RGB when sh_degree < 0
  const void* features_rest = nullptr;   // [N,rest_stride,3]; may be null when sh_degree <= 0
  int rest_stride = 0;                   // SH coefficients stored per splat beyond DC
  DType attr_dtype = DType::F32;         // dtype of every attribute except means
  int sh_degree = 0;                     // 0..3, or -1 for precomputed RGB
};

struct SurfelCamera {
  float w2c[12];         // row-major 3x4, x_cam = R x_world + t; OpenCV axes
  float fx, fy, cx, cy;  // pixels; pixel (i,j) centre is (i+0.5, j+0.5)
  int width, height;
  float near_z;          // camera-space z cut
  float campos[3];       // world camera centre, for SH view directions
};

struct SurfelOptions {
  bool lowpass = false;      // 2DGS screen-space low-pass
  bool aux = false;          // normal + median depth outputs
  bool exact_tiles = false;  // per-tile-row conic test instead of the bounding box
  float alpha_min = 1.f / 255.f;
  float alpha_max = 0.99f;
  float t_min = 1e-4f;
  float background[3] = {0.f, 0.f, 0.f};
};

struct SurfelOutputs {        // caller-allocated device buffers
  float* rgb = nullptr;           // [3,H,W] planar, background composited
  float* depth = nullptr;         // [H,W] expected z, 0 where alpha == 0
  float* alpha = nullptr;         // [H,W]
  float* normal = nullptr;        // [3,H,W] camera space; required iff aux
  float* median_depth = nullptr;  // [H,W]; required iff aux
  uint8_t* rgb8 = nullptr;        // optional [H,W,3] interleaved
};

// Returns device memory valid until surfel_render returns, or nullptr on
// failure. The caller frees it afterwards (or recycles it, see ScratchArena).
using ScratchAlloc = void* (*)(void* ctx, size_t bytes);

enum Error : long long {
  kErrInvalidArgument = -1,
  kErrCuda = -2,
  kErrOutOfMemory = -3,
  kErrTooManyPairs = -4,
};

// Returns the number of splat-tile pairs rendered (>= 0), or an Error.
// Synchronizes the stream once, to size the pair buffers.
long long surfel_render(const SurfelInputs& in, const SurfelCamera& cam,
                        const SurfelOptions& opt, const SurfelOutputs& out,
                        ScratchAlloc alloc, void* alloc_ctx, cudaStream_t stream);

const char* surfel_error_string(long long code);

}  // namespace surfel_raster
