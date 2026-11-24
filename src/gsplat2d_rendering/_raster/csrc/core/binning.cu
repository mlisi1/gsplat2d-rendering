// SPDX-License-Identifier: Apache-2.0
//
// Splat-tile pair emission and per-tile ranges over the sorted pairs. The
// key is (tile_id << 32) | float_bits(p.z), so one radix sort orders pairs
// by tile, then front to back within a tile.
#include "preprocess.cuh"

namespace surfel_raster {
namespace {

__global__ void emit_kernel(SplatRecords rec, const long long* ends, int n, int tiles_x,
                            int width, unsigned long long* keys, uint32_t* vals) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long cnt = rec.count[i];
  if (cnt == 0) return;
  long long k = ends[i] - cnt;
  unsigned long long z = rec.zkey[i];
  ExactConic e = load_conic(rec.conic, i);
  for_each_tile(rec.rect[i], e, width, [&](int tx, int ty) {
    keys[k] = ((unsigned long long)(ty * tiles_x + tx) << 32) | z;
    vals[k] = (uint32_t)i;
    ++k;
  });
}

__global__ void ranges_kernel(const unsigned long long* keys, long long num_pairs, uint2* ranges) {
  long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_pairs) return;
  uint32_t tile = (uint32_t)(keys[k] >> 32);
  if (k == 0 || (uint32_t)(keys[k - 1] >> 32) != tile) ranges[tile].x = (uint32_t)k;
  if (k == num_pairs - 1 || (uint32_t)(keys[k + 1] >> 32) != tile) ranges[tile].y = (uint32_t)(k + 1);
}

}  // namespace

cudaError_t launch_emit(const SplatRecords& rec, const long long* ends, int n, int tiles_x,
                        int width, unsigned long long* keys, uint32_t* vals, cudaStream_t stream) {
  if (n == 0) return cudaSuccess;
  emit_kernel<<<(n + 255) / 256, 256, 0, stream>>>(rec, ends, n, tiles_x, width, keys, vals);
  return cudaGetLastError();
}

cudaError_t launch_ranges(const unsigned long long* keys, long long num_pairs, uint2* ranges,
                          cudaStream_t stream) {
  if (num_pairs == 0) return cudaSuccess;
  unsigned blocks = (unsigned)((num_pairs + 255) / 256);
  ranges_kernel<<<blocks, 256, 0, stream>>>(keys, num_pairs, ranges);
  return cudaGetLastError();
}

}  // namespace surfel_raster
