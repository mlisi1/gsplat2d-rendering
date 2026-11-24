// SPDX-License-Identifier: Apache-2.0
//
// surfel_render: preprocess -> scan -> emit -> sort -> ranges -> raster.
#include <climits>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include "preprocess.cuh"

namespace surfel_raster {
namespace {

struct Scratch {
  ScratchAlloc alloc;
  void* ctx;
  bool failed = false;
  template <typename T>
  T* get(size_t count) {
    if (count == 0) count = 1;
    void* p = alloc(ctx, count * sizeof(T));
    if (!p) failed = true;
    return static_cast<T*>(p);
  }
};

int bit_width(unsigned v) {
  int b = 0;
  while (v) {
    ++b;
    v >>= 1;
  }
  return b;
}

bool valid_args(const SurfelInputs& in, const SurfelCamera& cam, const SurfelOptions& opt,
                const SurfelOutputs& out) {
  if (in.n < 0 || cam.width <= 0 || cam.height <= 0) return false;
  if (in.n > 0 && (!in.means || !in.quats || !in.log_scales || !in.opacity_logits ||
                   !in.features_dc))
    return false;
  if (in.sh_degree > 3 || in.sh_degree < -1) return false;
  if (in.sh_degree > 0 && (in.n > 0 && !in.features_rest)) return false;
  if (in.sh_degree > 0 && in.rest_stride < (in.sh_degree + 1) * (in.sh_degree + 1) - 1)
    return false;
  if (!(cam.fx > 0.f) || !(cam.fy > 0.f) || !(cam.near_z > 0.f)) return false;
  if (!(opt.alpha_min > 0.f) || !(opt.alpha_max <= 1.f)) return false;
  if (!out.rgb || !out.depth || !out.alpha) return false;
  if (opt.aux && (!out.normal || !out.median_depth)) return false;
  return true;
}

}  // namespace

#define SR_CUDA(expr)                       \
  do {                                      \
    if ((expr) != cudaSuccess) return kErrCuda; \
  } while (0)
#define SR_ALLOC(s)                                   \
  do {                                                \
    if ((s).failed) return kErrOutOfMemory;           \
  } while (0)

long long surfel_render(const SurfelInputs& in, const SurfelCamera& cam,
                        const SurfelOptions& opt, const SurfelOutputs& out, ScratchAlloc alloc,
                        void* alloc_ctx, cudaStream_t stream) {
  if (!valid_args(in, cam, opt, out)) return kErrInvalidArgument;
  Scratch scratch{alloc, alloc_ctx};
  int tiles_x = (cam.width + kTile - 1) / kTile, tiles_y = (cam.height + kTile - 1) / kTile;
  int num_tiles = tiles_x * tiles_y;
  int n = in.n;

  SplatRecords rec;
  rec.rec = scratch.get<float4>(4ull * n);
  rec.blob = opt.lowpass ? scratch.get<float4>(n) : nullptr;
  rec.zkey = scratch.get<uint32_t>(n);
  rec.rect = scratch.get<int4>(n);
  rec.conic = opt.exact_tiles ? scratch.get<float>((size_t)kConicFloats * n) : nullptr;
  rec.count = scratch.get<long long>(n);
  long long* ends = scratch.get<long long>(n);
  uint2* ranges = scratch.get<uint2>(num_tiles);
  SR_ALLOC(scratch);

  SR_CUDA(launch_preprocess(in, cam, opt, tiles_x, tiles_y, rec, stream));

  long long num_pairs = 0;
  if (n > 0) {
    size_t temp_bytes = 0;
    SR_CUDA(cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, rec.count, ends, n, stream));
    void* temp = scratch.get<char>(temp_bytes);
    SR_ALLOC(scratch);
    SR_CUDA(cub::DeviceScan::InclusiveSum(temp, temp_bytes, rec.count, ends, n, stream));
    SR_CUDA(cudaMemcpyAsync(&num_pairs, ends + (n - 1), sizeof(long long),
                            cudaMemcpyDeviceToHost, stream));
    SR_CUDA(cudaStreamSynchronize(stream));
  }
  if (num_pairs > INT_MAX) return kErrTooManyPairs;

  SR_CUDA(cudaMemsetAsync(ranges, 0, sizeof(uint2) * num_tiles, stream));
  uint32_t* sorted_vals = nullptr;
  if (num_pairs > 0) {
    unsigned long long* keys = scratch.get<unsigned long long>(num_pairs);
    unsigned long long* keys_alt = scratch.get<unsigned long long>(num_pairs);
    uint32_t* vals = scratch.get<uint32_t>(num_pairs);
    uint32_t* vals_alt = scratch.get<uint32_t>(num_pairs);
    SR_ALLOC(scratch);
    SR_CUDA(launch_emit(rec, ends, n, tiles_x, cam.width, keys, vals, stream));

    cub::DoubleBuffer<unsigned long long> kb(keys, keys_alt);
    cub::DoubleBuffer<uint32_t> vb(vals, vals_alt);
    int end_bit = 32 + bit_width((unsigned)num_tiles);
    size_t sort_bytes = 0;
    SR_CUDA(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, kb, vb, (int)num_pairs, 0,
                                            end_bit, stream));
    void* sort_temp = scratch.get<char>(sort_bytes);
    SR_ALLOC(scratch);
    SR_CUDA(cub::DeviceRadixSort::SortPairs(sort_temp, sort_bytes, kb, vb, (int)num_pairs, 0,
                                            end_bit, stream));
    SR_CUDA(launch_ranges(kb.Current(), num_pairs, ranges, stream));
    sorted_vals = vb.Current();
  }

  SR_CUDA(launch_raster(rec, sorted_vals, ranges, cam, opt, out, tiles_x, tiles_y, stream));
  return num_pairs;
}

const char* surfel_error_string(long long code) {
  switch (code) {
    case kErrInvalidArgument: return "invalid argument";
    case kErrCuda: return "CUDA error";
    case kErrOutOfMemory: return "scratch allocation failed";
    case kErrTooManyPairs: return "more than INT_MAX splat-tile pairs";
    default: return code >= 0 ? "ok" : "unknown error";
  }
}

}  // namespace surfel_raster
