// SPDX-License-Identifier: Apache-2.0
#include "scratch_arena.h"

#include <cuda_runtime.h>

namespace surfel_raster {
namespace {
constexpr size_t kAlign = 256;
size_t align_up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }
}  // namespace

ScratchArena::~ScratchArena() {
  for (void* p : overflow_) cudaFree(p);
  if (base_) cudaFree(base_);
}

void ScratchArena::reset() {
  if (!overflow_.empty()) {
    size_t want = cap_ + overflow_bytes_;
    for (void* p : overflow_) cudaFree(p);
    overflow_.clear();
    overflow_bytes_ = 0;
    if (base_) cudaFree(base_);
    base_ = nullptr;
    cap_ = 0;
    void* p = nullptr;
    if (cudaMalloc(&p, want) == cudaSuccess) {
      base_ = static_cast<char*>(p);
      cap_ = want;
    }
  }
  used_ = 0;
}

void* ScratchArena::bump(size_t bytes) {
  bytes = align_up(bytes);
  if (used_ + bytes <= cap_) {
    void* p = base_ + used_;
    used_ += bytes;
    return p;
  }
  void* p = nullptr;
  if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
  overflow_.push_back(p);
  overflow_bytes_ += bytes;
  return p;
}

void* ScratchArena::alloc(void* ctx, size_t bytes) {
  return static_cast<ScratchArena*>(ctx)->bump(bytes);
}

}  // namespace surfel_raster
