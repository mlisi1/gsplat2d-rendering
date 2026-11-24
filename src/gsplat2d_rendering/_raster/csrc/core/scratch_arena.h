// SPDX-License-Identifier: Apache-2.0
//
// A growing device-memory arena for C++ callers of surfel_render (the
// torch binding uses torch's caching allocator instead). After a frame or
// two it settles into one block and per-frame allocation is a pointer bump.
//
//   surfel_raster::ScratchArena arena;
//   arena.reset();
//   surfel_render(in, cam, opt, out, ScratchArena::alloc, &arena, stream);
#pragma once

#include <cstddef>
#include <vector>

namespace surfel_raster {

class ScratchArena {
 public:
  ScratchArena() = default;
  ScratchArena(const ScratchArena&) = delete;
  ScratchArena& operator=(const ScratchArena&) = delete;
  ~ScratchArena();

  // Call before each surfel_render, once the previous frame's kernels no
  // longer need their scratch (e.g. after the stream synchronized). Folds
  // any overflow from the last frame into one larger block.
  void reset();

  // ScratchAlloc-compatible: ctx is the ScratchArena*.
  static void* alloc(void* ctx, size_t bytes);

  size_t capacity() const { return cap_; }

 private:
  void* bump(size_t bytes);

  char* base_ = nullptr;
  size_t cap_ = 0;
  size_t used_ = 0;
  std::vector<void*> overflow_;
  size_t overflow_bytes_ = 0;
};

}  // namespace surfel_raster
